#!/usr/bin/env python3
"""
Rationale (verified): The cloud variant is the easy path, but not everyone
wants to pay or hand over a card. This runs the same training on the user's
own GPU. bootstrap_windows.ps1 has already installed ai-toolkit and gated the
hardware by the time this runs, so this script's job is the pipeline itself:
photos, captions, config, training, and previews.

What it does:
  1. Loads settings, or runs the first-time wizard (no RunPod key needed)
  2. Reads the hardware report the bootstrap wrote and picks a VRAM tier
  3. Prepares photos and captions them with a local vision model
  4. Generates a tier-appropriate ai-toolkit config
  5. Runs training as a subprocess, translating tqdm output into plain English
  6. Checks the first checkpoint for the all-zero LoRA bug (issue #925) and
     stops immediately if it hits, rather than wasting another hour
  7. Renders preview images afterwards on tiers that cannot sample during
     training
  8. Copies the results into the output folder

Usage:
  Double-click "2 - TRAIN ON MY PC.bat" in the parent folder.
  python train_local.py --steps 1500 --no-preview

Maintenance: The low-VRAM configs live in _configs.py, not here. Sampling is
deliberately disabled during training below 28GB: it is the step that
reassembles the full model, and cache_text_embeddings has already discarded
the text encoder by then. See build_preview_config for how previews are
produced instead.
"""
from __future__ import annotations

# =============================================================================
#  SETTINGS - you can fill these in if you would rather not answer questions.
#  Leaving them blank is fine: the program will just ask you once and
#  remember. Keep the quotes.
# =============================================================================

MY_HUGGINGFACE_TOKEN = ""
MY_CHARACTER_NAME = ""

# =============================================================================
#  Nothing below here needs changing.
# =============================================================================

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import _shared as sh
from _shared import UserFacingError
import _configs

# Default lower than the cloud's 2000: on a 16GB card every step costs real
# minutes, and 1500 already gives a strong likeness with 25-40 photos.
DEFAULT_LOCAL_STEPS = 1500
SAVE_EVERY = 250


def ai_toolkit_python() -> Path:
    """Locate the interpreter inside ai-toolkit's own environment."""
    for relative in ("Scripts/python.exe", "bin/python", "bin/python3"):
        candidate = sh.AI_TOOLKIT_DIR / ".venv" / relative
        if candidate.exists():
            return candidate
    raise UserFacingError(
        "The training software is not installed yet.\n\n"
        "  Close this window and double-click \"2 - TRAIN ON MY PC.bat\"\n"
        "  again - it will install everything the first time."
    )


def read_hardware() -> dict:
    """Read the hardware report written by bootstrap_windows.ps1.

    Falls back to asking the GPU directly, so this script still works when
    run straight from a terminal on Linux or macOS.
    """
    report = sh.WORK_DIR / "hardware.json"
    if report.exists():
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
            if data.get("vram_gb"):
                return data
        except Exception:
            pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            best = None
            for line in result.stdout.strip().splitlines():
                parts = line.split(",")
                if len(parts) < 2:
                    continue
                gb = round(int(parts[1].strip()) / 1024.0, 1)
                if best is None or gb > best["vram_gb"]:
                    best = {"gpu_name": parts[0].strip(), "vram_gb": gb}
            if best:
                return best
    except Exception:
        pass

    raise UserFacingError(
        "I could not find an NVIDIA graphics card on this PC.\n\n"
        "  Training on your own machine needs one with at least 16GB.\n\n"
        "  You can still train in the cloud for about $3 - it works on\n"
        "  any computer. Double-click \"1 - TRAIN IN THE CLOUD.bat\"."
    )


def caption_photos(python: Path, dataset_dir: Path, trigger_word: str, env: dict) -> None:
    """Describe each photo with a local vision model.

    Never fatal. If the captioner cannot run, simple template captions still
    train a perfectly usable character LoRA - the user just gets less control
    over pose and setting in their prompts later.
    """
    script = sh.TRAINER_DIR / "caption_dataset.py"
    if not script.exists():
        sh.warn("Captioning tool missing; using simple captions.")
        sh.write_fallback_captions(dataset_dir, trigger_word)
        return

    sh.step("Describing your photos (a few minutes, downloads a small model)...")
    log_path = sh.WORK_DIR / "captioning_log.txt"
    try:
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [str(python), str(script),
                 "--dataset", str(dataset_dir),
                 "--trigger", trigger_word],
                stdout=log, stderr=subprocess.STDOUT,
                env=env, timeout=3600,
            )
        code = result.returncode
    except Exception as exc:
        sh.warn(f"Captioning could not run ({exc}).")
        code = 1

    if code == 0:
        sh.ok("Photos described.")
    else:
        sh.warn("Automatic descriptions failed, using simple captions instead.")
        sh.step(f"Training still works. Details: {log_path.name}")

    sh.write_fallback_captions(dataset_dir, trigger_word)
    sh.ensure_trigger_in_captions(dataset_dir, trigger_word)


def find_checkpoints(train_output: Path) -> list[Path]:
    if not train_output.exists():
        return []
    return sorted(train_output.rglob("*.safetensors"))


def is_hidden(path: Path) -> bool:
    """True for anything inside a dot-directory.

    ai-toolkit mirrors every sample into a hidden samples/.thumbs/ folder,
    naming the copies "<original>.jpg.jpg". Those are thumbnails, not
    results, and collecting them hands the user two files per picture.
    """
    return any(part.startswith(".") for part in path.parts)


def preview_images(source: Path):
    for image in source.rglob("*"):
        if image.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            continue
        if is_hidden(image.relative_to(source)):
            continue
        yield image


def run_training(
    python: Path, config_path: Path, env: dict, steps: int, train_output: Path,
) -> str:
    """Run ai-toolkit and translate its output into something readable.

    Returns 'done', 'dead' or 'failed'.
    """
    process = subprocess.Popen(
        [str(python), "run.py", str(config_path)],
        cwd=str(sh.AI_TOOLKIT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        bufsize=0,
    )

    printer = sh.ProgressPrinter(interval=30)
    log_path = sh.OUTPUT_DIR / "training_log.txt"
    sh.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    buffer = ""
    recent = ""
    collector = IncrementalCollector()
    checked_first_checkpoint = False
    announced_download = False
    started = time.time()
    verdict = "done"

    sh.say()
    sh.say("  Snapshots and preview pictures are copied into the 'output'")
    sh.say("  folder as soon as each one is ready, so you can look at them")
    sh.say("  while the rest carries on.")
    sh.say()

    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        assert process.stdout is not None
        while True:
            chunk = process.stdout.read(4096)
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            log.write(text)
            log.flush()

            # tqdm redraws with carriage returns, so treat \r as a line break
            # or the whole progress bar arrives as one enormous line.
            buffer += text.replace("\r", "\n")
            if "\n" in buffer:
                lines = buffer.split("\n")
                buffer = lines[-1]
                recent = "\n".join((recent + "\n" + "\n".join(lines[:-1])).splitlines()[-40:])

            progress = sh.parse_progress(recent, steps)
            if progress["step"] is None:
                if not announced_download and time.time() - started > 90:
                    announced_download = True
                    sh.say()
                    sh.say("  Downloading the AI model (about 35 GB).")
                    sh.say("  The first time, this can take an hour or more.")
                    sh.say("  Nothing is wrong - please leave it running.")
                    sh.say()
            else:
                printer.maybe_print(progress)

            if not checked_first_checkpoint:
                checkpoints = find_checkpoints(train_output)
                if checkpoints:
                    checked_first_checkpoint = True
                    if not check_first_checkpoint(checkpoints[0]):
                        verdict = "dead"
                        try:
                            process.terminate()
                            process.wait(timeout=60)
                        except Exception:
                            try:
                                process.kill()
                            except Exception:
                                pass
                        break

            # Copy snapshots and previews out while training continues, so
            # the user can compare step 250 against step 500 without waiting
            # for the whole run. Skipped until the dead-LoRA gate passes.
            if checked_first_checkpoint:
                collector.poll(train_output, preview_root=None)

    code = process.wait()
    if verdict == "dead":
        return "dead"
    if code != 0:
        sh.say()
        sh.warn(f"Training stopped with an error (code {code}).")
        explain_failure(recent)
        return "failed"
    return "done"


def check_first_checkpoint(path: Path) -> bool:
    """Confirm the first saved LoRA is not the known all-zero failure."""
    sh.say()
    sh.step("First snapshot saved. Checking it actually learned something...")
    # ai-toolkit may still be flushing the file.
    for _ in range(10):
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        if size > 1024:
            break
        time.sleep(3)

    try:
        alive, message = sh.verify_checkpoint_alive(path)
    except Exception as exc:
        sh.warn(f"Could not check the snapshot ({exc}). Carrying on anyway.")
        return True

    if alive:
        sh.ok(f"Looks good. {message}")
        sh.say()
        return True

    sh.say()
    sh.problem(f"The snapshot is empty: {message}")
    sh.say(sh.DEAD_LORA_EXPLANATION)
    return False


def explain_failure(recent: str) -> None:
    """Turn the most common crashes into something actionable."""
    lowered = recent.lower()
    sh.say()
    if "out of memory" in lowered or "cuda oom" in lowered:
        sh.say("  WHAT HAPPENED: your graphics card ran out of memory.")
        sh.say()
        sh.say("  Try these, in order:")
        sh.say("    1. Close everything else - games, Chrome, Discord, OBS.")
        sh.say("       A browser can quietly hold on to 2GB of your card.")
        sh.say("    2. Restart your PC and run this again before opening")
        sh.say("       anything else.")
        sh.say("    3. Use the cloud instead. It has far more memory and")
        sh.say("       costs about $3:  1 - TRAIN IN THE CLOUD.bat")
    elif "gated" in lowered or "401" in lowered or "403" in lowered:
        sh.say("  WHAT HAPPENED: HuggingFace refused to give us the model.")
        sh.say()
        sh.say("  Your account probably has not accepted the licence yet:")
        sh.say(f"    {sh.LICENSE_URL}")
        sh.say()
        sh.say("  Accept it, then delete my_settings.json and run this again.")
    elif "no space left" in lowered or "disk" in lowered and "full" in lowered:
        sh.say("  WHAT HAPPENED: your disk filled up.")
        sh.say()
        sh.say("  The AI model needs about 35GB, plus room to work.")
        sh.say("  Free up some space and try again.")
    elif "connectionerror" in lowered or "timed out" in lowered or "max retries" in lowered:
        sh.say("  WHAT HAPPENED: the download was interrupted.")
        sh.say()
        sh.say("  Just run this again - it picks up where it left off.")
    else:
        sh.say("  The last few lines from the training log:")
        for line in recent.strip().splitlines()[-15:]:
            if line.strip():
                sh.say("    " + line.strip())
        sh.say()
        sh.say("  The full log is in output\\training_log.txt")
        sh.say("  If you are stuck, send a screenshot of this window.")
    sh.say()


def generate_previews(
    python: Path, env: dict, run_name: str, trigger_word: str,
    lora_path: Path, tier: _configs.Tier, train_output: Path,
) -> Path | None:
    """Render sample images from the finished LoRA.

    A separate pass because sampling cannot run alongside low-VRAM training.
    Entirely optional - the LoRA is the deliverable, previews are a nicety,
    so every failure here is swallowed.
    """
    sh.say()
    sh.step("Making a few preview pictures so you can see how it turned out.")
    sh.step("This takes about 10 minutes. The LoRA is already saved and safe.")

    config_path = sh.WORK_DIR / "preview_config.yaml"
    config_path.write_text(
        _configs.build_preview_config(
            run_name=run_name,
            trigger_word=trigger_word,
            dataset_dir=str(sh.DATASET_DIR),
            output_dir=str(train_output.parent),
            lora_path=str(lora_path),
            tier=tier,
        ),
        encoding="utf-8",
    )

    log_path = sh.WORK_DIR / "preview_log.txt"
    try:
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(
                [str(python), "run.py", str(config_path)],
                cwd=str(sh.AI_TOOLKIT_DIR),
                stdout=log, stderr=subprocess.STDOUT,
                env=env, timeout=3600,
            )
    except Exception as exc:
        sh.warn(f"Preview pictures could not be made ({exc}).")
        return None

    preview_root = train_output.parent / f"{run_name}_preview"
    samples = sorted(preview_root.rglob("*.jpg")) + sorted(preview_root.rglob("*.png"))
    if not samples:
        sh.warn("No preview pictures were produced. Your LoRA is still fine.")
        sh.step(f"Details are in work\\{log_path.name}")
        return None
    return preview_root


class IncrementalCollector:
    """Copies snapshots and previews into output/ as training produces them.

    Same reasoning as the cloud variant: nobody wants to stare at an empty
    folder for an hour and then receive four files at once. Locally the copy
    is instant, but ai-toolkit still writes a checkpoint progressively, so
    the same settle-before-copy rule applies.
    """

    def __init__(self) -> None:
        self.tracker = sh.SettleTracker()

    def poll(self, train_output: Path, preview_root: Path | None) -> int:
        sh.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        landed = 0

        for checkpoint in find_checkpoints(train_output):
            key = str(checkpoint)
            if self.tracker.is_done(key):
                continue
            try:
                size = checkpoint.stat().st_size
            except OSError:
                continue
            if not self.tracker.is_ready(key, size):
                continue
            destination = sh.OUTPUT_DIR / checkpoint.name
            if destination.exists() and destination.stat().st_size == size:
                self.tracker.mark_done(key)
                continue
            try:
                shutil.copy2(checkpoint, destination)
                self.tracker.mark_done(key)
                landed += 1
                sh.ok(f"  Saved {destination.name} ({sh.human_size(size)})")
                sh.say(f"     -> {sh.OUTPUT_DIR}")
            except Exception as exc:
                sh.warn(f"Could not copy {checkpoint.name}: {exc}")

        sources = [train_output]
        if preview_root is not None:
            sources.append(preview_root)
        preview_dir = sh.OUTPUT_DIR / "preview_images"
        for source in sources:
            if not source.exists():
                continue
            for image in preview_images(source):
                key = str(image)
                if self.tracker.is_done(key):
                    continue
                try:
                    size = image.stat().st_size
                except OSError:
                    continue
                if not self.tracker.is_ready(key, size):
                    continue
                destination = preview_dir / image.name
                if destination.exists() and destination.stat().st_size == size:
                    self.tracker.mark_done(key)
                    continue
                preview_dir.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(image, destination)
                    self.tracker.mark_done(key)
                    landed += 1
                    sh.ok(f"  Saved preview image {destination.name}")
                except Exception:
                    pass
        return landed


def collect_results(train_output: Path, preview_root: Path | None) -> None:
    """Final sweep. Idempotent, so anything already copied is left alone."""
    sh.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for checkpoint in find_checkpoints(train_output):
        destination = sh.OUTPUT_DIR / checkpoint.name
        if destination.exists() and destination.stat().st_size == checkpoint.stat().st_size:
            continue
        try:
            shutil.copy2(checkpoint, destination)
        except Exception as exc:
            sh.warn(f"Could not copy {checkpoint.name}: {exc}")

    sources = [train_output]
    if preview_root is not None:
        sources.append(preview_root)

    preview_dir = sh.OUTPUT_DIR / "preview_images"
    copied = 0
    for source in sources:
        if not source.exists():
            continue
        for image in preview_images(source):
            if "samples" not in str(image).lower():
                continue
            preview_dir.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(image, preview_dir / image.name)
                copied += 1
            except Exception:
                pass


def run(args: argparse.Namespace) -> int:
    sh.header("Krystal's LoRA Trainer - Your PC")

    settings = sh.load_or_ask(
        need_runpod=False,
        character_name=MY_CHARACTER_NAME,
        huggingface_token=MY_HUGGINGFACE_TOKEN,
    )
    steps = args.steps or DEFAULT_LOCAL_STEPS
    slug = sh.make_slug(settings.character_name)
    run_name = f"{slug}_krea2"

    python = ai_toolkit_python()
    hardware = read_hardware()
    vram = float(hardware.get("vram_gb", 0))
    tier = _configs.pick_tier(vram)

    sh.header("Step 1 of 4 - your PC")
    sh.say(f"  Graphics card:  {hardware.get('gpu_name', 'unknown')}")
    sh.say(f"  Card memory:    {vram} GB")
    sh.say(f"  Settings:       {tier.label}")
    sh.say(f"  {tier.note}")
    sh.say()

    if vram < _configs.VRAM_MINIMUM_GB:
        raise UserFacingError(
            f"Your graphics card has {vram}GB of memory, but at least 16GB\n"
            "  is needed.\n\n"
            "  You can still train in the cloud for about $3 - it works on\n"
            "  any computer. Double-click \"1 - TRAIN IN THE CLOUD.bat\"."
        )

    hours = (steps * (4.0 if tier.key == "16gb" else 2.5)) / 3600.0
    sh.say(f"  Training {steps} steps will take roughly {sh.human_duration(hours * 3600)},")
    sh.say("  plus up to an hour the first time for the model download.")
    sh.say()

    sh.header("Step 2 of 4 - your photos")
    sh.prepare_photos(sh.PHOTOS_DIR, sh.DATASET_DIR)

    env = dict(os.environ)
    env["HF_TOKEN"] = settings.huggingface_token
    env["HUGGING_FACE_HUB_TOKEN"] = settings.huggingface_token
    env["DISABLE_TELEMETRY"] = "YES"
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    # Keep the 35GB cache inside this folder so the disk check the bootstrap
    # ran is accurate, and so deleting the folder actually reclaims the space.
    env["HF_HOME"] = str(sh.ROOT / "model_cache")
    env.setdefault("PYTHONUNBUFFERED", "1")

    caption_photos(python, sh.DATASET_DIR, settings.trigger_word, env)

    sh.header("Step 3 of 4 - training")
    train_root = sh.WORK_DIR / "train_output"
    train_output = train_root / run_name
    train_root.mkdir(parents=True, exist_ok=True)

    config_path = sh.WORK_DIR / "training_config.yaml"
    config_path.write_text(
        _configs.build_training_config(
            run_name=run_name,
            trigger_word=settings.trigger_word,
            dataset_dir=str(sh.DATASET_DIR),
            output_dir=str(train_root),
            steps=steps,
            tier=tier,
            save_every=SAVE_EVERY,
        ),
        encoding="utf-8",
    )
    sh.ok(f"Settings written for a {tier.label}.")
    sh.say()
    sh.say("  Training is starting. You can leave this running and come back.")
    sh.say("  Do not play games or open heavy programs while it runs - they")
    sh.say("  compete for the same graphics memory and can crash it.")
    sh.say()

    result = run_training(python, config_path, env, steps, train_output)

    if result == "dead":
        collect_results(train_output, None)
        return 1

    checkpoints = find_checkpoints(train_output)
    if not checkpoints:
        if result == "failed":
            return 1
        raise UserFacingError(
            "Training finished but produced no LoRA file.\n\n"
            "  Check output\\training_log.txt, or just try the cloud version:\n"
            "      1 - TRAIN IN THE CLOUD.bat"
        )

    sh.header("Step 4 of 4 - finishing up")
    preview_root = None
    want_previews = not args.no_preview and not tier.sample_during_training
    if want_previews:
        preview_root = generate_previews(
            python, env, run_name, settings.trigger_word,
            checkpoints[-1], tier, train_output,
        )

    collect_results(train_output, preview_root)
    sh.summarise_results(sh.OUTPUT_DIR, slug)
    sh.say(f"  Type the word '{settings.trigger_word}' in a prompt to use it.")
    sh.say()
    if result == "failed":
        sh.warn("Training ended early, but the snapshots above were saved.")
        sh.say()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a Krea 2 character LoRA on your own GPU"
    )
    parser.add_argument("--steps", type=int, default=0, help="Override training steps")
    parser.add_argument("--no-preview", action="store_true",
                        help="Skip generating preview images afterwards")
    args = parser.parse_args()
    sh.disable_click_to_pause()

    try:
        return run(args)
    except UserFacingError as exc:
        sh.say()
        sh.say("=" * 66)
        sh.problem(str(exc))
        sh.say("=" * 66)
        sh.say()
        return 1
    except KeyboardInterrupt:
        sh.say()
        sh.warn("Stopped by you. Nothing is running and nothing is damaged.")
        sh.say("  Any snapshots already saved are in the output folder.")
        sh.say()
        return 130
    except Exception as exc:
        sh.say()
        sh.say("=" * 66)
        sh.problem(f"Something unexpected went wrong: {exc}")
        sh.say("=" * 66)
        sh.say()
        sh.say("  Please send a screenshot of this window.")
        sh.say()
        sh.say("  You can also try the cloud version, which avoids most")
        sh.say("  PC-specific problems:  1 - TRAIN IN THE CLOUD.bat")
        sh.say()
        return 1


if __name__ == "__main__":
    sys.exit(main())
