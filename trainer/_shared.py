"""
Rationale (verified): Shared plumbing for the two distributable Krea 2
character trainers (train_runpod.py and train_local.py).

This module deliberately depends only on `requests` and `pillow`. The
audience is a non-programmer on a Windows box who double-clicks a .bat file,
so every extra dependency is another way first-run setup can fail in front of
someone with no means to diagnose it. Anything heavier - a settings
framework, an API client, an SSH wrapper - is either inlined here or done
without.

What it does:
  1. ASCII-only, cp1252-safe console output helpers
  2. First-run interactive setup wizard, persisted to my_settings.json
  3. Live HuggingFace token + gated-repo access validation (plain HTTP,
     no huggingface_hub dependency)
  4. Photo intake: HEIC/EXIF handling, downscale, re-encode, validate
  5. Caption helpers (fallback captions + trigger-word injection)
  6. verify_checkpoint_alive(): guards ai-toolkit issue #925, where a Krea 2
     LoRA trains with a healthy loss curve but saves all-zero lora_B weights
  7. ai-toolkit tqdm log parsing into a friendly step count and ETA

Usage:
  Imported by train_runpod.py and train_local.py. Not run directly.

Maintenance: The ai-toolkit config keys referenced here and in _configs.py
were verified against toolkit/config_modules.py on main (Aug 2026). If
ai-toolkit changes its tqdm log format, update parse_progress(). If it
changes LoRA tensor naming away from lora_A/lora_B, update
verify_checkpoint_alive().
"""
from __future__ import annotations

import json
import os
import re
import shutil
import struct
import sys
import time
from statistics import median
import webbrowser
from array import array
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Iterable

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

TRAINER_DIR = Path(__file__).resolve().parent
ROOT = TRAINER_DIR.parent
PHOTOS_DIR = ROOT / "my_photos"
OUTPUT_DIR = ROOT / "output"
WORK_DIR = ROOT / "work"
DATASET_DIR = WORK_DIR / "dataset"
SETTINGS_FILE = ROOT / "my_settings.json"
AI_TOOLKIT_DIR = ROOT / "ai-toolkit"

BASE_MODEL = "krea/Krea-2-Raw"
LICENSE_URL = "https://huggingface.co/krea/Krea-2-Raw"
HF_TOKEN_URL = "https://huggingface.co/settings/tokens"
RUNPOD_KEY_URL = "https://console.runpod.io/user/settings"
RUNPOD_PODS_URL = "https://console.runpod.io/pods"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp",
    ".heic", ".heif", ".tif", ".tiff",
}
MIN_PHOTOS = 15
RECOMMENDED_PHOTOS = 25
MAX_PHOTO_EDGE = 1536


class UserFacingError(Exception):
    """An error with a message written for a non-programmer.

    main() catches these and prints the message alone, with no traceback.
    """


# ---------------------------------------------------------------------------
# Console output. Everything here stays inside ASCII: Windows consoles still
# default to cp1252 and a stray box-drawing character raises UnicodeEncodeError
# in the middle of a two-hour run.
# ---------------------------------------------------------------------------

def _emit(text: str = "") -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def say(text: str = "") -> None:
    _emit(text)


def step(text: str) -> None:
    _emit("  " + text)


def ok(text: str) -> None:
    _emit("  [OK] " + text)


def warn(text: str) -> None:
    _emit("  [!]  " + text)


def problem(text: str) -> None:
    _emit("  [X]  " + text)


def header(text: str) -> None:
    _emit("")
    _emit("=" * 66)
    _emit("  " + text.upper())
    _emit("=" * 66)
    _emit("")


def rule() -> None:
    _emit("  " + "-" * 62)


def bullets(lines: Iterable[str]) -> None:
    for line in lines:
        _emit("    - " + line)


def disable_click_to_pause() -> None:
    """Turn off QuickEdit for this console window (Windows only).

    With QuickEdit on, one stray click inside the window starts a text
    selection, and Windows freezes the program on its next print until the
    selection ends. The title gains "Select" and nothing else hints at it,
    so to the user the trainer simply looks hung - for hours, in practice.
    The cost is that text can no longer be copied out with the mouse; the
    run's log file covers that.
    """
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        std_input = kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(std_input, ctypes.byref(mode)):
            return
        enable_quick_edit, enable_extended_flags = 0x0040, 0x0080
        kernel32.SetConsoleMode(
            std_input, (mode.value & ~enable_quick_edit) | enable_extended_flags
        )
    except Exception:
        pass


def ask(prompt: str, default: str = "") -> str:
    """Prompt for a line of input, tolerating a missing console."""
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"  {prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        raise UserFacingError(
            "Cancelled. Nothing was started and nothing is running."
        )
    return answer or default


def ask_secret(prompt: str) -> str:
    """Prompt for a key. Deliberately echoes: hidden input confuses people
    who are pasting a long token and want to confirm it arrived."""
    return ask(prompt)


def confirm(prompt: str, expect: str = "YES") -> bool:
    answer = ask(f"{prompt} (type {expect} to continue)")
    return answer.strip().upper() == expect.upper()


def pause(text: str = "Press Enter to continue") -> None:
    try:
        input(f"  {text}... ")
    except (EOFError, KeyboardInterrupt):
        raise UserFacingError("Cancelled.")


def open_browser(url: str) -> None:
    try:
        webbrowser.open(url)
    except Exception:
        pass


def human_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds} seconds"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} minute" + ("s" if minutes != 1 else "")
    hours = minutes / 60.0
    return f"{hours:.1f} hours"


def human_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.0f} {unit}" if unit == "B" else f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


class SettleTracker:
    """Decides when a file has finished being written.

    Both variants copy checkpoints out while training is still running, so
    the user can look at snapshot 250 without waiting for snapshot 1000. A
    Krea 2 LoRA checkpoint is a few hundred MB and lands progressively, so
    grabbing one the moment it appears can capture a half-written file.

    The rule here is simply: only treat a file as ready once its size is
    identical on two consecutive polls. Cheap, and it needs no cooperation
    from the writer.
    """

    def __init__(self) -> None:
        self._last_size: dict[str, int] = {}
        self._done: set[str] = set()

    def is_ready(self, key: str, size: int) -> bool:
        if key in self._done or size <= 0:
            return False
        previous = self._last_size.get(key)
        self._last_size[key] = size
        return previous == size

    def mark_done(self, key: str) -> None:
        self._done.add(key)

    def is_done(self, key: str) -> bool:
        return key in self._done


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass
class Settings:
    character_name: str = ""
    trigger_word: str = ""
    huggingface_token: str = ""
    runpod_api_key: str = ""
    training_steps: int = 2000
    extras: dict = field(default_factory=dict)

    def save(self) -> None:
        data = asdict(self)
        SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        try:
            os.chmod(SETTINGS_FILE, 0o600)
        except Exception:
            pass


def _load_settings_file() -> dict:
    if not SETTINGS_FILE.exists():
        return {}
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        warn("my_settings.json could not be read, so I will ask again.")
        return {}


def make_trigger_word(name: str) -> str:
    """Turn a display name into a token the model can learn.

    Single lowercase word, letters and digits only. 'Laura Smith' -> 'laura'.
    A rare-ish token works better than a common English word, but a first
    name is what people actually want to type, so we keep it simple.
    """
    cleaned = re.sub(r"[^A-Za-z0-9 ]+", " ", name).strip()
    if not cleaned:
        return ""
    first = cleaned.split()[0].lower()
    return first


def make_slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    return slug or "character"


# ---------------------------------------------------------------------------
# Credential validation
# ---------------------------------------------------------------------------

def _requests():
    try:
        import requests  # noqa: PLC0415
    except ImportError:
        raise UserFacingError(
            "A required tool is missing. Close this window and run the\n"
            "  .bat file again - it will install what is needed."
        )
    return requests


def check_huggingface_token(token: str) -> tuple[bool, str]:
    """Validate a HF token and its access to the gated Krea 2 model.

    Returns (ok, message). Uses the plain HTTP API rather than
    huggingface_hub so the client install stays tiny.
    """
    requests = _requests()
    headers = {"Authorization": f"Bearer {token}"}

    try:
        who = requests.get(
            "https://huggingface.co/api/whoami-v2", headers=headers, timeout=30
        )
    except Exception as exc:
        return False, f"Could not reach huggingface.co ({exc}). Check your internet."

    if who.status_code == 401:
        return False, (
            "That token was rejected. It may have a typo, or it may have been\n"
            "     deleted. Tokens start with 'hf_'. Make a fresh one and try again."
        )
    if who.status_code != 200:
        return False, f"HuggingFace returned an unexpected error (code {who.status_code})."

    try:
        username = who.json().get("name", "your account")
    except Exception:
        username = "your account"

    try:
        model = requests.get(
            f"https://huggingface.co/api/models/{BASE_MODEL}",
            headers=headers,
            timeout=30,
        )
    except Exception as exc:
        return False, f"Could not reach huggingface.co ({exc}). Check your internet."

    if model.status_code == 200:
        return True, f"Signed in as {username}, and you have access to the model."
    if model.status_code in (401, 403):
        return False, (
            f"Signed in as {username}, but that account has NOT accepted the\n"
            f"     Krea 2 licence yet. Open this page, click the button to accept,\n"
            f"     wait a few seconds, then try again:\n"
            f"       {LICENSE_URL}"
        )
    if model.status_code == 404:
        return False, (
            "The model page could not be found. Either the licence has not been\n"
            f"     accepted on this account, or the model moved. Check {LICENSE_URL}"
        )
    return False, f"HuggingFace returned an unexpected error (code {model.status_code})."


def check_runpod_key(api_key: str) -> tuple[bool, str]:
    """Validate a RunPod API key and report the account balance."""
    requests = _requests()
    query = "query { myself { id currentSpendPerHr clientBalance } }"
    try:
        response = requests.post(
            "https://api.runpod.io/graphql",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"query": query},
            timeout=30,
        )
    except Exception as exc:
        return False, f"Could not reach runpod.io ({exc}). Check your internet."

    if response.status_code in (401, 403):
        return False, (
            "That RunPod key was rejected. Make sure you copied the whole thing\n"
            f"     from {RUNPOD_KEY_URL} and that it has read/write permission."
        )
    if response.status_code != 200:
        return False, f"RunPod returned an unexpected error (code {response.status_code})."

    try:
        payload = response.json()
    except Exception:
        return False, "RunPod sent back something unreadable. Try again in a minute."

    if payload.get("errors"):
        message = payload["errors"][0].get("message", "unknown error")
        return False, f"RunPod rejected the key: {message}"

    myself = (payload.get("data") or {}).get("myself") or {}
    balance = myself.get("clientBalance")
    if isinstance(balance, (int, float)):
        if balance < 5:
            return True, (
                f"Key works, but your balance is only ${balance:.2f}. A training run\n"
                "     costs about $3, so please top up before starting."
            )
        return True, f"Key works. Your balance is ${balance:.2f}."
    return True, "Key works."


# ---------------------------------------------------------------------------
# Setup wizard
# ---------------------------------------------------------------------------

def _wizard_character_name(current: str) -> tuple[str, str]:
    say("  WHAT IS THIS CHARACTER CALLED?")
    say()
    say("  Pick one word. It becomes the magic word you type later to")
    say("  summon this person in a picture. A first name is perfect.")
    say()
    while True:
        name = ask("Character name", current)
        if not name:
            problem("Please type something.")
            continue
        trigger = make_trigger_word(name)
        if not trigger:
            problem("Please use ordinary letters.")
            continue
        say()
        ok(f"Got it. Your magic word will be: {trigger}")
        say()
        return name, trigger


def _wizard_huggingface(current: str) -> str:
    say("  HUGGINGFACE - THE SITE THAT HOSTS THE AI MODEL")
    say()
    say("  The model is free, but its makers require you to agree to their")
    say("  terms first. This takes about two minutes and you only do it once.")
    say()
    say("  I will open two pages in your browser.")
    say()
    say("    PAGE 1 - the model page.")
    say("             Make an account if you do not have one, then find the")
    say("             box near the top asking you to accept the licence and")
    say("             click the button. Wait for it to say you have access.")
    say()
    say("    PAGE 2 - your tokens page.")
    say("             Click 'Create new token', choose the 'Read' type, give")
    say("             it any name, create it, then COPY it. It starts 'hf_'.")
    say("             You can only see it once, so copy it right away.")
    say()

    if current:
        say("  (You already have a token saved. Press Enter to keep using it.)")
        say()

    pause("Press Enter to open both pages")
    open_browser(LICENSE_URL)
    time.sleep(1.5)
    open_browser(HF_TOKEN_URL)
    say()

    while True:
        token = ask_secret("Paste your HuggingFace token here") or current
        if not token:
            problem("I need the token to continue. It starts with 'hf_'.")
            say()
            continue
        say()
        step("Checking that token...")
        good, message = check_huggingface_token(token)
        if good:
            ok(message)
            say()
            return token
        problem(message)
        say()
        say("  Fix that, then paste the token again (or press Ctrl+C to quit).")
        say()


def _wizard_runpod(current: str) -> str:
    say("  RUNPOD - THE SITE THAT RENTS THE COMPUTER")
    say()
    say("  You need an account with some money on it. Training one character")
    say("  costs about $3.")
    say()
    say("  IMPORTANT, please do this:")
    bullets([
        "Add about $30. Not more. RunPod is prepaid, so whatever you",
        "  put on the account is the absolute maximum you can be charged.",
        "Turn OFF auto-pay. It cancels out the protection above.",
    ])
    say()
    say("  Then go to Settings, find 'API Keys', create one with read and")
    say("  write permission, and copy it.")
    say()

    if current:
        say("  (You already have a key saved. Press Enter to keep using it.)")
        say()

    pause("Press Enter to open your RunPod settings page")
    open_browser(RUNPOD_KEY_URL)
    say()

    while True:
        key = ask_secret("Paste your RunPod API key here") or current
        if not key:
            problem("I need the key to rent the computer.")
            say()
            continue
        say()
        step("Checking that key...")
        good, message = check_runpod_key(key)
        if good:
            ok(message)
            say()
            return key
        problem(message)
        say()


def load_or_ask(
    *,
    need_runpod: bool,
    character_name: str = "",
    huggingface_token: str = "",
    runpod_api_key: str = "",
    training_steps: int = 0,
) -> Settings:
    """Load saved settings, filling any gaps with an interactive wizard.

    Values passed in (from the SETTINGS block at the top of the calling
    script) win over the saved file, so someone who prefers editing a file
    can still do that.
    """
    saved = _load_settings_file()
    settings = Settings(
        character_name=character_name or saved.get("character_name", ""),
        trigger_word=saved.get("trigger_word", ""),
        huggingface_token=huggingface_token or saved.get("huggingface_token", ""),
        runpod_api_key=runpod_api_key or saved.get("runpod_api_key", ""),
        training_steps=training_steps or saved.get("training_steps", 2000),
        extras=saved.get("extras", {}) or {},
    )
    if character_name:
        settings.trigger_word = make_trigger_word(character_name)

    needs_wizard = (
        not settings.character_name
        or not settings.trigger_word
        or not settings.huggingface_token
        or (need_runpod and not settings.runpod_api_key)
    )

    if not needs_wizard:
        ok(f"Using saved settings for '{settings.character_name}'.")
        step(f"To start over, delete the file: {SETTINGS_FILE.name}")
        return settings

    header("First-time setup")
    say("  A few quick questions. I will remember the answers, so you only")
    say("  have to do this once.")
    say()
    rule()
    say()

    if not settings.character_name or not settings.trigger_word:
        settings.character_name, settings.trigger_word = _wizard_character_name(
            settings.character_name
        )
        rule()
        say()

    if not settings.huggingface_token:
        settings.huggingface_token = _wizard_huggingface(settings.huggingface_token)
        rule()
        say()

    if need_runpod and not settings.runpod_api_key:
        settings.runpod_api_key = _wizard_runpod(settings.runpod_api_key)
        rule()
        say()

    settings.save()
    ok(f"Saved. Next time this is all skipped. ({SETTINGS_FILE.name})")
    return settings


# ---------------------------------------------------------------------------
# Photo intake
# ---------------------------------------------------------------------------

def _load_pillow():
    try:
        from PIL import Image, ImageOps  # noqa: PLC0415
    except ImportError:
        raise UserFacingError(
            "The image tool is missing. Close this window and run the .bat\n"
            "  file again - it will install what is needed."
        )
    try:
        import pillow_heif  # noqa: PLC0415

        pillow_heif.register_heif_opener()
    except Exception:
        # iPhone .HEIC support is a bonus, not a requirement. If the plugin
        # is unavailable we still handle every ordinary format; HEIC files
        # get reported individually below.
        pass
    return Image, ImageOps


def find_photos(source_dir: Path) -> list[Path]:
    if not source_dir.is_dir():
        raise UserFacingError(
            f"I could not find the photos folder:\n    {source_dir}\n\n"
            "  Make sure you are running the .bat file from inside the\n"
            "  extracted folder, not from inside the zip file itself."
        )
    return sorted(
        p for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def prepare_photos(source_dir: Path, dataset_dir: Path) -> int:
    """Normalise the user's photos into a clean training dataset.

    Converts HEIC, applies EXIF rotation, flattens transparency, downscales
    anything huge (training tops out at 1024px, so 1536 is ample and keeps
    the cloud upload quick) and re-encodes to JPEG.
    """
    Image, ImageOps = _load_pillow()

    photos = find_photos(source_dir)
    if not photos:
        raise UserFacingError(
            f"The 'my_photos' folder is empty.\n\n"
            f"  Put at least {MIN_PHOTOS} photos of your character in:\n"
            f"    {source_dir}\n\n"
            "  Then run this again."
        )

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir, ignore_errors=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    step(f"Found {len(photos)} photos. Preparing them...")

    kept = 0
    skipped: list[str] = []
    small: list[str] = []

    for source in photos:
        try:
            with Image.open(source) as img:
                img = ImageOps.exif_transpose(img)
                if img.mode in ("RGBA", "LA", "P"):
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    converted = img.convert("RGBA")
                    background.paste(converted, mask=converted.split()[-1])
                    img = background
                else:
                    img = img.convert("RGB")

                width, height = img.size
                if min(width, height) < 384:
                    small.append(source.name)

                longest = max(width, height)
                if longest > MAX_PHOTO_EDGE:
                    scale = MAX_PHOTO_EDGE / float(longest)
                    img = img.resize(
                        (max(1, int(width * scale)), max(1, int(height * scale))),
                        Image.LANCZOS,
                    )

                kept += 1
                img.save(dataset_dir / f"photo_{kept:03d}.jpg", "JPEG", quality=95)
        except Exception as exc:
            reason = "not a readable image"
            if source.suffix.lower() in (".heic", ".heif"):
                reason = "iPhone HEIC support unavailable"
            skipped.append(f"{source.name} ({reason}: {exc.__class__.__name__})")

    if skipped:
        warn(f"Skipped {len(skipped)} file(s) I could not read:")
        bullets(skipped[:8])
        if len(skipped) > 8:
            step(f"    ...and {len(skipped) - 8} more")

    if small:
        warn(
            f"{len(small)} photo(s) are quite low resolution. They still work, "
            "but sharper photos give a better result."
        )

    if kept < MIN_PHOTOS:
        raise UserFacingError(
            f"Only {kept} usable photo(s) found, but at least {MIN_PHOTOS} are needed\n"
            f"  for a decent result ({RECOMMENDED_PHOTOS}-40 is ideal).\n\n"
            f"  Add more photos to:\n    {source_dir}\n\n"
            "  Then run this again."
        )

    ok(f"{kept} photos ready.")
    if kept < RECOMMENDED_PHOTOS:
        warn(
            f"{kept} will work, but {RECOMMENDED_PHOTOS}-40 photos usually gives a "
            "noticeably better likeness."
        )
    return kept


def write_fallback_captions(dataset_dir: Path, trigger_word: str) -> int:
    """Write a plain caption for every image that does not have one.

    Used when the automatic captioner is unavailable or fails. A simple
    consistent caption still trains a perfectly usable character LoRA - it
    just gives you less control over pose and setting later.
    """
    written = 0
    for image in sorted(dataset_dir.iterdir()):
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        caption_path = image.with_suffix(".txt")
        if caption_path.exists() and caption_path.read_text(encoding="utf-8").strip():
            continue
        caption_path.write_text(
            f"{trigger_word}, a photograph of {trigger_word}", encoding="utf-8"
        )
        written += 1
    return written


def ensure_trigger_in_captions(dataset_dir: Path, trigger_word: str) -> int:
    """Make sure every caption starts with the trigger word.

    This is not optional. When cache_text_embeddings is on - which is how the
    low-VRAM local configs fit - ai-toolkit ignores the `trigger_word` config
    key entirely, so the word has to be physically present in the caption text
    or the LoRA never learns to respond to it.
    """
    fixed = 0
    for caption_path in sorted(dataset_dir.glob("*.txt")):
        text = caption_path.read_text(encoding="utf-8").strip()
        if not text:
            text = f"a photograph of {trigger_word}"
        lowered = text.lower()
        if lowered.startswith(trigger_word.lower()):
            continue
        caption_path.write_text(f"{trigger_word}, {text}", encoding="utf-8")
        fixed += 1
    return fixed


def count_dataset(dataset_dir: Path) -> tuple[int, int]:
    if not dataset_dir.is_dir():
        return 0, 0
    images = sum(
        1 for p in dataset_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    captions = sum(1 for p in dataset_dir.iterdir() if p.suffix.lower() == ".txt")
    return images, captions


# ---------------------------------------------------------------------------
# Checkpoint health, guarding ai-toolkit issue #925
# ---------------------------------------------------------------------------

_UP_TENSOR_MARKERS = ("lora_b", "lora_up")

# (exponent mask, shift, all-ones value) per safetensors dtype string.
_FLOAT_LAYOUT = {
    "F64": (8, "d", None),
    "F32": (4, "f", (0xFF, 23, 0xFF)),
    "F16": (2, None, (0x1F, 10, 0x1F)),
    "BF16": (2, None, (0xFF, 7, 0xFF)),
}


def _read_safetensors_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) < 8:
            raise UserFacingError(f"The file {path.name} is truncated or empty.")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length <= 0 or header_length > 200_000_000:
            raise UserFacingError(f"The file {path.name} does not look like a LoRA.")
        header_bytes = handle.read(header_length)
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except Exception:
        raise UserFacingError(f"The file {path.name} has an unreadable header.")
    return header, 8 + header_length


def _has_non_finite(chunk: bytes, dtype: str) -> bool:
    """Sample a tensor for NaN/Inf by inspecting exponent bits directly."""
    layout = _FLOAT_LAYOUT.get(dtype)
    if not layout:
        return False
    width, _, exponent_spec = layout
    if not exponent_spec:
        return False
    mask, shift, all_ones = exponent_spec

    sample = chunk[: 4096 * width]
    count = len(sample) // width
    if count == 0:
        return False

    if width == 4:
        values = array("I")
        values.frombytes(sample[: count * 4])
    elif width == 2:
        values = array("H")
        values.frombytes(sample[: count * 2])
    else:
        return False
    if sys.byteorder != "little":
        values.byteswap()

    return any(((value >> shift) & mask) == all_ones for value in values)


def verify_checkpoint_alive(path: Path) -> tuple[bool, str]:
    """Check that a saved LoRA actually contains learned weights.

    ai-toolkit issue #925 (open as of Aug 2026) produces Krea 2 LoRAs that
    train with a healthy decreasing loss, save without error, and load into
    ComfyUI attaching every patch - while every lora_B tensor is exactly
    zero. The result has literally no effect on generated images.

    A LoRA's "up" matrices are initialised to zero on purpose, so they are
    zero at step 0 and must be non-zero after real training. That makes an
    all-zero check both cheap and exact. Comparing raw bytes avoids needing
    numpy or torch on the client.
    """
    if not path.exists():
        return False, f"{path.name} was not created."
    if path.stat().st_size < 1024:
        return False, f"{path.name} is suspiciously small."

    header, data_start = _read_safetensors_header(path)
    up_tensors = {
        name: info
        for name, info in header.items()
        if name != "__metadata__"
        and isinstance(info, dict)
        and any(marker in name.lower() for marker in _UP_TENSOR_MARKERS)
    }

    if not up_tensors:
        # Unrecognised naming scheme. Refusing to guess is better than
        # reporting a false failure on a perfectly good file.
        return True, "Could not identify the LoRA layers, so no check was possible."

    non_zero = 0
    non_finite = 0
    with path.open("rb") as handle:
        for name, info in up_tensors.items():
            try:
                start, end = info["data_offsets"]
                dtype = info.get("dtype", "F16")
            except Exception:
                continue
            handle.seek(data_start + start)
            chunk = handle.read(end - start)
            if chunk.strip(b"\x00"):
                non_zero += 1
            if _has_non_finite(chunk, dtype):
                non_finite += 1

    total = len(up_tensors)
    if non_finite:
        return False, (
            f"{non_finite} of {total} layers contain broken numbers (NaN/Inf)."
        )
    if non_zero == 0:
        return False, (
            f"All {total} layers are completely empty - the LoRA learned nothing."
        )
    if non_zero < total * 0.5:
        return False, (
            f"Only {non_zero} of {total} layers learned anything. "
            "This LoRA is mostly empty."
        )
    return True, f"{non_zero} of {total} layers contain real learned weights."


DEAD_LORA_EXPLANATION = """
  WHAT THIS MEANS

  Training appeared to run normally, but the file it produced is blank.
  It would load without complaining and do absolutely nothing.

  This is a known bug in the training software when it is squeezing a
  very large model into a small graphics card. It is not something you
  did wrong, and your photos are fine.

  WHAT TO DO

    1. Easiest and most reliable: use the cloud instead.
       Double-click "1 - TRAIN IN THE CLOUD.bat". The cloud computer has
       enough memory that it does not need the trick that triggers this
       bug. It costs about $3.

    2. Or try again on this PC. The bug is intermittent, so a second run
       sometimes works. Just run the same .bat file again.
"""


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------

_TQDM_STEPS = re.compile(r"(\d+)\s*/\s*(\d+)\s*\[")
_TQDM_RATE = re.compile(r"([\d.]+)\s*s/it")
_TQDM_RATE_INVERTED = re.compile(r"([\d.]+)\s*it/s")
_LOSS = re.compile(r"loss:\s*([\d.eE+-]+)")


def parse_progress(text: str, total_steps: int) -> dict[str, Any]:
    """Pull the current step, rate and loss out of ai-toolkit's tqdm output.

    ai-toolkit writes a tqdm bar like:
      name:  17%|#7   | 340/2000 [11:20<55:20,  2.00s/it, lr: 1.0e-04 loss: 3.9e-01]

    We compute the ETA ourselves from the rate rather than reading tqdm's,
    because tqdm's estimate swings wildly early in a run.
    """
    result: dict[str, Any] = {
        "step": None, "total": total_steps, "seconds_per_step": None,
        "eta_seconds": None, "loss": None,
    }
    if not text:
        return result

    # Only ever trust a bar whose total is exactly the step count we asked
    # for. Everything else on that pod emits tqdm too - pip, the ~35GB model
    # shard download, the dataset scan, the captioner - and an earlier
    # `or total_i > 1` fallback here matched all of them. Observed live: a
    # 1000-step run reported "Step 730 of 730 (100%)" while still downloading
    # the model. If no training bar has appeared yet, report nothing and let
    # the caller say "still working" rather than invent a number.
    steps_matches = _TQDM_STEPS.findall(text)
    for current, total in reversed(steps_matches):
        if int(total) == total_steps:
            result["step"] = int(current)
            result["total"] = total_steps
            break

    # Take the median of recent rates rather than the latest one. Saving a
    # 218MB checkpoint stalls the step clock, so the reading immediately
    # after one is wildly high: observed live, a steady 1.8 s/it run reported
    # "about 7.0 hours left" at step 249 and "about 2.2 hours left" at 749,
    # both right after a save, before settling back to ~20 minutes. A median
    # ignores those isolated spikes without lagging a genuine slowdown.
    rates = [float(value) for value in _TQDM_RATE.findall(text)]
    if not rates:
        rates = [
            1.0 / float(value)
            for value in _TQDM_RATE_INVERTED.findall(text)
            if float(value) > 0
        ]
    if rates:
        result["seconds_per_step"] = median(rates[-9:])

    loss_matches = _LOSS.findall(text)
    if loss_matches:
        try:
            result["loss"] = float(loss_matches[-1])
        except ValueError:
            pass

    if result["step"] is not None and result["seconds_per_step"]:
        remaining = max(0, result["total"] - result["step"])
        result["eta_seconds"] = remaining * result["seconds_per_step"]

    return result


def format_progress(progress: dict[str, Any]) -> str:
    stepno = progress.get("step")
    total = progress.get("total")
    if stepno is None:
        return "  Warming up..."
    percent = (100.0 * stepno / total) if total else 0.0
    line = f"  Step {stepno} of {total}  ({percent:.0f}%)"
    eta = progress.get("eta_seconds")
    if eta:
        line += f"  -  about {human_duration(eta)} left"
    return line


class ProgressPrinter:
    """Prints progress at most once every `interval` seconds."""

    def __init__(self, interval: float = 30.0) -> None:
        self.interval = interval
        self._last = 0.0
        self._last_step = -1

    def maybe_print(self, progress: dict[str, Any], force: bool = False) -> None:
        now = time.time()
        stepno = progress.get("step")
        if not force:
            if now - self._last < self.interval:
                return
            if stepno == self._last_step:
                return
        self._last = now
        self._last_step = stepno
        say(format_progress(progress))


# ---------------------------------------------------------------------------
# Result collection
# ---------------------------------------------------------------------------

def summarise_results(output_dir: Path, slug: str) -> None:
    loras = sorted(output_dir.glob("*.safetensors"))
    previews = output_dir / "preview_images"
    preview_count = len(list(previews.glob("*"))) if previews.is_dir() else 0

    header("Finished")
    if loras:
        ok(f"{len(loras)} LoRA file(s) saved to the 'output' folder:")
        for lora in loras:
            step(f"  {lora.name}  ({human_size(lora.stat().st_size)})")
        say()
        if len(loras) > 1:
            say("  Several snapshots were saved from different points in training.")
            say("  Later is NOT always better - a LoRA can overtrain and start")
            say("  copying your photos too literally. Try the one around step 1000")
            say("  first, then compare against the others.")
            say()
    else:
        warn("No LoRA file was produced. Something went wrong above.")

    if preview_count:
        ok(f"{preview_count} preview image(s) in output\\preview_images")
        say("  Open them to check the likeness before you use the LoRA.")
    else:
        say("  No preview images were generated for this run.")
    say()
