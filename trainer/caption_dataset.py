#!/usr/bin/env python3
"""
Rationale (verified): ai-toolkit ships a captioner in
extensions_built_in/captioner/, but it is wired to the web UI rather than a
documented CLI. A non-programmer cannot be asked to write 30 captions by
hand, so we need a headless captioner we control. This script is uploaded to
the pod for cloud runs and executed inside the ai-toolkit venv for local
runs - one implementation for both.

What it does:
  1. Loads a small vision-language model (Qwen3-VL-2B by default)
  2. Describes each photo in one sentence: framing, clothing, pose, setting
  3. Deliberately omits facial and identity detail so the trigger word is
     what carries the likeness
  4. Writes <image>.txt next to each image, always starting with the trigger
  5. Falls back cleanly - any failure exits non-zero and the caller writes
     simple template captions instead, which still trains a usable LoRA

Usage:
  python caption_dataset.py --dataset DIR --trigger laura
  python caption_dataset.py --dataset DIR --trigger laura --model Qwen/Qwen3-VL-2B-Instruct

Maintenance: Runs inside ai-toolkit's venv and relies only on transformers +
torch, which ai-toolkit already requires. If the Qwen3-VL model id changes,
update MODEL_CANDIDATES. Never let this script become required for training
to succeed - the caller must always be able to fall back.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

MODEL_CANDIDATES = [
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen2.5-VL-3B-Instruct",
]

# Describe everything EXCEPT who the person is. The trigger word has to be
# the only thing carrying identity, otherwise the model learns "blonde woman"
# rather than this specific person.
INSTRUCTION = (
    "Describe this photograph in one short sentence for an image-generation "
    "caption. Say the framing (close-up portrait, waist-up shot, or full body "
    "shot), what the person is wearing, their pose, and the setting and "
    "lighting. Do NOT describe their face, hair colour, age, or who they are. "
    "Do not start with 'The image' or 'This photo'. Just the description."
)

_PREAMBLE = re.compile(
    r"^(the (image|photo|photograph|picture)( shows| depicts| features| is of)?|"
    r"this (image|photo|photograph|picture)( shows| depicts| features| is of)?|"
    r"in (this|the) (image|photo|photograph|picture),?|"
    r"a (photo|photograph|picture) of|here is|it (shows|depicts))\s*",
    re.IGNORECASE,
)


def clean_caption(text: str) -> str:
    text = " ".join(text.strip().split())
    for _ in range(3):
        cleaned = _PREAMBLE.sub("", text).strip()
        if cleaned == text:
            break
        text = cleaned
    text = text.strip(" .,;:\"'")
    if len(text) > 400:
        text = text[:400].rsplit(" ", 1)[0]
    return text


def load_model(model_id: str):
    import torch
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoVLM

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoVLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


def describe(model, processor, image_path: Path, max_new_tokens: int) -> str:
    import torch
    from PIL import Image

    with Image.open(image_path) as raw:
        image = raw.convert("RGB")
        # The captioner does not need full resolution and this keeps VRAM
        # predictable on small cards.
        image.thumbnail((512, 512))

        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": INSTRUCTION},
            ],
        }]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[prompt], images=[image], return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items() if hasattr(v, "to")}

        with torch.no_grad():
            generated = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )

        trimmed = generated[0][inputs["input_ids"].shape[1]:]
        return processor.decode(trimmed, skip_special_tokens=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Caption a LoRA training dataset")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--trigger", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    dataset = args.dataset.expanduser()
    if not dataset.is_dir():
        print(f"CAPTION: dataset folder not found: {dataset}", flush=True)
        return 1

    images = sorted(
        p for p in dataset.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        print("CAPTION: no images found", flush=True)
        return 1

    candidates = [args.model] if args.model else list(MODEL_CANDIDATES)
    model = processor = None
    chosen = ""
    errors = []
    for model_id in candidates:
        try:
            print(f"CAPTION: loading {model_id} ...", flush=True)
            model, processor = load_model(model_id)
            chosen = model_id
            break
        except Exception as exc:
            errors.append(f"{model_id}: {exc}")
            print(f"CAPTION: could not load {model_id} ({exc})", flush=True)

    if model is None:
        print("CAPTION: no captioning model could be loaded.", flush=True)
        for line in errors:
            print(f"CAPTION:   {line}", flush=True)
        return 1

    print(f"CAPTION: using {chosen} on {len(images)} images", flush=True)

    trigger = args.trigger.strip()
    written = 0
    failed = 0

    for index, image_path in enumerate(images, start=1):
        caption_path = image_path.with_suffix(".txt")
        try:
            raw = describe(model, processor, image_path, args.max_new_tokens)
            caption = clean_caption(raw)
        except Exception as exc:
            print(f"CAPTION: failed on {image_path.name}: {exc}", flush=True)
            caption = ""
            failed += 1

        if not caption:
            caption = "a photograph of a person"

        caption_path.write_text(f"{trigger}, {caption}", encoding="utf-8")
        written += 1
        if index == 1 or index % 5 == 0 or index == len(images):
            print(f"CAPTION: {index}/{len(images)}", flush=True)

    print(f"CAPTION: wrote {written} captions ({failed} fell back to a default)", flush=True)
    print(f"CAPTION: example -> {(images[0].with_suffix('.txt')).read_text(encoding='utf-8')}", flush=True)

    # A handful of per-image failures is fine; a total washout is not.
    return 1 if failed == len(images) else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:  # never take the whole run down with us
        print(f"CAPTION: unexpected error: {exc}", flush=True)
        sys.exit(1)
