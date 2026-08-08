#!/usr/bin/env python3
"""
Build the zip that gets sent to friends.

This directory is two things at once: the source of the distributable, and
the author's own working copy. So it holds a settings file with a live
RunPod key and HuggingFace token, a few hundred MB of somebody's photos, and
the LoRAs trained from them. Zipping the folder would post all of that.

Hence an allowlist. Nothing ships unless it is named below, so a new file
dropped in this directory is excluded by default rather than included by
default. A denylist would leak the first time anyone added a file the list
had not anticipated, and the thing it would leak is an API key.

After staging, every shipped file is searched for anything that looks like a
credential or like the maintainer's real-world identity, and the build fails
outright if something is found. The check is deliberately paranoid: a false
alarm costs a minute, a miss costs a leaked key.

Note that "Krystal" is the public name of this project, not a secret, so it
is not on the forbidden list. What must never ship is a credential or a name
that ties the release back to a person rather than to the project.

Usage:
    python build_distributable.py            # build into ./dist
    python build_distributable.py --out DIR  # build somewhere else
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE_NAME = "krystals-lora-trainer"

# Exactly what a friend receives. Anything absent from this list stays home.
SHIPPED_FILES = [
    "READ ME FIRST.txt",
    "1 - TRAIN IN THE CLOUD.bat",
    "2 - TRAIN ON MY PC.bat",
    "3 - EMERGENCY - SHUT DOWN CLOUD.bat",
    "my_photos/PUT YOUR PHOTOS IN THIS FOLDER.txt",
    "output/YOUR RESULTS WILL APPEAR HERE.txt",
    "trainer/_configs.py",
    "trainer/_shared.py",
    "trainer/bootstrap_windows.ps1",
    "trainer/caption_dataset.py",
    "trainer/setup_and_train.sh",
    "trainer/terminate_pod.py",
    "trainer/train_local.py",
    "trainer/train_runpod.py",
]

# Windows runs the first group; the Linux pod runs the second. A .bat with LF
# endings mis-parses `goto` labels, and a .sh with CRLF breaks its shebang.
# Enforced here rather than trusted, because a checkout with the wrong
# autocrlf setting would otherwise ship broken files.
CRLF_SUFFIXES = {".bat", ".ps1", ".txt"}
LF_SUFFIXES = {".sh", ".py", ".json", ".yaml", ".yml"}

SECRET_PATTERNS = [
    (re.compile(r"hf_[A-Za-z0-9]{20,}"), "HuggingFace token"),
    (re.compile(r"rpa_[A-Za-z0-9]{20,}"), "RunPod API key"),
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "API key"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),
]

# Words that would tie a release back to a person rather than to the project:
# a maintainer's real name, an account handle, the path of a private repo.
#
# Those words live in an untracked file rather than in this source, because
# this source is itself published. Hardcoding the list here would print the
# exact strings it exists to catch into a public repo, which is the leak
# rather than the guard against it. One word per line, '#' comments and blank
# lines ignored, matching is case-insensitive substring.
PRIVATE_WORDS_FILE = HERE / ".forbidden_words"

# The project's own name. A maintainer whose character is called Krystal would
# otherwise trip the settings-derived check below on every single build.
BRAND_WORDS = {"krystal", "krystals", "lora", "trainer"}


def private_words() -> list[str]:
    """Maintainer-specific words, read from the untracked list beside us."""
    if not PRIVATE_WORDS_FILE.exists():
        return []
    words = []
    for line in PRIVATE_WORDS_FILE.read_text(encoding="utf-8").splitlines():
        word = line.split("#", 1)[0].strip().lower()
        # A one- or two-character entry would match almost every file and
        # make the build unpassable, which trains people to ignore it.
        if len(word) >= 3:
            words.append(word)
    return words


def forbidden_words() -> list[str]:
    """Words that must not appear in anything handed to someone else."""
    words = private_words()
    settings_file = HERE / "my_settings.json"
    if settings_file.exists():
        try:
            saved = json.loads(settings_file.read_text(encoding="utf-8"))
        except Exception:
            return words
        for key in ("character_name", "trigger_word"):
            value = str(saved.get(key) or "").strip().lower()
            # Guard against a one-letter or empty name matching everything,
            # and against the project's own name banning itself.
            if len(value) >= 3 and value not in words and value not in BRAND_WORDS:
                words.append(value)
    return words


def normalise(data: bytes, suffix: str) -> bytes:
    if suffix in CRLF_SUFFIXES:
        return data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    if suffix in LF_SUFFIXES:
        return data.replace(b"\r\n", b"\n")
    return data


def stage(target: Path) -> list[Path]:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    staged = []
    missing = []
    for relative in SHIPPED_FILES:
        source = HERE / relative
        if not source.is_file():
            missing.append(relative)
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(normalise(source.read_bytes(), source.suffix.lower()))
        staged.append(destination)

    if missing:
        raise SystemExit(
            "These files are on the ship list but not on disk:\n  "
            + "\n  ".join(missing)
        )
    return staged


def audit(target: Path, staged: list[Path]) -> list[str]:
    """Every reason this build must not be sent to anyone."""
    problems = []
    words = forbidden_words()

    for path in staged:
        relative = path.relative_to(target)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            problems.append(f"{relative}: not text, so it cannot be checked")
            continue

        for pattern, label in SECRET_PATTERNS:
            if pattern.search(text):
                problems.append(f"{relative}: contains a {label}")

        lowered = text.lower()
        for word in words:
            if word in lowered:
                line_no = next(
                    (i for i, line in enumerate(text.splitlines(), 1)
                     if word in line.lower()),
                    0,
                )
                problems.append(
                    f"{relative}:{line_no}: mentions '{word}'"
                )

    # A stray photo or checkpoint means the staging logic let user data past.
    for path in target.rglob("*"):
        if path.is_dir():
            continue
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".heic", ".webp",
                                   ".safetensors", ".pt", ".ckpt"}:
            problems.append(f"{path.relative_to(target)}: user data in the build")

    extra = {p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()}
    unexpected = extra - set(SHIPPED_FILES)
    for name in sorted(unexpected):
        problems.append(f"{name}: not on the ship list")

    return problems


def make_zip(target: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in SHIPPED_FILES:
            archive.write(target / relative, f"{PACKAGE_NAME}/{relative}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(HERE / "dist"),
                        help="where to put the build (default: ./dist)")
    parser.add_argument("--no-zip", action="store_true",
                        help="stage the folder but do not zip it")
    args = parser.parse_args()

    out_dir = Path(args.out).resolve()
    target = out_dir / PACKAGE_NAME

    print(f"Staging into {target}")
    staged = stage(target)

    print(f"Checking {len(staged)} files for anything that must not be shared...")
    if not PRIVATE_WORDS_FILE.exists():
        # Say so rather than passing quietly, or a clean build reads as proof
        # of something that was never actually checked.
        print(f"  note: no {PRIVATE_WORDS_FILE.name}, so personal names are"
              f" NOT being checked for.")
        print("        credential and user-data checks still run.")
    problems = audit(target, staged)
    if problems:
        print("\nBUILD REFUSED. Fix these first:\n")
        for problem in problems:
            print(f"  {problem}")
        print()
        shutil.rmtree(target, ignore_errors=True)
        return 1
    print("  clean: no credentials, no personal names, no user data")

    total = sum(p.stat().st_size for p in staged)
    print(f"\n{len(staged)} files, {total / 1024:.0f} KB")

    if args.no_zip:
        print(f"\nFolder ready: {target}")
        return 0

    zip_path = out_dir / f"{PACKAGE_NAME}.zip"
    make_zip(target, zip_path)
    print(f"\nSend this file: {zip_path}")
    print(f"  {zip_path.stat().st_size / 1024:.0f} KB")
    print("\nThey unzip it, put photos in 'my_photos', and double-click")
    print("\"1 - TRAIN IN THE CLOUD.bat\".")
    return 0


if __name__ == "__main__":
    sys.exit(main())
