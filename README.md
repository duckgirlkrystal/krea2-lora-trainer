# Krystal's LoRA Trainer

Teach an image generator what one specific person looks like.

You give it 20–50 photos of one person. It gives you back a **LoRA** — a small
file you load into an image generator to make new pictures of that person —
plus preview images so you can see that it worked.

No command line. No Python knowledge. No config files to edit. You put photos
in a folder and double-click a file.

Built on [Krea 2](https://huggingface.co/krea/krea-2) via
[ai-toolkit](https://github.com/ostris/ai-toolkit).

---

## Getting it

Download the latest `krystals-lora-trainer.zip` from
[Releases](../../releases), unzip it anywhere, and open **READ ME FIRST.txt**.

Windows only for the double-click experience. The training code itself is
platform-independent, but the launchers are `.bat` files.

## The two ways to train

| | Cloud | Your own PC |
|---|---|---|
| **Cost** | ~$2–3 per character | Free |
| **Requirements** | A RunPod account with credit | NVIDIA GPU, 16GB+ VRAM, 60GB disk |
| **Works on** | Any computer, even an old laptop | Only a machine that meets the above |
| **Time** | ~1.5 hours | ~1.5–3 hours |

The cloud path rents a GPU, trains on it, downloads the result, and shuts the
rented machine down so you stop paying. There are several independent
backstops against a machine being left running, plus a one-click
**3 - EMERGENCY - SHUT DOWN CLOUD.bat** that kills anything this tool rented.

Read the money section of `READ ME FIRST.txt` before your first cloud run. The
short version: prepay ~$30 on RunPod and turn auto-pay off. A prepaid balance
is a hard ceiling on what you can ever be charged.

## What you need

**Photos.** At least 15, ideally 25–40, all of the same person. Mixed framing
(face, waist-up, full body), mixed outfits, mixed lighting. Sharp and in
focus. iPhone `.HEIC` files are converted automatically. Thirty good photos
beat a hundred mediocre ones.

**A HuggingFace account**, to accept the Krea 2 licence and get a read token.
The program walks you through it and tells you exactly which page to open.

**A RunPod account**, only for the cloud path.

The program asks for these once and remembers them.

---

## Layout

```
READ ME FIRST.txt               Start here — written for non-technical users
1 - TRAIN IN THE CLOUD.bat      Rent a GPU, train, download, shut down
2 - TRAIN ON MY PC.bat          Train on your own NVIDIA card
3 - EMERGENCY - SHUT DOWN CLOUD.bat   Kill anything this tool rented
my_photos/                      Your input photos go here
output/                         LoRAs and preview images land here
trainer/                        The actual implementation
build_distributable.py          Builds the release zip
```

`trainer/` is the interesting part:

| File | Does what |
|---|---|
| `train_runpod.py` | Cloud path: provision, upload, train, poll, download, terminate |
| `train_local.py` | Local path: hardware checks, then train in-process |
| `_shared.py` | Console output, settings, token validation, checkpoint verification |
| `_configs.py` | Generates the ai-toolkit training config |
| `caption_dataset.py` | Auto-captions the photo set |
| `setup_and_train.sh` | Runs on the rented pod |
| `terminate_pod.py` | Backs the emergency shutdown button |
| `bootstrap_windows.ps1` | Sets up Python and dependencies on first run |

## Building a release

```
python build_distributable.py
```

Stages into `dist/krystals-lora-trainer/` and zips it. The build uses an
**allowlist** — a file ships only if it is named in `SHIPPED_FILES`, so a new
file dropped into the working tree is excluded by default rather than
included by default. After staging it scans every shipped file for API keys,
private keys, personal identifiers, and stray user data, and refuses to build
if it finds any.

That paranoia is deliberate. The working copy of this directory holds a live
RunPod key, a HuggingFace token, and someone's personal photos. A denylist
would leak the first time somebody added a file it hadn't anticipated.

## Contributing

Issues and pull requests welcome. Please don't include API keys, personal
photos, or trained LoRAs in a PR — `build_distributable.py` will catch most of
that, but the `.gitignore` is the first line of defence.

## Licence

MIT. See [LICENSE](LICENSE).

Krea 2 itself is under its own licence, which you accept on HuggingFace before
you can download it. This tool does not redistribute model weights.
