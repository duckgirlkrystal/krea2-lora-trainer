"""
Rationale (verified): ai-toolkit takes a native YAML config. The config this
was derived from was pinned to one character and to an A100 80GB, so it could
not be shipped as-is. This module generates a config for an arbitrary
character at a VRAM tier appropriate to the hardware it will run on.

What it does:
  1. Picks a hardware tier from available VRAM
  2. Emits ai-toolkit native YAML for training
  3. Emits a second "preview" config that loads the finished LoRA and samples
     from it, for tiers where sampling cannot run during training

Usage:
  Imported by train_runpod.py and train_local.py. Not run directly.

Maintenance: Every config key below was verified against ai-toolkit
toolkit/config_modules.py on main in Aug 2026 (SaveConfig L23, SampleConfig
L79, NetworkConfig L169, TrainConfig L375, ModelConfig L680, DatasetConfig
L906). Two behaviours are load-bearing and easy to regress:
  - ModelConfig L709-712 silently rewrites qtype "qfloat8" -> "float8" when
    layer_offloading is on. We pin "float8" so the file says what happens.
  - config_modules L1480-1490 requires cache_text_embeddings on BOTH the
    train block and every dataset block, or it raises.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Unquantized bf16 Krea 2 is a 12B DiT plus a 4B text encoder, needing ~45GB.
# Below that we quantize; below ~28GB we also stream layers from system RAM.
VRAM_MINIMUM_GB = 15.0


@dataclass(frozen=True)
class Tier:
    key: str
    label: str
    resolutions: tuple[int, ...]
    quantize: bool
    layer_offloading: bool
    transformer_percent: float
    low_vram: bool
    cache_text_embeddings: bool
    sample_during_training: bool
    note: str


# Ordered from smallest card to largest. pick_tier walks this backwards.
TIERS: tuple[Tier, ...] = (
    Tier(
        key="16gb",
        label="16GB graphics card",
        resolutions=(512,),
        quantize=True,
        layer_offloading=True,
        transformer_percent=0.35,
        low_vram=True,
        cache_text_embeddings=True,
        sample_during_training=False,
        # 0.35 is the value reported working on an RTX 5080 16GB at ~2s/it.
        # Note that ai-toolkit picks the offloaded set randomly per linear
        # layer, so peak VRAM varies between runs - do not tune this tighter.
        note="Squeezing hard. Training will be slower and previews are off.",
    ),
    Tier(
        key="24gb",
        label="24GB graphics card",
        resolutions=(768,),
        quantize=True,
        layer_offloading=True,
        transformer_percent=0.15,
        low_vram=True,
        cache_text_embeddings=True,
        sample_during_training=False,
        note="Comfortable. Previews are generated after training instead.",
    ),
    Tier(
        key="32gb",
        label="32GB graphics card",
        resolutions=(768, 1024),
        quantize=True,
        layer_offloading=False,
        transformer_percent=0.0,
        low_vram=False,
        cache_text_embeddings=False,
        sample_during_training=True,
        note="Plenty of room. Full quality with live previews.",
    ),
    Tier(
        key="big",
        label="48GB or larger",
        resolutions=(768, 1024),
        quantize=False,
        layer_offloading=False,
        transformer_percent=0.0,
        low_vram=False,
        cache_text_embeddings=False,
        sample_during_training=True,
        note="Best possible quality, nothing compromised.",
    ),
)

TIER_BY_KEY = {tier.key: tier for tier in TIERS}


# ---------------------------------------------------------------------------
# Base models
# ---------------------------------------------------------------------------
# Krea 2 is the default and the only model the friend-facing .bat files use.
# MOSS-Image is a maintainer option (train_runpod.py --model moss).
#
# MOSS-Image settings, verified against ai-toolkit main at e9c72b7 (Oct 31
# 2026), extensions_built_in/diffusion_models/moss_image and the UI defaults
# for "moss_image:base" in extensions_built_in/diffusion_models/ui.tsx:
#   - Trained on Base, not SFT. OpenMOSS's USERGUIDE.md names Base as the
#     checkpoint for LoRA training, and SFT has a known vertical-stripe
#     artifact. Base LoRAs load on SFT too.
#   - arch "moss_image". ai-toolkit strips any ":tag" (ModelConfig L773), and
#     MossImageModel picks Base vs SFT from name_or_path, not from the arch.
#   - Unquantized on the cloud. The comfy repack names its int8 file
#     moss_image_base_int8_convrot.safetensors but ai-toolkit's
#     checkpoints.py looks for ..._int8_convrot_attn, so a convrot8 request
#     silently falls through to the bf16 download plus an on-load quantize.
#     bf16 avoids that untested path entirely. Files fetched (all public, no
#     licence step): Comfy-Org/MOSS-Image diffusion_models/
#     moss_image_base_bf16 (10.2GB), Comfy-Org/Qwen3.5 text_encoders/
#     qwen3.5_9b_bf16 (19.3GB), Comfy-Org/MOSS-Image vae/flux2-vae (0.34GB),
#     tokenizer from OpenMOSS-Team/MOSS-Image-Base.
#   - timestep_type "shift" and sampling at CFG 3, 30 steps, block 22 skipped
#     on the negative branch: the ui.tsx and registry.py defaults, matching
#     the vendor's recommended sampling (CFG 3, shift 5, uncond skip 22).
#   - Block 22 stays IN the LoRA. ui.tsx excludes it by default, but the
#     community likeness recipes say excluding it costs likeness, and the
#     exclusion only exists to keep CFG's skipped-block contrast pristine.


@dataclass(frozen=True)
class ModelProfile:
    key: str
    label: str
    base_model: str
    arch: str
    # Appended to the character slug to name the run and the LoRA file.
    run_suffix: str
    guidance_scale: float
    sample_steps: int
    # Below this the pod-side check warns that the card is undersized.
    expected_vram_gb: int
    # One line shown in the model menu.
    description: str
    # Pinned ai-toolkit commit, or "" for main. MOSS support was days old when
    # this was written, so it is pinned to the commit it was verified against.
    ai_toolkit_ref: str


MODELS: dict[str, ModelProfile] = {
    "krea2": ModelProfile(
        key="krea2",
        label="Krea 2",
        base_model="krea/Krea-2-Raw",
        arch="krea2",
        run_suffix="krea2",
        guidance_scale=3,
        sample_steps=25,
        expected_vram_gb=40,
        description="The original. Best tested. Needs the Krea 2 licence accepted on HuggingFace.",
        ai_toolkit_ref="",
    ),
    "moss": ModelProfile(
        key="moss",
        label="MOSS-Image Base",
        base_model="OpenMOSS-Team/MOSS-Image-Base",
        arch="moss_image",
        run_suffix="moss",
        guidance_scale=3,
        sample_steps=30,
        expected_vram_gb=40,
        description="Newer and smaller (5B). Strong likeness up close. Cloud only.",
        ai_toolkit_ref="e9c72b762748e4d31837fbeb3b87d01a5b143321",
    ),
}
DEFAULT_MODEL = "krea2"


def pick_tier(vram_gb: float) -> Tier:
    if vram_gb >= 45:
        return TIER_BY_KEY["big"]
    if vram_gb >= 28:
        return TIER_BY_KEY["32gb"]
    if vram_gb >= 20:
        return TIER_BY_KEY["24gb"]
    return TIER_BY_KEY["16gb"]


def _q(value: str) -> str:
    """Quote a string for YAML. JSON string syntax is valid YAML."""
    return json.dumps(str(value))


def _yaml_list(values) -> str:
    return "[" + ", ".join(str(v) for v in values) + "]"


def sample_prompts(trigger_word: str, model: str = DEFAULT_MODEL) -> list[str]:
    if model == "moss":
        # MOSS was trained on natural-language captions, so its previews are
        # sentences in the same shape as the training captions.
        return [
            f"{trigger_word}, a photograph of a 35 year old woman. A close-up "
            "portrait in soft natural window light, with a soft confident half-smile.",
            f"{trigger_word}, a photograph of a 35 year old woman standing outdoors "
            "on a sunny day. A waist-up shot, looking at the camera.",
            f"{trigger_word}, a photograph of a 35 year old woman in a white tank top "
            "and jeans walking down a city sidewalk. A full body shot in bright daylight.",
            f"{trigger_word}, a photograph of a 35 year old woman sitting at a kitchen "
            "table in warm indoor light. A medium shot, smiling at the camera.",
        ]
    return [
        f"{trigger_word}, a photograph, close-up portrait, soft natural window light",
        f"{trigger_word}, a photograph, waist-up shot, standing outdoors on a sunny day",
        f"{trigger_word}, a photograph, full body shot, casual clothes, city street",
        f"{trigger_word}, a photograph, close-up portrait, warm indoor light, smiling",
    ]


def _model_block(tier: Tier, model: str = DEFAULT_MODEL) -> str:
    profile = MODELS[model]
    lines = [
        f'        name_or_path: {_q(profile.base_model)}',
        f'        arch: {_q(profile.arch)}',
    ]
    if model == "moss":
        lines += [
            # Explicit although it is MossImageModel's default: the sampler
            # skips this block on the negative CFG branch.
            "        model_kwargs:",
            "          uncond_skip_layers: [22]",
        ]
    if tier.quantize and model == "moss":
        # convrot8, the MOSS default in ai-toolkit's registry. Only the local
        # small-card tiers reach this; it has not been run end to end.
        lines += [
            "        quantize: true",
            '        qtype: "convrot8"',
            "        quantize_te: true",
            '        qtype_te: "convrot8"',
        ]
    elif tier.quantize:
        lines += [
            "        quantize: true",
            # Pinned rather than left as qfloat8: ai-toolkit rewrites qfloat8
            # to float8 whenever layer_offloading is on, and a config that
            # states the value actually used is far easier to debug.
            '        qtype: "float8"',
            "        quantize_te: true",
            '        qtype_te: "float8"',
        ]
    else:
        lines += [
            "        quantize: false",
            "        quantize_te: false",
        ]
    lines.append(f"        low_vram: {str(tier.low_vram).lower()}")
    if tier.layer_offloading:
        lines += [
            "        layer_offloading: true",
            f"        layer_offloading_transformer_percent: {tier.transformer_percent}",
            "        layer_offloading_text_encoder_percent: 1.0",
        ]
    return "\n".join(lines)


def build_training_config(
    *,
    run_name: str,
    trigger_word: str,
    dataset_dir: str,
    output_dir: str,
    steps: int,
    tier: Tier,
    model: str = DEFAULT_MODEL,
    save_every: int = 250,
    sample_during_training: bool | None = None,
) -> str:
    """Generate the ai-toolkit training config for one character.

    `sample_during_training` overrides the tier default. Cloud runs pass
    False: see the comment at the call site in train_runpod.py.
    """
    caching = tier.cache_text_embeddings
    # Caption dropout needs a cached blank embedding; skip it when the text
    # encoder has already been discarded rather than risk a mid-run crash.
    dropout = 0.0 if caching else 0.05
    profile = MODELS[model]
    # MOSS trains with the shifted timestep schedule its ai-toolkit defaults use.
    timestep = '\n        timestep_type: "shift"' if model == "moss" else ""
    prompts = "\n".join(
        f"          - {_q(p)}" for p in sample_prompts(trigger_word, model)
    )
    max_keep = max(4, (steps // save_every) + 1)
    sample_res = max(tier.resolutions)
    sampling = (
        tier.sample_during_training
        if sample_during_training is None
        else sample_during_training
    )
    # Deliberately NOT equal to save_every. ai-toolkit saves the checkpoint and
    # then samples within the same step when the two line up, so a single failed
    # preview write kills the run at the exact step its first checkpoint appears.
    # Observed live (Aug 2026): OSError [Errno 5] at PIL's fp.close() writing a
    # preview into samples/.tmp/ took down a 2000-step run at step 250, before
    # anything had ever been collected. save_every + 1 is coprime with
    # save_every, so the two never coincide over any realistic run length.
    sample_every = save_every + 1

    return f"""---
# {profile.label} character LoRA - generated automatically, do not hand-edit.
# Hardware tier: {tier.key} ({tier.label})
# {tier.note}
job: extension
config:
  name: {_q(run_name)}
  process:
    - type: 'sd_trainer'
      training_folder: {_q(output_dir)}
      device: cuda:0
      # Also written into every caption file. When cache_text_embeddings is
      # on, ai-toolkit ignores this key, so the caption text is what counts.
      trigger_word: {_q(trigger_word)}
      network:
        type: "lora"
        linear: 32
        linear_alpha: 32
      save:
        dtype: float16
        save_format: "safetensors"
        save_every: {save_every}
        max_step_saves_to_keep: {max_keep}
      datasets:
        - folder_path: {_q(dataset_dir)}
          caption_ext: "txt"
          caption_dropout_rate: {dropout}
          cache_text_embeddings: {str(caching).lower()}
          shuffle_tokens: false
          cache_latents_to_disk: true
          resolution: {_yaml_list(tier.resolutions)}
      train:
        batch_size: 1
        steps: {steps}
        gradient_accumulation: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"{timestep}
        optimizer: "adamw8bit"
        lr: 1e-4
        dtype: bf16
        cache_text_embeddings: {str(caching).lower()}
        disable_sampling: {str(not sampling).lower()}
        skip_first_sample: true
      model:
{_model_block(tier, model)}
      sample:
        sampler: "flowmatch"
        sample_every: {sample_every}
        width: {sample_res}
        height: {sample_res}
        prompts:
{prompts}
        neg: ""
        seed: 42
        walk_seed: true
        guidance_scale: {profile.guidance_scale:g}
        sample_steps: {profile.sample_steps}
meta:
  name: "[name]"
  version: '1.0'
"""


def build_preview_config(
    *,
    run_name: str,
    trigger_word: str,
    dataset_dir: str,
    output_dir: str,
    lora_path: str,
    tier: Tier,
    model: str = DEFAULT_MODEL,
    sample_once: bool = False,
) -> str:
    """Generate a config that loads a finished LoRA and samples from it.

    On low-VRAM tiers, sampling cannot run during training: it is the step
    that reassembles the whole model, and cache_text_embeddings has already
    discarded the text encoder. So previews happen afterwards, in a fresh
    process with no optimizer state and no gradients to hold.

    This trains for a single throwaway step but takes its sample first
    (force_first_sample with skip_first_sample off), so the images reflect
    the LoRA exactly as trained.

    ai-toolkit also samples once more at the end of every job unless sampling
    is disabled outright, which would disable the first sample too
    (BaseSDTrainProcess, end of train loop). So by default every prompt is
    rendered twice. sample_once skips the first round and keeps only the end
    one, taken after the one step at lr 1e-8 - the same LoRA in practice.
    Cloud runs use it: the end round rendered fine on an 80GB card (Oct
    2026), and skipping the first halves the preview time. Local runs keep the default because the end sample runs with that
    step's optimizer state still allocated, which is untested on small cards.
    """
    profile = MODELS[model]
    # MOSS trains with the shifted timestep schedule its ai-toolkit defaults use.
    timestep = '\n        timestep_type: "shift"' if model == "moss" else ""
    prompts = "\n".join(
        f"          - {_q(p)}" for p in sample_prompts(trigger_word, model)
    )
    sample_res = min(1024, max(tier.resolutions))
    skip_first = "true" if sample_once else "false"
    force_first = "false" if sample_once else "true"

    return f"""---
# Preview-only pass. Loads the finished LoRA and renders sample images.
job: extension
config:
  name: {_q(run_name + "_preview")}
  process:
    - type: 'sd_trainer'
      training_folder: {_q(output_dir)}
      device: cuda:0
      trigger_word: {_q(trigger_word)}
      network:
        type: "lora"
        linear: 32
        linear_alpha: 32
        pretrained_lora_path: {_q(lora_path)}
      save:
        dtype: float16
        save_format: "safetensors"
        save_every: 1000000
        max_step_saves_to_keep: 1
      datasets:
        - folder_path: {_q(dataset_dir)}
          caption_ext: "txt"
          caption_dropout_rate: 0.0
          cache_text_embeddings: false
          cache_latents_to_disk: true
          resolution: [512]
      train:
        batch_size: 1
        steps: 1
        gradient_accumulation: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"{timestep}
        optimizer: "adamw8bit"
        lr: 1e-8
        dtype: bf16
        cache_text_embeddings: false
        disable_sampling: false
        skip_first_sample: {skip_first}
        force_first_sample: {force_first}
      model:
{_model_block(tier, model)}
      sample:
        sampler: "flowmatch"
        sample_every: 1000000
        sample_start_step: 0
        width: {sample_res}
        height: {sample_res}
        prompts:
{prompts}
        neg: ""
        seed: 42
        walk_seed: true
        guidance_scale: {profile.guidance_scale:g}
        sample_steps: {profile.sample_steps}
meta:
  name: "[name]"
  version: '1.0'
"""
