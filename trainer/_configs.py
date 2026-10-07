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


def sample_prompts(trigger_word: str) -> list[str]:
    return [
        f"{trigger_word}, a photograph, close-up portrait, soft natural window light",
        f"{trigger_word}, a photograph, waist-up shot, standing outdoors on a sunny day",
        f"{trigger_word}, a photograph, full body shot, casual clothes, city street",
        f"{trigger_word}, a photograph, close-up portrait, warm indoor light, smiling",
    ]


def _model_block(tier: Tier) -> str:
    lines = [
        f'        name_or_path: {_q("krea/Krea-2-Raw")}',
        '        arch: "krea2"',
    ]
    if tier.quantize:
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
    prompts = "\n".join(f"          - {_q(p)}" for p in sample_prompts(trigger_word))
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
# Krea 2 character LoRA - generated automatically, do not hand-edit.
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
        noise_scheduler: "flowmatch"
        optimizer: "adamw8bit"
        lr: 1e-4
        dtype: bf16
        cache_text_embeddings: {str(caching).lower()}
        disable_sampling: {str(not sampling).lower()}
        skip_first_sample: true
      model:
{_model_block(tier)}
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
        guidance_scale: 3
        sample_steps: 25
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
    prompts = "\n".join(f"          - {_q(p)}" for p in sample_prompts(trigger_word))
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
        noise_scheduler: "flowmatch"
        optimizer: "adamw8bit"
        lr: 1e-8
        dtype: bf16
        cache_text_embeddings: false
        disable_sampling: false
        skip_first_sample: {skip_first}
        force_first_sample: {force_first}
      model:
{_model_block(tier)}
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
        guidance_scale: 3
        sample_steps: 25
meta:
  name: "[name]"
  version: '1.0'
"""
