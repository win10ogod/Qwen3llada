# Qwen3LLada Project Overview

This project adapts Qwen3 into a Diffusion-style Language Model (Diffusion LM, DLM) for non‑autoregressive masked‑prediction learning, without making substantial changes to the core Qwen3 architecture. It reuses Qwen3 layers, weights, and the LM head, and performs forward passes with **non‑causal** attention. Training follows LLaDA’s design and incorporates reinforcement strategies from dLLM‑RL to provide a more stable and controllable training/inference pipeline.

## Goals & Design Principles

* **Goal:** Implement DLM and LLaDA‑like training/inference on top of the Qwen3 architecture, while adopting dLLM‑RL optimizations (mask scheduling, loss, sampling, etc.) to achieve more stable convergence and more consistent sampling.
* **Principle:** Keep Qwen3 non‑intrusive; achieve the architectural shift via a thin wrapper plus forward masks (non‑causal/custom additive bias).
* **Compatibility:** Maintain Hugging Face–style APIs and checkpoint formats (`save_pretrained` / `from_pretrained`).

## Key Features

* **Non‑causal attention + optional Additive Attention Bias**

  * Default non‑causal operation (`attention_mask=None`). Supports user‑provided additive bias with shape `[B, 1, Q, K]` (0 = visible, −inf = masked), enabling “block attention bias during training” and other advanced visibility policies.
* **Three switchable training losses**

  * **guidelines:** Per the LLaDA paper, apply cross‑entropy (CE) over masked positions with CE/p\_mask and (b\*l) normalization.
  * **dllm:** Adapted from dLLM‑RL’s LLaDA loss (masked positions determined by `noisy == mask_id`), providing stronger stability.
  * **soft\_ce:** Masked CE with label smoothing, divided by p\_mask (inspired by dLLM‑RL’s soft‑CE), improving stability.
  * All CE computations run in **float32** (even when bf16/fp16 is enabled) to avoid numerical instability.
* **Masking/Noise scheduling (dLLM‑RL style, pretraining)**

  * `schedule`: `uniform` / `cosine` / `power` / `sigmoid`
  * `noise_type`: `mask` or `random_replace`
  * `min_masking_rate`: clamps tiny p\_mask values to stabilize 1/p weighting
  * `mask_contiguous_region_prob`: probability of contiguous span masking
  * `reweight_cap`: upper bound for 1/p\_mask
  * Randomly shorten **1%** of sequences (as recommended by LLaDA).
* **Block attention bias during training (optional)**

  * Consistent with semi‑autoregressive padding sampling: prompt tokens are mutually visible; within the answer region, each block only attends within itself (while still attending to the prompt).
  * Can be enabled in both training and inference to improve consistency and stability.
* **Inference (sampling)**

  * `remasking`: `low_confidence` / `random` / `random_topk` (stochastic low‑confidence selection)
  * **Block generation:** `--block_length` controls chunking; `--block_attention` aligns with the training bias.
  * Supports **CFG** (classifier‑free guidance).
* **Optimizers & LR scheduling**

  * Optimizers: `AdamW`, `AdamW8bit`, `Lion8bit` (automatic fallback)
  * LR: linear / cosine with warmup (via `transformers`).
* **Checkpoints & resumption**

  * Periodically saves to `{output_dir}/checkpoints/step-{N}` including `model`, `tokenizer`, `optimizer.pt`, and `trainer_state.json` (epoch/global\_step/epoch\_batch).
  * Resume from `latest` or a specified checkpoint; supports skipping batches within an epoch for true mid‑epoch continuation.
* **Hugging Face Datasets integration**

  * **Pretraining:** Tokenize from generic text fields (auto‑detected). In non‑streaming mode, sequences are packed to a fixed length.
  * **SFT:** Supports the HF conversation format (see below) and produces `prompt_lengths`. Training follows the LLaDA SFT loss design.

## Directory Layout & Core Files

* `modeling/qwen3llada/`

  * `configuration_qwen3llada.py`: `Qwen3LLadaConfig` (default `use_cache = False`).
  * `modeling_qwen3llada.py`: `Qwen3LLadaModel` / `Qwen3LLadaForMaskedLM`; non‑causal attention + additive bias.
* `scripts/`

  * `train.py`: Training entry point; supports pretraining/SFT, block bias, three loss variants, LR scheduler, checkpoints.
  * `inference.py`: Inference entry point; supports `block_attention`, `random_topk` remasking, CFG.
  * `loss.py`: Implementations of `guidelines` / `dllm` / `soft_ce` losses and forward‑time noising.
  * `convert.py`: Weight mapping Qwen3 → Qwen3LLada (copy without value changes).
  * `analyze.py`: Key/shape comparison between Qwen3 and Qwen3LLada weights.
* `config.yaml`: Complete configuration (model, training, datasets, evaluation, optimizer, noise, and block bias).

## SFT Data Format (HF Import)

Supports the following conversation schema (each record needs at least one human→gpt pair):

```json
[
  {
    "conversations": [
      {"from": "human", "value": "Human instruction"},
      {"from": "gpt",   "value": "Model answer"}
    ],
    "system": "System prompt (optional)"
  }
]
```

* **Mapping rules:** `BOS + system + human` compose the **prompt**; `gpt` is the **answer**; append `EOS` if present, and record `prompt_lengths`.
* During SFT training, **no noise** is added to the prompt region; loss is normalized by **answer length** (per the LLaDA SFT design).

## Quick Start

1. **Conversion (optional)**

```bash
python scripts/convert.py --src demo_model/Qwen3-0.6B-Base --dst out/qwen3llada-converted
```

2. **Pretraining**

```bash
python scripts/train.py --config config.yaml
```

3. **SFT (HF conversation format)**

* In `config.yaml`, set:

  * `training.mode: sft`
  * `dataset.name: <your HF dataset>` (containing `conversations` / `system`)

```bash
python scripts/train.py --config config.yaml
```

4. **Inference**

```bash
python scripts/inference.py \
  --model out/qwen3llada-base \
  --prompt "What is diffusion LM?" \
  --steps 128 --gen_length 128 --block_length 32 \
  --remasking random_topk --block_attention
```

## Configuration Highlights (`config.yaml`)

* `training.loss_impl`: `guidelines` | `dllm` | `soft_ce`
* `training.noise`: `schedule` (uniform/cosine/power/sigmoid), `noise_type` (mask/random\_replace), `min_masking_rate`, `mask_contiguous_region_prob`, `reweight_cap`, `random_length_prob`
* `training.block_attention.enabled`: `true`/`false`; `block_length`: keep consistent with inference
* `optim.name`: `adamw` | `adamw8bit` | `lion8bit` (auto‑fallback to AdamW)
* `evaluation.do_eval_loss`: `true` → run eval loss every `eval_every_steps` (same loss path as training)

## Notes

* Diffusion models do **not** use a KV‑cache; `Qwen3LLadaConfig` defaults to `use_cache = False`.
* In non‑streaming mode, packing to a fixed length helps stabilize (b\*l) normalization of the loss; packing is **not recommended** for the SFT path.
* On Windows, `bitsandbytes` may be unavailable; the program automatically falls back.
* Computing CE in **float32** is more stable (even when bf16/fp16 is enabled).

## Future Extensions

* More advanced masking strategies (two‑stage masking; comp/random hybrids), and toggles that mirror dLLM‑RL’s RL/SFT flows.
* Layered or alternating block‑attention strategies (SDAR‑like per‑layer planning).
* Richer sampling policies (dynamic‑threshold unmasking; confidence‑temperature mixing).
