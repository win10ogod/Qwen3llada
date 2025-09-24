import argparse
import os
import sys
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import json
from glob import glob

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import os as _os
_os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
_os.environ.setdefault("HF_TRANSFORMERS_NO_TF", "1")
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from transformers import AutoTokenizer
import yaml
from tqdm import tqdm
from typing import Iterable

# Ensure project root on sys.path for `modeling` imports when running as a script
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from modeling.qwen3llada import Qwen3LLadaForMaskedLM, Qwen3LLadaConfig
from transformers import AutoModelForCausalLM
try:
    from scripts.loss import forward_process, diffusion_ce_loss, sft_forward, sft_ce_loss
except ModuleNotFoundError:
    from loss import forward_process, diffusion_ce_loss, sft_forward, sft_ce_loss


def _sched_pmask(t: torch.Tensor, eps: float, schedule: str) -> torch.Tensor:
    if schedule == 'cosine':
        # Cosine schedule in [0,1], as used in some DLMs
        s = 0.5 * (1.0 - torch.cos(torch.pi * t))
    elif schedule == 'power':
        # power schedule (mask ratio ~ 1 - t^2)
        s = t**2
    elif schedule == 'sigmoid':
        # sigmoid-shaped schedule
        s = 1.0 / (1.0 + torch.exp(-12 * (t - 0.5)))
    else:
        s = t
    return (1 - eps) * s + eps


def forward_process_cfg(input_ids: torch.LongTensor, mask_token_id: int, cfg: Any, vocab_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, l = input_ids.shape
    device = input_ids.device
    noise_cfg = (cfg.__dict__.get('noise', {}) or {})
    eps = noise_cfg.get('eps', 1e-3)
    schedule = noise_cfg.get('schedule', 'uniform')
    noise_type = noise_cfg.get('noise_type', 'mask')
    min_masking_rate = float(noise_cfg.get('min_masking_rate', 0.0) or 0.0)
    span_prob = float(noise_cfg.get('mask_contiguous_region_prob', 0.0) or 0.0)
    reweight_cap = float(noise_cfg.get('reweight_cap', 0.0) or 0.0)

    t = torch.rand(b, device=device)
    p = _sched_pmask(t, eps, schedule)
    if min_masking_rate > 0.0:
        p = torch.clamp(p, min=min_masking_rate)
    p_mask = p[:, None].repeat(1, l)
    # choose between scattered and contiguous spans
    if span_prob > 0.0:
        masked_indices = torch.zeros((b, l), device=device, dtype=torch.bool)
        span_choice = torch.rand(b, device=device) < span_prob
        for i in range(b):
            if span_choice[i]:
                m = max(1, int(round(float(l) * float(p[i].item()))))
                if m >= l:
                    masked_indices[i, :] = True
                else:
                    start = torch.randint(0, l - m + 1, (1,), device=device).item()
                    masked_indices[i, start:start+m] = True
            else:
                masked_indices[i] = (torch.rand(l, device=device) < p[i])
    else:
        masked_indices = torch.rand((b, l), device=device) < p_mask
    # ensure at least one masked token per sample
    ensure = ~masked_indices.any(dim=1)
    if ensure.any():
        rand_pos = torch.randint(0, l, (int(ensure.sum().item()),), device=device)
        masked_indices[ensure, :] = False
        masked_indices[ensure, rand_pos] = True

    if noise_type == 'random_replace':
        # sample random tokens uniformly from vocab; for simplicity we do not exclude specials
        rand_tokens = torch.randint(0, vocab_size, (b, l), device=device)
        noisy_batch = torch.where(masked_indices, rand_tokens, input_ids)
    else:
        noisy_batch = torch.where(masked_indices, torch.as_tensor(mask_token_id, device=device), input_ids)
    # Optional cap on 1/p reweight: enforce p_mask >= 1/reweight_cap
    if reweight_cap and reweight_cap > 0.0:
        p_mask = torch.clamp(p_mask, min=1.0/float(reweight_cap))
    return noisy_batch, masked_indices, p_mask


def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@dataclass
class TrainConfig:
    seed: int
    output_dir: str
    log_dir: str
    device: str

    base_model_path: str
    qwen3llada_model_path: str
    mask_token_id: int

    mode: str
    loss_impl: str
    loss_smoothing: float
    epochs: int
    max_steps: int
    batch_size: int
    micro_batch_size: int
    max_length: int
    lr: float
    weight_decay: float
    warmup_ratio: float
    gradient_accumulation_steps: int
    fp16: bool
    bf16: bool
    save_steps: int
    save_total_limit: int
    resume_from: str
    # optimizer
    optim_name: str
    betas: List[float]
    eps: float

    dataset_name: str
    dataset_config: Optional[str]
    dataset_split: str
    text_field: str
    streaming: bool
    num_proc: int
    dataset_pack_to_max_length: bool

    eval_steps: int
    do_eval_loss: bool
    eval_every_steps: int
    eval_split: str
    eval_max_batches: int
    max_new_tokens: int
    sampling_steps: int
    block_length: int
    temperature: float
    remasking: str
    # block attention during training
    block_attention_enabled: bool
    block_attention_block_length: int


def load_config(path: str) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return TrainConfig(
        seed=cfg["project"]["seed"],
        output_dir=cfg["project"]["output_dir"],
        log_dir=cfg["project"]["log_dir"],
        device=cfg["project"].get("device", "auto"),
        base_model_path=cfg["model"]["base_model_path"],
        qwen3llada_model_path=cfg["model"].get("qwen3llada_model_path", ""),
        mask_token_id=cfg["model"].get("mask_token_id", 126336),
        mode=cfg["training"]["mode"],
        loss_impl=cfg["training"].get("loss_impl", "dllm"),
        loss_smoothing=cfg["training"].get("loss_smoothing", 0.1),
        epochs=cfg["training"]["epochs"],
        max_steps=cfg["training"].get("max_steps", -1),
        batch_size=cfg["training"]["batch_size"],
        micro_batch_size=cfg["training"].get("micro_batch_size", cfg["training"]["batch_size"]),
        max_length=cfg["training"]["max_length"],
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        warmup_ratio=cfg["training"].get("warmup_ratio", 0.01),
        gradient_accumulation_steps=cfg["training"].get("gradient_accumulation_steps", 1),
        fp16=cfg["training"].get("fp16", False),
        bf16=cfg["training"].get("bf16", False),
        save_steps=cfg["training"].get("save_steps", 0),
        save_total_limit=cfg["training"].get("save_total_limit", 2),
        resume_from=cfg["training"].get("resume_from", ""),
        optim_name=(cfg.get("optim", {}) or {}).get("name", "adamw").lower(),
        betas=(cfg.get("optim", {}) or {}).get("betas", [0.9, 0.95]),
        eps=(cfg.get("optim", {}) or {}).get("eps", 1.0e-8),
        dataset_name=cfg["dataset"]["name"],
        dataset_config=cfg["dataset"].get("config"),
        dataset_split=cfg["dataset"].get("split", "train"),
        text_field=cfg["dataset"].get("text_field", "text"),
        streaming=cfg["dataset"].get("streaming", False),
        num_proc=cfg["dataset"].get("num_proc", 2),
        dataset_pack_to_max_length=cfg["dataset"].get("pack_to_max_length", True),
        eval_steps=cfg["evaluation"].get("steps", 200),
        do_eval_loss=cfg["evaluation"].get("do_eval_loss", True),
        eval_every_steps=cfg["evaluation"].get("eval_every_steps", 200),
        eval_split=cfg["evaluation"].get("eval_split", "train[:1%]"),
        eval_max_batches=cfg["evaluation"].get("eval_max_batches", 50),
        max_new_tokens=cfg["evaluation"].get("max_new_tokens", 64),
        sampling_steps=cfg["evaluation"].get("steps", 64),
        block_length=cfg["evaluation"].get("block_length", 32),
        temperature=cfg["evaluation"].get("temperature", 0.0),
        remasking=cfg["evaluation"].get("remasking", "low_confidence"),
        block_attention_enabled=((cfg["training"].get("block_attention", {}) or {}).get("enabled", False)),
        block_attention_block_length=((cfg["training"].get("block_attention", {}) or {}).get("block_length", cfg["evaluation"].get("block_length", 32))),
    )


def prepare_dataset(cfg: TrainConfig, tokenizer: AutoTokenizer):
    ds = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.dataset_split, streaming=cfg.streaming)

    if cfg.mode == "sft":
        # HF SFT format: { conversations: [{from: human, value: ...}, {from: gpt, value: ...}], system: optional }
        def to_sft_batch(examples: Dict[str, List[Dict]]):
            input_ids_list: List[List[int]] = []
            prompt_lengths: List[int] = []
            bos_id = tokenizer.bos_token_id
            eos_id = tokenizer.eos_token_id
            convs_list = examples.get("conversations", [])
            systems = examples.get("system", [None] * len(convs_list))
            for convs, system in zip(convs_list, systems):
                human_text, gpt_text = None, None
                if isinstance(convs, list):
                    for turn in convs:
                        role = turn.get("from")
                        if human_text is None and role in ("human", "user"):
                            human_text = str(turn.get("value", ""))
                        elif gpt_text is None and role in ("gpt", "assistant"):
                            gpt_text = str(turn.get("value", ""))
                        if human_text is not None and gpt_text is not None:
                            break
                if human_text is None or gpt_text is None:
                    continue
                prompt_str = (str(system) + "\n" if system else "") + human_text
                answer_str = gpt_text
                ids_prompt = tokenizer(prompt_str, add_special_tokens=False, max_length=cfg.max_length, truncation=True)["input_ids"]
                # budget for answer and special tokens
                specials = (1 if bos_id is not None else 0) + (1 if eos_id is not None else 0)
                remaining = max(0, cfg.max_length - (len(ids_prompt) + specials))
                ids_answer = tokenizer(answer_str, add_special_tokens=False, max_length=remaining, truncation=True)["input_ids"]
                seq: List[int] = []
                if bos_id is not None:
                    seq.append(bos_id)
                seq.extend(ids_prompt)
                prompt_len = len(seq)
                seq.extend(ids_answer)
                if eos_id is not None and len(seq) < cfg.max_length:
                    seq.append(eos_id)
                seq = seq[: cfg.max_length]
                prompt_len = min(prompt_len, len(seq))
                input_ids_list.append(seq)
                prompt_lengths.append(prompt_len)
            return {"input_ids": input_ids_list, "prompt_lengths": prompt_lengths}

        if cfg.streaming:
            # streaming: map item by item
            def map_one(ex):
                out = to_sft_batch({"conversations": [ex["conversations"]], "system": [ex.get("system")]})
                # unwrap single
                return {"input_ids": out["input_ids"][0], "prompt_lengths": out["prompt_lengths"][0]}
            ds = ds.map(map_one)
        else:
            ds = ds.map(to_sft_batch, batched=True, num_proc=cfg.num_proc, remove_columns=ds.column_names)

        def collate(batch: List[Dict[str, List[int]]]):
            input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
            prompt_lens = torch.tensor([int(x["prompt_lengths"]) for x in batch], dtype=torch.long)
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
            return {"input_ids": input_ids, "prompt_lengths": prompt_lens}

        return ds, collate

    # Pretraining path
    def infer_text_field():
        preferred = cfg.text_field if cfg.text_field not in (None, "", "auto") else None
        if not cfg.streaming and hasattr(ds, "features") and preferred and preferred in ds.features:
            return preferred
        try:
            sample = next(iter(ds)) if cfg.streaming else ds[0]
        except Exception:
            sample = None
        candidates = [preferred] if preferred else []
        candidates += ["text", "content", "raw", "document", "content_text"]
        for k in candidates:
            if not k:
                continue
            if sample is not None and k in sample and (isinstance(sample[k], str) or (isinstance(sample[k], list) and sample[k] and isinstance(sample[k][0], str))):
                return k
        if sample is not None:
            for k, v in sample.items():
                if isinstance(v, str) or (isinstance(v, list) and v and isinstance(v[0], str)):
                    return k
        raise ValueError("Could not infer text field; please set dataset.text_field in config.yaml")

    resolved_text_field = infer_text_field()

    def tokenize_fn(examples: Dict[str, List[str]]):
        texts = examples[resolved_text_field]
        toks = tokenizer(texts, truncation=True, max_length=cfg.max_length, padding=False, add_special_tokens=True)
        return {"input_ids": toks["input_ids"]}

    if cfg.streaming:
        ds = ds.map(tokenize_fn)
    else:
        ds = ds.map(tokenize_fn, batched=True, num_proc=cfg.num_proc, remove_columns=ds.column_names)
        if cfg.dataset_pack_to_max_length:
            block_size = cfg.max_length
            def group_texts(examples):
                concatenated = sum(examples["input_ids"], [])
                total_length = (len(concatenated) // block_size) * block_size
                return {"input_ids": [concatenated[i : i + block_size] for i in range(0, total_length, block_size)]}
            ds = ds.map(group_texts, batched=True, num_proc=cfg.num_proc)

    def collate(batch: List[Dict[str, List[int]]]):
        input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        noise_cfg = (cfg.__dict__.get('noise', {}) or {})
        if random.random() < noise_cfg.get('random_length_prob', 0.01):
            random_length = random.randint(1, min(cfg.max_length, input_ids.shape[1]))
            input_ids = input_ids[:, :random_length]
        return {"input_ids": input_ids}

    return ds, collate


def prepare_eval_dataset(cfg: TrainConfig, tokenizer: AutoTokenizer):
    """Prepare evaluation dataset using cfg.evaluation split string.
    Mirrors prepare_dataset but always non-streaming and small subset.
    """
    eval_ds = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.eval_split, streaming=False)

    # Infer text field from small sample
    def infer_text_field(sample):
        preferred = cfg.text_field if cfg.text_field not in (None, "", "auto") else None
        if hasattr(eval_ds, "features") and preferred and preferred in eval_ds.features:
            return preferred
        candidates = [preferred] if preferred else []
        candidates += ["text", "content", "raw", "document", "content_text"]
        for k in candidates:
            if k and k in sample and (isinstance(sample[k], str) or (isinstance(sample[k], list) and sample[k] and isinstance(sample[k][0], str))):
                return k
        for k, v in sample.items():
            if isinstance(v, str) or (isinstance(v, list) and v and isinstance(v[0], str)):
                return k
        raise ValueError("Could not infer text field for eval dataset")

    sample = eval_ds[0] if len(eval_ds) > 0 else next(iter(eval_ds))
    resolved_text_field = infer_text_field(sample)

    if cfg.mode == "sft":
        def to_sft_batch(examples: Dict[str, List[Dict]]):
            input_ids_list: List[List[int]] = []
            prompt_lengths: List[int] = []
            bos_id = tokenizer.bos_token_id
            eos_id = tokenizer.eos_token_id
            convs_list = examples.get("conversations", [])
            systems = examples.get("system", [None] * len(convs_list))
            for convs, system in zip(convs_list, systems):
                human_text, gpt_text = None, None
                if isinstance(convs, list):
                    for turn in convs:
                        role = turn.get("from")
                        if human_text is None and role in ("human", "user"):
                            human_text = str(turn.get("value", ""))
                        elif gpt_text is None and role in ("gpt", "assistant"):
                            gpt_text = str(turn.get("value", ""))
                        if human_text is not None and gpt_text is not None:
                            break
                if human_text is None or gpt_text is None:
                    continue
                prompt_str = (str(system) + "\n" if system else "") + human_text
                answer_str = gpt_text
                ids_prompt = tokenizer(prompt_str, add_special_tokens=False, max_length=cfg.max_length, truncation=True)["input_ids"]
                specials = (1 if bos_id is not None else 0) + (1 if eos_id is not None else 0)
                remaining = max(0, cfg.max_length - (len(ids_prompt) + specials))
                ids_answer = tokenizer(answer_str, add_special_tokens=False, max_length=remaining, truncation=True)["input_ids"]
                seq: List[int] = []
                if bos_id is not None:
                    seq.append(bos_id)
                seq.extend(ids_prompt)
                prompt_len = len(seq)
                seq.extend(ids_answer)
                if eos_id is not None and len(seq) < cfg.max_length:
                    seq.append(eos_id)
                seq = seq[: cfg.max_length]
                prompt_len = min(prompt_len, len(seq))
                input_ids_list.append(seq)
                prompt_lengths.append(prompt_len)
            return {"input_ids": input_ids_list, "prompt_lengths": prompt_lengths}

        eval_ds = eval_ds.map(to_sft_batch, batched=True, num_proc=min(2, cfg.num_proc), remove_columns=eval_ds.column_names)
    else:
        def tokenize_fn(examples: Dict[str, List[str]]):
            texts = examples[resolved_text_field]
            toks = tokenizer(texts, truncation=True, max_length=cfg.max_length, padding=False, add_special_tokens=True)
            return {"input_ids": toks["input_ids"]}
        eval_ds = eval_ds.map(tokenize_fn, batched=True, num_proc=min(2, cfg.num_proc), remove_columns=eval_ds.column_names)

    if cfg.dataset_pack_to_max_length:
        block_size = cfg.max_length

        def group_texts(examples):
            concatenated = sum(examples["input_ids"], [])
            total_length = (len(concatenated) // block_size) * block_size
            return {"input_ids": [concatenated[i : i + block_size] for i in range(0, total_length, block_size)]}

        eval_ds = eval_ds.map(group_texts, batched=True, num_proc=min(2, cfg.num_proc))

    def collate(batch: List[Dict[str, List[int]]]):
        input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        out = {"input_ids": input_ids}
        if cfg.mode == "sft":
            out["prompt_lengths"] = torch.tensor([int(x["prompt_lengths"]) for x in batch], dtype=torch.long)
        return out

    return eval_ds, collate


def build_block_attention_bias(input_ids: torch.Tensor, block_length: int, prompt_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Build additive attention mask [B,1,Q,K] allowing prompt to be fully visible and
    answer tokens to attend only within their own block (plus prompt).
    Returns float mask with 0 for visible and -inf for masked positions.
    """
    device = input_ids.device
    B, L = input_ids.shape
    finfo_min = torch.finfo(torch.float32).min
    bias = torch.full((B, 1, L, L), 0.0, dtype=torch.float32, device=device)
    # initialize as fully visible, then mask out-of-block if needed
    if block_length is None or block_length <= 0:
        return bias
    for b in range(B):
        p = int(prompt_lengths[b].item()) if prompt_lengths is not None else 0
        for q in range(L):
            if q < p:
                # prompt token: can see prompt fully
                # no masking needed; optionally allow to see all tokens, we keep as full
                continue
            # compute its block in answer region
            start = p + ((q - p) // block_length) * block_length
            end = min(start + block_length, L)
            # allow prompt keys
            # mask keys outside prompt and outside current block
            if start > p:
                # mask [p, start)
                bias[b, 1-1, q, p:start] = finfo_min
            if end < L:
                # mask [end, L)
                bias[b, 1-1, q, end:L] = finfo_min
    return bias


@torch.no_grad()
def evaluate_loss(model: torch.nn.Module, eval_ds, collate_fn, device: torch.device, cfg: TrainConfig) -> float:
    dl = DataLoader(eval_ds, batch_size=cfg.micro_batch_size, shuffle=False, collate_fn=collate_fn)
    was_training = model.training
    model.eval()
    total = 0.0
    steps = 0
    for i, batch in enumerate(dl):
        input_ids = batch["input_ids"].to(device)
        if cfg.mode == "pretrain":
            if cfg.loss_impl == "dllm":
                noisy_batch, masked_indices, p_mask = forward_process_cfg(input_ids, cfg.mask_token_id, cfg, model.config.vocab_size)
            else:
                noisy_batch, masked_indices, p_mask = forward_process(input_ids, cfg.mask_token_id)
            attn_bias = None
            if cfg.block_attention_enabled:
                attn_bias = build_block_attention_bias(noisy_batch, cfg.block_attention_block_length)
            logits = model(input_ids=noisy_batch, attention_mask=attn_bias).logits
            if cfg.loss_impl == "dllm":
                from scripts.loss import dllm_llada_pretrain_loss
                loss = dllm_llada_pretrain_loss(logits, input_ids, masked_indices, p_mask)
            elif cfg.loss_impl == "soft_ce":
                from scripts.loss import soft_label_smoothing_loss
                loss = soft_label_smoothing_loss(logits, input_ids, masked_indices, p_mask, smoothing=cfg.loss_smoothing)
            else:
                loss = diffusion_ce_loss(logits, input_ids, masked_indices, p_mask)
        else:
            prompt_lengths = batch.get("prompt_lengths", torch.zeros(input_ids.size(0), dtype=torch.long, device=input_ids.device))
            noisy_batch, masked_indices, p_mask, answer_lengths = sft_forward(input_ids, prompt_lengths, cfg.mask_token_id)
            attn_bias = None
            if cfg.block_attention_enabled:
                attn_bias = build_block_attention_bias(noisy_batch, cfg.block_attention_block_length, prompt_lengths)
            logits = model(input_ids=noisy_batch, attention_mask=attn_bias).logits
            loss = sft_ce_loss(logits, input_ids, masked_indices, p_mask, answer_lengths)
        total += float(loss.item())
        steps += 1
        if steps >= max(1, cfg.eval_max_batches):
            break
    if was_training:
        model.train()
    return total / max(1, steps)


def build_model_and_tokenizer(cfg: TrainConfig, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        # ensure pad token exists
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token

    # Build model: load qwen3llada checkpoint if provided; otherwise create from Qwen3 and copy weights
    qll_path = cfg.qwen3llada_model_path
    valid_qll = bool(qll_path) and os.path.exists(os.path.join(qll_path, "config.json"))
    if valid_qll:
        try:
            model = Qwen3LLadaForMaskedLM.from_pretrained(qll_path)
        except Exception as e:
            print(f"[warn] failed to load qwen3llada checkpoint at '{qll_path}': {e}\nFalling back to base model conversion.")
            valid_qll = False
    if not valid_qll:
        # Load base Qwen3 and wrap
        base = AutoModelForCausalLM.from_pretrained(cfg.base_model_path)
        qcfg = Qwen3LLadaConfig(**base.config.to_dict())
        qcfg.mask_token_id = cfg.mask_token_id
        model = Qwen3LLadaForMaskedLM(qcfg)
        # Map state dict from Qwen3ForCausalLM -> Qwen3LLadaForMaskedLM
        sd = base.state_dict()
        new_sd = {}
        for k, v in sd.items():
            if k.startswith("model."):
                new_sd["backbone.backbone." + k[len("model."):]] = v
            elif k.startswith("lm_head."):
                new_sd["lm_head." + k[len("lm_head."):]] = v
            else:
                # ignore others (e.g., buffers)
                pass
        missing, unexpected = model.load_state_dict(new_sd, strict=False)
        if missing:
            print(f"[warn] missing keys: {missing[:5]} ... total={len(missing)}")
        if unexpected:
            print(f"[warn] unexpected keys: {unexpected[:5]} ... total={len(unexpected)}")

    dtype = torch.bfloat16 if cfg.bf16 else (torch.float16 if cfg.fp16 else torch.float32)
    model.to(device=device, dtype=dtype)
    model.train()
    return model, tokenizer


def save_checkpoint(model, tokenizer, optimizer, ckpt_dir: str, step: int, state: dict, rotate_limit: int = 0):
    os.makedirs(ckpt_dir, exist_ok=True)
    this_dir = os.path.join(ckpt_dir, f"step-{step}")
    os.makedirs(this_dir, exist_ok=True)
    model.save_pretrained(this_dir)
    tokenizer.save_pretrained(this_dir)
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(this_dir, "optimizer.pt"))
    with open(os.path.join(this_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f)

    # rotation
    if rotate_limit and rotate_limit > 0:
        all_ckpts = sorted(glob(os.path.join(ckpt_dir, "step-*")), key=lambda p: int(p.split("step-")[-1]))
        excess = len(all_ckpts) - rotate_limit
        for i in range(excess):
            try:
                import shutil
                shutil.rmtree(all_ckpts[i], ignore_errors=True)
            except Exception:
                pass

def find_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    if not os.path.isdir(ckpt_dir):
        return None
    paths = glob(os.path.join(ckpt_dir, "step-*"))
    if not paths:
        return None
    latest = max(paths, key=lambda p: int(p.split("step-")[-1]))
    return latest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    device = get_device(cfg.device)
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    model, tokenizer = build_model_and_tokenizer(cfg, device)
    ds, collate = prepare_dataset(cfg, tokenizer)

    # resume support
    ckpt_root = os.path.join(cfg.output_dir, "checkpoints")
    start_epoch = 0
    start_epoch_batch = 0  # number of batches already processed within the start_epoch
    global_step = 0
    resume_path = None
    if cfg.resume_from:
        resume_path = cfg.resume_from
        if cfg.resume_from == "latest":
            rp = find_latest_checkpoint(ckpt_root)
            if rp:
                resume_path = rp
        if resume_path and os.path.isdir(resume_path):
            try:
                # reload model/tokenizer for consistency
                model = Qwen3LLadaForMaskedLM.from_pretrained(resume_path).to(device=device, dtype=model.dtype)
                tokenizer = AutoTokenizer.from_pretrained(resume_path, use_fast=True)
                # load state
                st_path = os.path.join(resume_path, "trainer_state.json")
                if os.path.isfile(st_path):
                    with open(st_path, "r", encoding="utf-8") as f:
                        st = json.load(f)
                    start_epoch = int(st.get("epoch", 0))
                    global_step = int(st.get("global_step", 0))
                    start_epoch_batch = int(st.get("epoch_batch", 0))
                print(f"Resumed from checkpoint: {resume_path} (epoch={start_epoch}, step={global_step}, epoch_batch={start_epoch_batch})")
            except Exception as e:
                print(f"[warn] Failed to resume from '{resume_path}': {e}")

    def _dataloader():
        return DataLoader(ds, batch_size=cfg.micro_batch_size, shuffle=not cfg.streaming, collate_fn=collate)

    # Optimizer (supports AdamW, AdamW8bit, Lion8bit)
    def build_optimizer():
        params = [p for p in model.parameters() if p.requires_grad]
        betas = tuple(cfg.betas) if isinstance(cfg.betas, (list, tuple)) else (0.9, 0.95)
        if cfg.optim_name == "adamw8bit":
            try:
                import bitsandbytes as bnb
                return bnb.optim.AdamW8bit(params, lr=cfg.lr, betas=betas, eps=cfg.eps, weight_decay=cfg.weight_decay)
            except Exception as e:
                print(f"[warn] AdamW8bit requested but bitsandbytes not available: {e}. Falling back to AdamW.")
        if cfg.optim_name == "lion8bit":
            try:
                import bitsandbytes as bnb
                return bnb.optim.Lion8bit(params, lr=cfg.lr, betas=betas, weight_decay=cfg.weight_decay)
            except Exception as e:
                print(f"[warn] Lion8bit requested but bitsandbytes not available: {e}. Falling back to AdamW.")
        return torch.optim.AdamW(params, lr=cfg.lr, betas=betas, eps=cfg.eps, weight_decay=cfg.weight_decay)

    optimizer = build_optimizer()
    # optional LR scheduler (linear/cosine/constant with warmup)
    total_steps = cfg.max_steps if cfg.max_steps and cfg.max_steps > 0 else None
    scheduler = None
    try:
        from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
        warmup_steps = int(cfg.warmup_ratio * (total_steps or 1000))
        sched_name = (getattr(cfg, 'optim_name', 'adamw') or 'adamw').lower()  # reuse field for minimal config change
        # Prefer cosine if present in config via optim.scheduler; fallback to linear
        optim_cfg = getattr(cfg, '__dict__', {})
        sched = (optim_cfg.get('optim', {}) or {}).get('scheduler', 'linear') if isinstance(optim_cfg.get('optim', {}), dict) else 'linear'
        if sched == 'cosine' and total_steps:
            scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        elif sched == 'linear' and total_steps:
            scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    except Exception:
        scheduler = None
    # Load optimizer state if resuming
    if resume_path:
        opt_path = os.path.join(resume_path, "optimizer.pt")
        if os.path.isfile(opt_path):
            try:
                optimizer.load_state_dict(torch.load(opt_path, map_location=device))
                print("Loaded optimizer state from checkpoint.")
            except Exception as e:
                print(f"[warn] Failed to load optimizer state: {e}")

    # if resuming, keep global_step as loaded
    running_loss = 0.0
    start_time = time.time()

    for epoch in range(start_epoch, cfg.epochs):
        dl = _dataloader()
        pbar = tqdm(dl, desc=f"epoch {epoch+1}")
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(pbar):
            # Skip already-processed batches if resuming within the same epoch
            if epoch == start_epoch and step < start_epoch_batch:
                continue
            input_ids = batch["input_ids"].to(device)

            if cfg.mode == "pretrain":
                if cfg.loss_impl == "dllm":
                    noisy_batch, masked_indices, p_mask = forward_process_cfg(input_ids, cfg.mask_token_id, cfg, model.config.vocab_size)
                else:
                    noisy_batch, masked_indices, p_mask = forward_process(input_ids, cfg.mask_token_id)
                attn_bias = None
                if cfg.block_attention_enabled:
                    attn_bias = build_block_attention_bias(noisy_batch, cfg.block_attention_block_length)
                logits = model(input_ids=noisy_batch, attention_mask=attn_bias).logits
                if cfg.loss_impl == "dllm":
                    from scripts.loss import dllm_llada_pretrain_loss
                    loss = dllm_llada_pretrain_loss(logits, input_ids, masked_indices, p_mask)
                elif cfg.loss_impl == "soft_ce":
                    from scripts.loss import soft_label_smoothing_loss
                    loss = soft_label_smoothing_loss(logits, input_ids, masked_indices, p_mask, smoothing=cfg.loss_smoothing)
                else:
                    loss = diffusion_ce_loss(logits, input_ids, masked_indices, p_mask)
            elif cfg.mode == "sft":
                # Expect prompt_lengths stored elsewhere; fallback to full noise if absent
                prompt_lengths = batch.get("prompt_lengths", torch.zeros(input_ids.size(0), dtype=torch.long, device=input_ids.device))
                noisy_batch, masked_indices, p_mask, answer_lengths = sft_forward(input_ids, prompt_lengths, cfg.mask_token_id)
                attn_bias = None
                if cfg.block_attention_enabled:
                    attn_bias = build_block_attention_bias(noisy_batch, cfg.block_attention_block_length, prompt_lengths)
                logits = model(input_ids=noisy_batch, attention_mask=attn_bias).logits
                loss = sft_ce_loss(logits, input_ids, masked_indices, p_mask, answer_lengths)
            else:
                raise ValueError(f"Unknown training.mode: {cfg.mode}")

            (loss / cfg.gradient_accumulation_steps).backward()
            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            running_loss += loss.item()
            if (global_step % max(1, cfg.eval_steps)) == 0 and global_step > 0:
                avg = running_loss / max(1, cfg.eval_steps)
                pbar.set_postfix({"loss": f"{avg:.4f}", "step": global_step})
                running_loss = 0.0

            # periodic eval loss
            if cfg.do_eval_loss and cfg.eval_every_steps and global_step > 0 and (global_step % cfg.eval_every_steps == 0):
                try:
                    eval_ds, eval_collate = prepare_eval_dataset(cfg, tokenizer)
                    eval_loss = evaluate_loss(model, eval_ds, eval_collate, device, cfg)
                    print(f"[eval] step={global_step} loss={eval_loss:.4f}")
                except Exception as e:
                    print(f"[warn] eval loss failed: {e}")

            # save checkpoint
            if cfg.save_steps and global_step > 0 and (global_step % cfg.save_steps == 0):
                state = {"epoch": epoch, "global_step": global_step, "epoch_batch": step + 1}
                save_checkpoint(model, tokenizer, optimizer, ckpt_root, global_step, state, cfg.save_total_limit)

            if cfg.max_steps > 0 and global_step >= cfg.max_steps:
                break

        if cfg.max_steps > 0 and global_step >= cfg.max_steps:
            break

    # final save to output_dir
    os.makedirs(cfg.output_dir, exist_ok=True)
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    elapsed = time.time() - start_time
    print(f"done. steps={global_step}, elapsed={elapsed:.1f}s, saved to {cfg.output_dir}")


if __name__ == "__main__":
    main()
