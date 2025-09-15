import argparse
import os
import sys
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import os as _os
_os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
_os.environ.setdefault("HF_TRANSFORMERS_NO_TF", "1")
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from transformers import AutoTokenizer

# Ensure project root on sys.path for `modeling` imports when running as a script
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from modeling.qwen3llada import Qwen3LLadaForMaskedLM


def random_topk_mask_indices(confidence: torch.Tensor, k: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Select k positions with lowest confidence but with randomization among near-ties.
    Returns a boolean mask of selected indices.
    """
    B, L = confidence.shape
    sel = torch.zeros_like(confidence, dtype=torch.bool)
    # perturb confidence with small random noise to break ties
    noise = torch.randn_like(confidence) * 1e-6
    perturbed = confidence + noise
    # lower confidence should be selected first
    vals, idx = torch.topk(-perturbed, k=k, dim=-1)
    for b in range(B):
        sel[b, idx[b]] = True
    return sel


def add_gumbel_noise(logits: torch.Tensor, temperature: float):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


def build_block_attention_bias(batch_size: int, seq_len: int, prompt_len: int, block_length: int, device: torch.device) -> torch.Tensor:
    """Build additive attention mask [B,1,Q,K]: allow prompt fully visible;
    for answer region, only allow attention within its own block (and prompt).
    """
    bias = torch.zeros((batch_size, 1, seq_len, seq_len), dtype=torch.float32, device=device)
    if block_length is None or block_length <= 0:
        return bias
    finfo_min = torch.finfo(bias.dtype).min
    for b in range(batch_size):
        p = prompt_len
        for q in range(seq_len):
            if q < p:
                continue
            start = p + ((q - p) // block_length) * block_length
            end = min(start + block_length, seq_len)
            if start > p:
                bias[b, 0, q, p:start] = finfo_min
            if end < seq_len:
                bias[b, 0, q, end:seq_len] = finfo_min
    return bias


@torch.no_grad()
def generate(
    model: Qwen3LLadaForMaskedLM,
    prompt: torch.LongTensor,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    block_attention: bool = False,
) -> torch.LongTensor:
    x = torch.full((prompt.size(0), prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=prompt.device)
    x[:, : prompt.shape[1]] = prompt.clone()

    prompt_index = x != mask_id
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_slice = slice(prompt.shape[1] + num_block * block_length, prompt.shape[1] + (num_block + 1) * block_length)
        block_mask_index = x[:, block_slice] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        # precompute attention bias for this block if enabled
        attn_bias = None
        if block_attention:
            attn_bias = build_block_attention_bias(
                batch_size=x.size(0),
                seq_len=x.size(1),
                prompt_len=prompt.shape[1],
                block_length=block_length,
                device=x.device,
            )

        for i in range(steps_per_block):
            mask_index = x == mask_id
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attn_bias is not None:
                    ab = torch.cat([attn_bias, attn_bias], dim=0)
                else:
                    ab = None
                logits = model(x_, attention_mask=ab).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attn_bias).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == "random_topk":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            # disallow transitions outside current block
            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length :] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                k = int(num_transfer_tokens[j, i].item())
                if remasking == "random_topk":
                    sel = random_topk_mask_indices(-confidence[j].unsqueeze(0), k)[0]
                    transfer_index[j] = sel
                else:
                    _, select_index = torch.topk(confidence[j], k=k)
                    transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to qwen3llada model")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg", type=float, default=0.0)
    parser.add_argument("--remasking", type=str, default="low_confidence")
    parser.add_argument("--block_attention", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Qwen3LLadaForMaskedLM.from_pretrained(args.model).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Prepare prompt ids
    ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
    out = generate(
        model,
        ids,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
        cfg_scale=args.cfg,
        remasking=args.remasking,
        mask_id=getattr(model.config, "mask_token_id", 126336),
        block_attention=args.block_attention,
    )
    print(tokenizer.batch_decode(out[:, ids.shape[1] :], skip_special_tokens=True)[0])


if __name__ == "__main__":
    main()
