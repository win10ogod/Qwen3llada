import argparse
import os
import sys
from typing import Dict

import torch
import os as _os
_os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
_os.environ.setdefault("HF_TRANSFORMERS_NO_TF", "1")
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from transformers import AutoTokenizer

# Ensure project root on sys.path for `modeling` imports when running as a script
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from transformers import AutoModelForCausalLM
from modeling.qwen3llada import Qwen3LLadaForMaskedLM, Qwen3LLadaConfig


def convert_qwen3_to_llada(src: str, dst: str, mask_token_id: int = 126336):
    os.makedirs(dst, exist_ok=True)
    base = AutoModelForCausalLM.from_pretrained(src)
    cfg = Qwen3LLadaConfig(**base.config.__dict__)
    cfg.mask_token_id = mask_token_id
    model = Qwen3LLadaForMaskedLM(cfg)

    sd = base.state_dict()
    new_sd: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith("model."):
            new_sd["backbone.backbone." + k[len("model."):]] = v
        elif k.startswith("lm_head."):
            new_sd["lm_head." + k[len("lm_head."):]] = v
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        print(f"[warn] missing keys: {missing[:8]} ... total={len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected[:8]} ... total={len(unexpected)}")

    model.save_pretrained(dst)
    # copy tokenizer
    tok = AutoTokenizer.from_pretrained(src, use_fast=True)
    tok.save_pretrained(dst)
    print(f"Converted and saved qwen3llada to: {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to Qwen3 checkpoint")
    ap.add_argument("--dst", required=True, help="Output path for qwen3llada checkpoint")
    ap.add_argument("--mask_token_id", type=int, default=126336)
    args = ap.parse_args()
    convert_qwen3_to_llada(args.src, args.dst, args.mask_token_id)


if __name__ == "__main__":
    main()
