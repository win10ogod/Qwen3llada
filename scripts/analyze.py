import argparse
import os
import sys
from typing import Dict, Tuple

import torch
import os as _os
_os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
_os.environ.setdefault("HF_TRANSFORMERS_NO_TF", "1")
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Ensure project root on sys.path for `modeling` imports when running as a script
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from transformers import AutoModelForCausalLM
from modeling.qwen3llada import Qwen3LLadaForMaskedLM, Qwen3LLadaConfig


def analyze(src: str):
    base = AutoModelForCausalLM.from_pretrained(src)
    llada_cfg = Qwen3LLadaConfig(**base.config.__dict__)
    llada = Qwen3LLadaForMaskedLM(llada_cfg)

    base_sd = base.state_dict()
    llada_sd = llada.state_dict()

    # map keys
    def map_key(k: str) -> str:
        if k.startswith("model."):
            return "backbone.backbone." + k[len("model."):]
        elif k.startswith("lm_head."):
            return "lm_head." + k[len("lm_head."):]
        return "__ignore__"

    matched, missing, shape_mismatch = 0, [], []
    for k, v in base_sd.items():
        mk = map_key(k)
        if mk == "__ignore__":
            continue
        if mk not in llada_sd:
            missing.append((k, mk))
        else:
            if llada_sd[mk].shape != v.shape:
                shape_mismatch.append((k, v.shape, llada_sd[mk].shape))
            else:
                matched += 1

    print(f"Matched tensors: {matched}")
    if missing:
        print(f"Missing mappings: {len(missing)} (showing first 10)")
        for i, (a, b) in enumerate(missing[:10]):
            print(f"  {a} -> {b}")
    if shape_mismatch:
        print(f"Shape mismatches: {len(shape_mismatch)} (showing first 10)")
        for i, (k, s1, s2) in enumerate(shape_mismatch[:10]):
            print(f"  {k}: {s1} vs {s2}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    args = ap.parse_args()
    analyze(args.src)


if __name__ == "__main__":
    main()
