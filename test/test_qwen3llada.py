import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from modeling.qwen3llada import Qwen3LLadaConfig, Qwen3LLadaForMaskedLM
from scripts.loss import dllm_llada_pretrain_loss
from scripts.train import forward_process_cfg


def test_forward_small_config():
    # Small config to avoid heavy memory
    cfg = Qwen3LLadaConfig(
        vocab_size=32000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=64,
        max_position_embeddings=256,
        use_cache=False,
        mask_token_id=126336,
    )
    model = Qwen3LLadaForMaskedLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(input_ids=x)
    assert out.logits.shape == (2, 16, cfg.vocab_size)


def test_dllm_loss_random_replace_positive():
    cfg = Qwen3LLadaConfig(
        vocab_size=32000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=64,
        max_position_embeddings=256,
        use_cache=False,
        mask_token_id=126336,
    )
    model = Qwen3LLadaForMaskedLM(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))

    class NoiseCfg:
        def __init__(self):
            self.noise = {"noise_type": "random_replace"}

    noise_cfg = NoiseCfg()
    noisy_batch, masked_indices, p_mask = forward_process_cfg(
        input_ids, cfg.mask_token_id, noise_cfg, cfg.vocab_size
    )
    logits = model(input_ids=noisy_batch).logits
    loss = dllm_llada_pretrain_loss(logits, input_ids, masked_indices, p_mask)

    assert torch.isfinite(loss)
    assert float(loss.item()) > 0.0

