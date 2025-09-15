import torch

from modeling.qwen3llada import Qwen3LLadaConfig, Qwen3LLadaForMaskedLM


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

