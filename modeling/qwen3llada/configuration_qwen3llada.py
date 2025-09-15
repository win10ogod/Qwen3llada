from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


class Qwen3LLadaConfig(Qwen3Config):
    """Configuration for Qwen3LLada.

    Extends Qwen3Config with diffusion-style masked modeling options while preserving
    compatibility with Qwen3 weights and shapes.
    """

    model_type = "qwen3llada"

    def __init__(self, mask_token_id: int = 126336, use_cache: bool = False, **kwargs):
        # Default diffusion models disable KV cache
        kwargs.setdefault("use_cache", use_cache)
        super().__init__(**kwargs)
        self.mask_token_id = mask_token_id


__all__ = ["Qwen3LLadaConfig"]
