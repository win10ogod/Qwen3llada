from typing import Optional, Union

import torch
from torch import nn

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers import AutoModel, PreTrainedModel
from .configuration_qwen3llada import Qwen3LLadaConfig
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache


class Qwen3LLadaPreTrainedModel(PreTrainedModel):
    base_model_prefix = "backbone"
    config_class = Qwen3LLadaConfig


class Qwen3LLadaModel(Qwen3LLadaPreTrainedModel):
    """A Qwen3 backbone run in non-causal (encoder-like) mode.

    It reuses Qwen3Model layers but disables causal/sliding masks by passing None to attention_mask.
    """

    def __init__(self, config: Union[Qwen3LLadaConfig, Qwen3Config]):
        # Allow using a plain Qwen3Config for weight loading convenience
        if not isinstance(config, Qwen3LLadaConfig):
            config = Qwen3LLadaConfig(**config.__dict__)
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Reuse the exact Qwen3 stack
        # Build a Qwen3 backbone from the standard transformers registry lazily
        base_dict = {k: v for k, v in config.to_dict().items() if k not in ("model_type", "mask_token_id")}
        base_dict["model_type"] = "qwen3"
        base_cfg = Qwen3Config(**base_dict)
        self.backbone = AutoModel.from_config(base_cfg)
        self.ln_f = nn.Identity()  # already inside Qwen3Model as final norm; keep identity for clarity

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,  # optional additive bias for advanced masks
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        # Optionally accept an additive attention bias (float mask) and pass it through the mapping expected by Qwen3Model.
        # By default, enforce non-causal by using None (full bidirectional attention).
        attention_mapping = {"full_attention": None}
        if getattr(self.backbone, "has_sliding_layers", False):
            attention_mapping["sliding_attention"] = None

        if attention_mask is not None:
            # Support boolean or additive float masks; align to model expectations by passing directly.
            attention_mapping["full_attention"] = attention_mask
            if getattr(self.backbone, "has_sliding_layers", False):
                attention_mapping["sliding_attention"] = attention_mask

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mapping,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        return outputs


class Qwen3LLadaForMaskedLM(Qwen3LLadaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Union[Qwen3LLadaConfig, Qwen3Config]):
        if not isinstance(config, Qwen3LLadaConfig):
            config = Qwen3LLadaConfig(**config.__dict__)
        super().__init__(config)
        self.backbone = Qwen3LLadaModel(config)
        # Ensure non-causal attention backends do not inject causal mask via kernels
        try:
            if hasattr(self.backbone.backbone.config, "_attn_implementation"):
                self.backbone.backbone.config._attn_implementation = "eager"
        except Exception:
            pass
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,  # ignored; compute external diffusion loss
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.backbone(
            input_ids=input_ids,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "Qwen3LLadaPreTrainedModel",
    "Qwen3LLadaModel",
    "Qwen3LLadaForMaskedLM",
]
