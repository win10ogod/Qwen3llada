import torch
import torch.nn.functional as F


def forward_process(input_ids: torch.LongTensor, mask_token_id: int, eps: float = 1e-3):
    """Apply LLaDA forward noising to a batch of token ids.

    Args:
        input_ids: Long tensor of shape (b, l)
        mask_token_id: Reserved token id used for [MASK] (126336 in LLaDA)
        eps: Small epsilon to avoid p=0 masking probability
    Returns:
        noisy_batch: input with a subset of tokens replaced by mask token
        masked_indices: bool tensor of shape (b, l) indicating masked positions
        p_mask: per-token masking probability used for reweighting
    """
    b, l = input_ids.shape
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)

    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    # ensure at least one masked token per sample for stable gradients
    if masked_indices.dim() == 2:
        ensure = ~masked_indices.any(dim=1)
        if ensure.any():
            rand_pos = torch.randint(low=0, high=l, size=(int(ensure.sum().item()),), device=input_ids.device)
            masked_indices[ensure, :] = False
            masked_indices[ensure, rand_pos] = True
    noisy_batch = torch.where(masked_indices, torch.as_tensor(mask_token_id, device=input_ids.device), input_ids)
    return noisy_batch, masked_indices, p_mask


def diffusion_ce_loss(logits: torch.Tensor, input_ids: torch.LongTensor, masked_indices: torch.Tensor, p_mask: torch.Tensor):
    """Token-level cross entropy reweighted by 1/p_mask and normalized by sequence length, per LLaDA guidelines.

    Compute CE in float32 for numerical stability (especially with bf16/fp16 training).
    """
    logits_f32 = logits.float()
    token_loss = F.cross_entropy(logits_f32[masked_indices], input_ids[masked_indices], reduction="none") / p_mask[masked_indices]
    # Normalize by batch*seq_len as in guidelines
    return token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])


def sft_forward(input_ids: torch.LongTensor, prompt_lengths: torch.LongTensor, mask_token_id: int, eps: float = 1e-3):
    """SFT variant: do not add noise to prompt tokens; reweight by answer length.

    Returns noisy_batch, masked_indices, p_mask, answer_lengths_mask
    """
    noisy_batch, _, p_mask = forward_process(input_ids, mask_token_id, eps)
    token_positions = torch.arange(noisy_batch.shape[1], device=noisy_batch.device).expand(noisy_batch.size(0), noisy_batch.size(1))
    prompt_mask = (token_positions < prompt_lengths.unsqueeze(1))
    noisy_batch[prompt_mask] = input_ids[prompt_mask]
    masked_indices = (noisy_batch == mask_token_id)

    prompt_mask_i64 = prompt_mask.to(torch.int64)
    answer_lengths = torch.sum((1 - prompt_mask_i64), dim=-1, keepdim=True)
    answer_lengths = answer_lengths.repeat(1, noisy_batch.shape[1])
    return noisy_batch, masked_indices, p_mask, answer_lengths


def sft_ce_loss(logits: torch.Tensor, input_ids: torch.LongTensor, masked_indices: torch.Tensor, p_mask: torch.Tensor, answer_lengths: torch.Tensor):
    logits_f32 = logits.float()
    token_loss = F.cross_entropy(logits_f32[masked_indices], input_ids[masked_indices], reduction="none") / p_mask[masked_indices]
    ce_loss = torch.sum(token_loss / answer_lengths[masked_indices]) / input_ids.shape[0]
    return ce_loss


def soft_label_smoothing_loss(
    logits: torch.Tensor,
    input_ids: torch.LongTensor,
    masked_indices: torch.Tensor,
    p_mask: torch.Tensor,
    smoothing: float = 0.1,
) -> torch.Tensor:
    """Masked label-smoothing cross entropy, inspired by dLLM-RL soft-target CE.

    loss = (1 - eps) * CE(one_hot) + eps * mean(-log_probs) on masked positions, reweighted by 1/p_mask,
    and normalized by batch*seq_len.
    """
    logits_f32 = logits.float()
    vocab_size = logits_f32.shape[-1]
    log_probs = F.log_softmax(logits_f32, dim=-1)
    # NLL term on true labels
    nll = -log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
    # Smooth term: uniform over vocab
    smooth = -log_probs.mean(dim=-1)
    loss_tok = (1.0 - smoothing) * nll + smoothing * smooth
    loss_tok = (loss_tok[masked_indices] / p_mask[masked_indices])
    return loss_tok.sum() / (input_ids.shape[0] * input_ids.shape[1])


def dllm_llada_pretrain_loss(
    logits: torch.Tensor,
    labels: torch.LongTensor,
    masked_indices: torch.Tensor,
    p_mask: torch.Tensor,
) -> torch.Tensor:
    """Pretrain loss adapted from dLLM-RL-main (llada loss block).

    The caller provides the boolean mask of noised positions (e.g. from
    :func:`forward_process_cfg`), allowing noise types such as random replace
    where masked tokens are not necessarily equal to ``mask_token_id``. The
    cross-entropy is reweighted by ``1 / p_mask`` and normalized by
    ``batch * seq_len``.
    """
    logits_f32 = logits.float()
    token_loss = F.cross_entropy(logits_f32[masked_indices], labels[masked_indices], reduction='none') / p_mask[masked_indices]
    loss = token_loss.sum() / (labels.shape[0] * labels.shape[1])
    return loss
