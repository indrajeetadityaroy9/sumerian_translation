"""
Modern loss functions for low-resource NMT.

Implements:
- Contrastive Loss (InfoNCE): Force encoder-decoder alignment
- Unlikelihood Loss: Penalize repetitive token generation
- R-Drop: Consistency regularization via KL divergence
"""

import torch
import torch.nn.functional as F
from typing import Optional


def contrastive_loss(
    encoder_output: torch.Tensor,
    decoder_hidden: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    temperature: float = 0.1
) -> torch.Tensor:
    """
    InfoNCE contrastive loss between encoder and decoder representations.

    Forces the decoder to attend to encoder outputs rather than ignoring them.

    Args:
        encoder_output: Encoder hidden states [B, S, D]
        decoder_hidden: Decoder hidden states [B, T, D]
        attention_mask: Optional mask for encoder [B, S]
        temperature: Softmax temperature (lower = sharper)

    Returns:
        Scalar contrastive loss
    """
    # Pool encoder output (mean over sequence, respecting mask)
    if attention_mask is not None:
        mask = attention_mask.unsqueeze(-1).float()
        enc_pooled = (encoder_output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    else:
        enc_pooled = encoder_output.mean(dim=1)

    # Pool decoder output (mean over sequence)
    dec_pooled = decoder_hidden.mean(dim=1)

    # L2 normalize
    enc_pooled = F.normalize(enc_pooled, p=2, dim=-1)
    dec_pooled = F.normalize(dec_pooled, p=2, dim=-1)

    # Cosine similarity matrix [B, B]
    logits = torch.matmul(enc_pooled, dec_pooled.T) / temperature

    # Diagonal entries are positive pairs
    labels = torch.arange(logits.size(0), device=logits.device)

    return F.cross_entropy(logits, labels)


def unlikelihood_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_id: int = -100
) -> torch.Tensor:
    """
    Unlikelihood loss to penalize repetitive token generation.

    When the model generates the same token as the previous position,
    we minimize log(1 - p(token)) to discourage this behavior.

    Reference: Welleck et al. "Neural Text Generation with Unlikelihood Training" (ICLR 2020)

    Args:
        logits: Model logits [B, T, V]
        targets: Target token IDs [B, T]
        pad_id: Padding token ID to ignore

    Returns:
        Scalar unlikelihood loss
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Shift targets to get previous tokens
    prev_tokens = torch.cat([
        torch.full((batch_size, 1), pad_id, device=targets.device, dtype=targets.dtype),
        targets[:, :-1]
    ], dim=1)

    # Identify repeated tokens (same as previous, excluding padding)
    repeated = (targets == prev_tokens) & (targets != pad_id) & (prev_tokens != pad_id)

    if not repeated.any():
        return torch.tensor(0.0, device=logits.device)

    # Get probabilities
    probs = F.softmax(logits, dim=-1)

    # Gather probabilities of target tokens
    # Clamp targets to valid range for gather
    gather_targets = targets.clamp(min=0, max=vocab_size - 1)
    token_probs = probs.gather(-1, gather_targets.unsqueeze(-1)).squeeze(-1)

    # Unlikelihood: -log(1 - p(repeated_token))
    # Add epsilon for numerical stability
    ul_loss = -torch.log(1 - token_probs + 1e-8)

    # Apply only to repeated tokens
    ul_loss = (ul_loss * repeated.float()).sum() / repeated.float().sum().clamp(min=1)

    return ul_loss


def token_frequency_penalty(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_id: int = -100,
    alpha: float = 1.0
) -> torch.Tensor:
    """
    Penalize over-frequent tokens within each sequence.

    Encourages lexical diversity by penalizing tokens that appear
    more than once in the target sequence.

    Args:
        logits: Model logits [B, T, V]
        targets: Target token IDs [B, T]
        pad_id: Padding token ID
        alpha: Penalty strength

    Returns:
        Scalar frequency penalty loss
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Count token frequencies per sequence
    # Create one-hot and sum
    valid_mask = (targets != pad_id).float()

    total_loss = torch.tensor(0.0, device=logits.device)

    for b in range(batch_size):
        seq_targets = targets[b][targets[b] != pad_id]
        if len(seq_targets) == 0:
            continue

        # Count occurrences
        unique, counts = torch.unique(seq_targets, return_counts=True)

        # Penalize tokens appearing more than once
        frequent_mask = counts > 1
        if not frequent_mask.any():
            continue

        frequent_tokens = unique[frequent_mask]

        # Get positions of frequent tokens
        for tok in frequent_tokens:
            positions = (targets[b] == tok).nonzero(as_tuple=True)[0]
            if len(positions) > 1:
                # Penalize all but first occurrence
                for pos in positions[1:]:
                    if pos < seq_len:
                        prob = F.softmax(logits[b, pos], dim=-1)[tok]
                        total_loss = total_loss - alpha * torch.log(1 - prob + 1e-8)

    return total_loss / batch_size


def rdrop_loss(
    logits1: torch.Tensor,
    logits2: torch.Tensor,
    pad_mask: Optional[torch.Tensor] = None,
    alpha: float = 0.5
) -> torch.Tensor:
    """
    R-Drop consistency loss via bidirectional KL divergence.

    Runs same input twice with different dropout, minimizes KL
    divergence between the two output distributions.

    Reference: Wu et al. "R-Drop: Regularized Dropout for Neural Networks" (NeurIPS 2021)

    Args:
        logits1: First forward pass logits [B, T, V]
        logits2: Second forward pass logits [B, T, V]
        pad_mask: Optional mask for padding positions [B, T]
        alpha: Weight for the loss

    Returns:
        Scalar R-Drop loss
    """
    # Convert to log probabilities
    log_p = F.log_softmax(logits1, dim=-1)
    log_q = F.log_softmax(logits2, dim=-1)

    p = log_p.exp()
    q = log_q.exp()

    # Bidirectional KL divergence
    kl_pq = F.kl_div(log_q, p, reduction='none').sum(dim=-1)
    kl_qp = F.kl_div(log_p, q, reduction='none').sum(dim=-1)

    kl_loss = (kl_pq + kl_qp) / 2

    # Apply padding mask if provided
    if pad_mask is not None:
        kl_loss = kl_loss * pad_mask.float()
        return alpha * kl_loss.sum() / pad_mask.float().sum().clamp(min=1)

    return alpha * kl_loss.mean()


def label_smoothing_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smoothing: float = 0.1,
    pad_id: int = -100
) -> torch.Tensor:
    """
    Cross-entropy with label smoothing.

    Args:
        logits: Model logits [B, T, V]
        targets: Target token IDs [B, T]
        smoothing: Smoothing factor (0 = no smoothing)
        pad_id: Padding token ID

    Returns:
        Scalar smoothed cross-entropy loss
    """
    vocab_size = logits.size(-1)

    # Create smoothed targets
    confidence = 1.0 - smoothing
    smooth_value = smoothing / (vocab_size - 1)

    # One-hot with smoothing
    one_hot = torch.zeros_like(logits).scatter_(
        -1, targets.clamp(min=0).unsqueeze(-1), confidence
    )
    one_hot = one_hot + smooth_value
    one_hot = one_hot * (targets != pad_id).unsqueeze(-1).float()

    # Renormalize
    one_hot = one_hot / one_hot.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    # Cross-entropy
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(one_hot * log_probs).sum(dim=-1)

    # Mask padding
    mask = (targets != pad_id).float()

    return (loss * mask).sum() / mask.sum().clamp(min=1)


class CombinedNMTLoss:
    """
    Combined loss for modern NMT training.

    Combines:
    - Standard cross-entropy (or label-smoothed)
    - Contrastive alignment loss
    - Unlikelihood loss for diversity
    - R-Drop consistency loss
    """

    def __init__(
        self,
        contrastive_weight: float = 0.1,
        unlikelihood_weight: float = 0.5,
        rdrop_weight: float = 0.1,
        label_smoothing: float = 0.1,
        temperature: float = 0.1
    ):
        self.contrastive_weight = contrastive_weight
        self.unlikelihood_weight = unlikelihood_weight
        self.rdrop_weight = rdrop_weight
        self.label_smoothing = label_smoothing
        self.temperature = temperature

    def __call__(
        self,
        outputs1,
        outputs2,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Compute combined loss.

        Args:
            outputs1: First forward pass model outputs
            outputs2: Second forward pass model outputs (for R-Drop)
            labels: Target token IDs [B, T]
            attention_mask: Encoder attention mask [B, S]

        Returns:
            Dictionary with total loss and component losses
        """
        logits1 = outputs1.logits
        logits2 = outputs2.logits

        # Primary NMT loss with label smoothing
        nmt_loss = label_smoothing_loss(
            logits1, labels,
            smoothing=self.label_smoothing
        )

        # Contrastive alignment loss
        contra_loss = torch.tensor(0.0, device=logits1.device)
        if self.contrastive_weight > 0 and hasattr(outputs1, 'encoder_last_hidden_state'):
            encoder_hidden = outputs1.encoder_last_hidden_state
            # Use last decoder hidden state
            if hasattr(outputs1, 'decoder_hidden_states') and outputs1.decoder_hidden_states:
                decoder_hidden = outputs1.decoder_hidden_states[-1]
            else:
                # Fallback: use logits projected back
                decoder_hidden = logits1

            contra_loss = contrastive_loss(
                encoder_hidden, decoder_hidden,
                attention_mask=attention_mask,
                temperature=self.temperature
            )

        # Unlikelihood loss for diversity
        ul_loss = torch.tensor(0.0, device=logits1.device)
        if self.unlikelihood_weight > 0:
            ul_loss = unlikelihood_loss(logits1, labels)

        # R-Drop consistency loss
        rdrop = torch.tensor(0.0, device=logits1.device)
        if self.rdrop_weight > 0:
            pad_mask = (labels != -100)
            rdrop = rdrop_loss(logits1, logits2, pad_mask=pad_mask)

        # Combined loss
        total_loss = (
            nmt_loss
            + self.contrastive_weight * contra_loss
            + self.unlikelihood_weight * ul_loss
            + self.rdrop_weight * rdrop
        )

        return {
            'loss': total_loss,
            'nmt_loss': nmt_loss.item(),
            'contrastive_loss': contra_loss.item() if torch.is_tensor(contra_loss) else contra_loss,
            'unlikelihood_loss': ul_loss.item() if torch.is_tensor(ul_loss) else ul_loss,
            'rdrop_loss': rdrop.item() if torch.is_tensor(rdrop) else rdrop
        }
