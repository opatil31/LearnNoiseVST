"""
Leave-One-Out (LOO) pooling for tabular blind-spot denoising.

The key insight is that for tabular data with d features, we can achieve
blind-spot denoising by:
1. Embedding each feature independently: e_f = φ(z_f)
2. Pooling all embeddings: S = sum(e_f)
3. Computing leave-one-out context: c_f = (S - e_f) / (d - 1)
4. Predicting each feature from its LOO context: ẑ_f = ψ(c_f)

This ensures ẑ_f is independent of z_f (only depends on other features).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LeaveOneOutPooling(nn.Module):
    """
    Efficient leave-one-out mean pooling.

    Given embeddings e ∈ [B, d, H], computes:
        S = sum(e, dim=1)  # [B, H]
        c_f = (S - e_f) / (n - 1)  # [B, d, H]

    where n is the count of valid features (for handling missingness).

    This is O(d) instead of O(d²) naive implementation.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        e: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute leave-one-out pooled context for each feature.

        Args:
            e: Embeddings [B, d, H] where d is num_features, H is embed_dim.
            mask: Optional binary mask [B, d] where 1 = observed, 0 = missing.

        Returns:
            c: Leave-one-out context [B, d, H].
        """
        B, d, H = e.shape

        if mask is None:
            # All features observed
            mask = torch.ones(B, d, device=e.device, dtype=e.dtype)

        # Expand mask for broadcasting: [B, d, 1]
        mask_expanded = mask.unsqueeze(-1)

        # Masked embeddings
        e_masked = e * mask_expanded

        # Total sum: [B, H]
        S = e_masked.sum(dim=1)

        # Count of observed features: [B, 1]
        n = mask.sum(dim=1, keepdim=True)

        # Leave-one-out sum: [B, d, H]
        S_minus_f = S.unsqueeze(1) - e_masked

        # Leave-one-out count: [B, d, 1]
        n_minus_f = n.unsqueeze(-1) - mask_expanded

        # Leave-one-out mean
        c = S_minus_f / (n_minus_f.clamp(min=1) + self.eps)

        return c


class LeaveOneOutSumPooling(nn.Module):
    """
    Leave-one-out sum pooling (without normalization).

    Returns c_f = S - e_f (the sum of all other embeddings).
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        e: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute leave-one-out sum for each feature.

        Args:
            e: Embeddings [B, d, H].
            mask: Optional binary mask [B, d].

        Returns:
            c: Leave-one-out sum [B, d, H].
        """
        if mask is None:
            S = e.sum(dim=1, keepdim=True)  # [B, 1, H]
            return S - e
        else:
            mask_expanded = mask.unsqueeze(-1)
            e_masked = e * mask_expanded
            S = e_masked.sum(dim=1, keepdim=True)
            return S - e_masked


class LeaveOneOutAttentionPooling(nn.Module):
    """
    Attention-based leave-one-out pooling.

    Instead of simple mean, uses attention to weight the contribution
    of other features to each feature's context.

    For feature f, attends over all other features {g : g ≠ f}.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.scale = (embed_dim // num_heads) ** -0.5

    def forward(
        self,
        e: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Attention-based LOO pooling.

        Args:
            e: Embeddings [B, d, H].
            mask: Optional mask [B, d].

        Returns:
            c: Context vectors [B, d, H].
        """
        B, d, H = e.shape
        heads = self.num_heads
        head_dim = H // heads

        # Project to Q, K, V
        Q = self.q_proj(e)  # [B, d, H]
        K = self.k_proj(e)
        V = self.v_proj(e)

        # Reshape for multi-head attention: [B, heads, d, head_dim]
        Q = Q.view(B, d, heads, head_dim).transpose(1, 2)
        K = K.view(B, d, heads, head_dim).transpose(1, 2)
        V = V.view(B, d, heads, head_dim).transpose(1, 2)

        # Attention scores: [B, heads, d, d]
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Mask out self-attention (diagonal) to enforce LOO property
        # Set diagonal to -inf so softmax gives 0 weight
        diag_mask = torch.eye(d, device=e.device, dtype=torch.bool)
        attn = attn.masked_fill(diag_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Apply missingness mask if provided
        if mask is not None:
            # mask: [B, d] -> [B, 1, 1, d] for key masking
            key_mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(~key_mask.bool(), float('-inf'))

        # Softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, V)  # [B, heads, d, head_dim]

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, d, H)
        out = self.out_proj(out)

        return out


class ChunkedLeaveOneOutPooling(nn.Module):
    """
    Memory-efficient LOO pooling for large number of features.

    For very large d (e.g., 100K features), materializing the full
    [B, d, H] embedding tensor can be expensive. This implementation
    uses two passes:
        1. Accumulate sum S without storing all embeddings
        2. Recompute embeddings chunk-wise and compute LOO context

    This trades compute for memory.
    """

    def __init__(self, encoder: nn.Module, chunk_size: int = 1000, eps: float = 1e-8):
        """
        Args:
            encoder: Module that encodes features, called as encoder(z_chunk, feature_ids).
            chunk_size: Number of features to process at once.
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.encoder = encoder
        self.chunk_size = chunk_size
        self.eps = eps

    def forward(
        self,
        z: torch.Tensor,
        feature_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Two-pass LOO computation.

        Args:
            z: Input features [B, d].
            feature_ids: Feature indices [d].
            mask: Optional mask [B, d].

        Returns:
            c: LOO context [B, d, H].
            S: Total sum [B, H] (for potential reuse).
        """
        B, d = z.shape

        if mask is None:
            mask = torch.ones(B, d, device=z.device)

        # Pass 1: Accumulate sum
        S = None
        n = mask.sum(dim=1, keepdim=True)  # [B, 1]

        for start in range(0, d, self.chunk_size):
            end = min(start + self.chunk_size, d)

            z_chunk = z[:, start:end]
            ids_chunk = feature_ids[start:end]
            mask_chunk = mask[:, start:end]

            # Encode chunk
            e_chunk = self.encoder(z_chunk, ids_chunk)  # [B, chunk, H]
            e_chunk = e_chunk * mask_chunk.unsqueeze(-1)

            # Accumulate
            chunk_sum = e_chunk.sum(dim=1)  # [B, H]
            S = chunk_sum if S is None else S + chunk_sum

        # Pass 2: Compute LOO context
        H = S.shape[-1]
        c = torch.zeros(B, d, H, device=z.device)

        for start in range(0, d, self.chunk_size):
            end = min(start + self.chunk_size, d)

            z_chunk = z[:, start:end]
            ids_chunk = feature_ids[start:end]
            mask_chunk = mask[:, start:end]

            # Re-encode chunk
            e_chunk = self.encoder(z_chunk, ids_chunk)
            e_chunk_masked = e_chunk * mask_chunk.unsqueeze(-1)

            # LOO for this chunk
            S_minus_chunk = S.unsqueeze(1) - e_chunk_masked  # [B, chunk, H]
            n_minus_chunk = n.unsqueeze(-1) - mask_chunk.unsqueeze(-1)  # [B, chunk, 1]

            c_chunk = S_minus_chunk / (n_minus_chunk.clamp(min=1) + self.eps)
            c[:, start:end] = c_chunk

        return c, S
