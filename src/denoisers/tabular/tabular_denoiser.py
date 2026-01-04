"""
Tabular blind-spot denoiser using leave-one-out pooling.

Architecture (DeepSets-style with feature conditioning):
    1. Per-feature embedding: e_f = FiLM(MLP_val(z_f), u_f)
       - MLP_val: Shared scalar value encoder
       - u_f: Learned feature ID embedding
       - FiLM: Feature-wise Linear Modulation

    2. Leave-one-out pooling: c_f = LOO_mean({e_g : g ≠ f})

    3. Prediction head: ẑ_f = <v_f, MLP_ctx(c_f)> + b_f
       - v_f: Feature-specific projection vector
       - MLP_ctx: Shared context decoder

This achieves blind-spot property because ẑ_f only depends on {z_g : g ≠ f}.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .loo_pooling import LeaveOneOutPooling, LeaveOneOutAttentionPooling
from ..base import TabularBlindSpotDenoiser


class FiLMModulation(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM).

    Applies affine transformation conditioned on feature embedding:
        output = gamma(u) * input + beta(u)

    This allows the shared encoder to behave differently for different features.
    """

    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.gamma = nn.Linear(feature_dim, hidden_dim)
        self.beta = nn.Linear(feature_dim, hidden_dim)

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM modulation.

        Args:
            x: Input tensor [B, d, H] or [B, H].
            u: Feature embeddings [d, H] or [H].

        Returns:
            Modulated output, same shape as x.
        """
        gamma = self.gamma(u)  # [d, H] or [H]
        beta = self.beta(u)

        if x.dim() == 3 and u.dim() == 2:
            # x: [B, d, H], u: [d, H]
            gamma = gamma.unsqueeze(0)  # [1, d, H]
            beta = beta.unsqueeze(0)

        return gamma * x + beta


class TabularBlindSpotDenoiser(TabularBlindSpotDenoiser):
    """
    Full tabular blind-spot denoiser.

    Uses the architecture from the implementation plan:
    - Shared value encoder with per-feature FiLM conditioning
    - Leave-one-out mean pooling
    - Shared context decoder with per-feature output projection

    Args:
        num_features: Number of input features (d).
        embed_dim: Embedding dimension (H).
        hidden_dim: Hidden layer dimension for MLPs.
        num_encoder_layers: Number of layers in value encoder.
        num_decoder_layers: Number of layers in context decoder.
        dropout: Dropout rate.
        use_layer_norm: Whether to use LayerNorm.
        use_attention_pooling: Whether to use attention-based LOO pooling.
    """

    def __init__(
        self,
        num_features: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        use_attention_pooling: bool = False,
        attention_heads: int = 4,
    ):
        super().__init__(num_features)

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Feature ID embeddings
        # u_f for encoder (to differentiate how features are embedded)
        self.feature_embed_enc = nn.Embedding(num_features, embed_dim)

        # v_f for decoder (to differentiate how predictions are made)
        self.feature_embed_dec = nn.Embedding(num_features, embed_dim)

        # Per-feature bias
        self.output_bias = nn.Parameter(torch.zeros(num_features))

        # Shared value encoder MLP
        # Input: (z_f, mask_f) -> embed_dim
        encoder_layers = []
        in_dim = 2  # (value, mask indicator)

        for i in range(num_encoder_layers):
            out_dim = hidden_dim if i < num_encoder_layers - 1 else embed_dim
            encoder_layers.append(nn.Linear(in_dim, out_dim))
            if i < num_encoder_layers - 1:
                if use_layer_norm:
                    encoder_layers.append(nn.LayerNorm(out_dim))
                encoder_layers.append(nn.ReLU())
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        self.mlp_val = nn.Sequential(*encoder_layers)

        # FiLM modulation
        self.film = FiLMModulation(embed_dim, embed_dim)

        # LayerNorm after embedding (stabilizes training with large d)
        self.embed_norm = nn.LayerNorm(embed_dim) if use_layer_norm else nn.Identity()

        # Leave-one-out pooling
        if use_attention_pooling:
            self.loo_pool = LeaveOneOutAttentionPooling(
                embed_dim, num_heads=attention_heads, dropout=dropout
            )
        else:
            self.loo_pool = LeaveOneOutPooling()

        # Shared context decoder MLP
        # Input: c_f (embed_dim) -> embed_dim (for dot product with v_f)
        decoder_layers = []
        in_dim = embed_dim

        for i in range(num_decoder_layers):
            out_dim = hidden_dim if i < num_decoder_layers - 1 else embed_dim
            decoder_layers.append(nn.Linear(in_dim, out_dim))
            if i < num_decoder_layers - 1:
                if use_layer_norm:
                    decoder_layers.append(nn.LayerNorm(out_dim))
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        self.mlp_ctx = nn.Sequential(*decoder_layers)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        # Feature embeddings: small random init
        nn.init.normal_(self.feature_embed_enc.weight, std=0.02)
        nn.init.normal_(self.feature_embed_dec.weight, std=0.02)

        # Output bias: zero init
        nn.init.zeros_(self.output_bias)

    def forward(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Denoise tabular data with blind-spot property.

        Args:
            z: Input tensor [B, d] where d = num_features.
            mask: Optional binary mask [B, d] (1 = observed, 0 = missing).

        Returns:
            z_hat: Denoised output [B, d] where ẑ_f is independent of z_f.
        """
        B, d = z.shape
        assert d == self.num_features, f"Expected {self.num_features} features, got {d}"

        if mask is None:
            mask = torch.ones(B, d, device=z.device, dtype=z.dtype)

        # Feature indices: [d]
        feature_ids = torch.arange(d, device=z.device)

        # --- Step 1: Per-feature embedding ---

        # Prepare input: [B, d, 2] = (value, mask)
        z_input = torch.stack([z, mask], dim=-1)  # [B, d, 2]

        # Reshape for shared MLP: [B*d, 2] -> [B*d, H]
        z_flat = z_input.view(B * d, 2)
        v_encoded = self.mlp_val(z_flat)  # [B*d, H]
        v_encoded = v_encoded.view(B, d, self.embed_dim)  # [B, d, H]

        # Get feature embeddings for conditioning
        u = self.feature_embed_enc(feature_ids)  # [d, H]

        # Apply FiLM modulation
        e = self.film(v_encoded, u)  # [B, d, H]

        # LayerNorm
        e = self.embed_norm(e)

        # --- Step 2: Leave-one-out pooling ---

        # c_f depends only on {e_g : g ≠ f}, hence only on {z_g : g ≠ f}
        c = self.loo_pool(e, mask)  # [B, d, H]

        # --- Step 3: Context decoding and prediction ---

        # Decode context
        c_flat = c.view(B * d, self.embed_dim)
        h = self.mlp_ctx(c_flat)  # [B*d, H]
        h = h.view(B, d, self.embed_dim)  # [B, d, H]

        # Get feature-specific projections
        v = self.feature_embed_dec(feature_ids)  # [d, H]

        # Dot product prediction: ẑ_f = <v_f, h_f> + b_f
        # h: [B, d, H], v: [d, H] -> need to broadcast
        z_hat = (h * v.unsqueeze(0)).sum(dim=-1)  # [B, d]
        z_hat = z_hat + self.output_bias

        return z_hat


class LightweightTabularDenoiser(TabularBlindSpotDenoiser):
    """
    Lightweight tabular denoiser for smaller datasets or quick experiments.

    Simpler architecture with fewer parameters.
    """

    def __init__(
        self,
        num_features: int,
        embed_dim: int = 32,
        hidden_dim: int = 64,
    ):
        # Don't call parent __init__ with wrong signature
        nn.Module.__init__(self)
        self.num_features = num_features
        self.embed_dim = embed_dim

        # Feature embeddings
        self.feature_embed = nn.Embedding(num_features, embed_dim)

        # Value encoder (simpler)
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # LOO pooling
        self.loo_pool = LeaveOneOutPooling()

        # Decoder (simpler)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, d = z.shape

        if mask is None:
            mask = torch.ones(B, d, device=z.device, dtype=z.dtype)

        # Encode
        z_input = torch.stack([z, mask], dim=-1)  # [B, d, 2]
        e = self.encoder(z_input)  # [B, d, H]

        # Add feature embeddings
        feature_ids = torch.arange(d, device=z.device)
        u = self.feature_embed(feature_ids)  # [d, H]
        e = e + u.unsqueeze(0)  # [B, d, H]

        # LOO pooling
        c = self.loo_pool(e, mask)  # [B, d, H]

        # Decode
        z_hat = self.decoder(c).squeeze(-1)  # [B, d]

        return z_hat


class DeepTabularDenoiser(TabularBlindSpotDenoiser):
    """
    Deeper tabular denoiser with expressive per-position processing.

    Architecture ensures blind-spot property by:
    1. Embedding all features
    2. Applying a SINGLE LOO layer to get blind-spot context (c_f depends on {z_g : g ≠ f})
    3. Processing with per-position FFN layers (no cross-position mixing)

    Note: Stacking multiple LOO layers breaks blind-spot because the second layer's
    output at position f depends on position g's output (g≠f), which depends on z_f.
    """

    def __init__(
        self,
        num_features: int,
        embed_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        nn.Module.__init__(self)
        self.num_features = num_features
        self.embed_dim = embed_dim

        # Input embedding
        self.input_embed = nn.Linear(2, embed_dim)  # (value, mask)
        self.feature_embed = nn.Embedding(num_features, embed_dim)

        # Single LOO attention layer (to maintain blind-spot property)
        self.loo_attn = LeaveOneOutAttentionPooling(embed_dim, num_heads, dropout)
        self.loo_norm = nn.LayerNorm(embed_dim)

        # Per-position FFN layers (no cross-position mixing, so blind-spot is preserved)
        self.ffn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.ffn_layers.append(nn.ModuleDict({
                'ffn': nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, embed_dim),
                    nn.Dropout(dropout),
                ),
                'norm': nn.LayerNorm(embed_dim),
            }))

        # Output head
        self.output_head = nn.Linear(embed_dim, 1)

    def forward(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, d = z.shape

        if mask is None:
            mask = torch.ones(B, d, device=z.device, dtype=z.dtype)

        # Initial embedding
        z_input = torch.stack([z, mask], dim=-1)
        x = self.input_embed(z_input)  # [B, d, H]

        # Add feature embeddings
        feature_ids = torch.arange(d, device=z.device)
        x = x + self.feature_embed(feature_ids).unsqueeze(0)

        # Single LOO attention to get blind-spot context
        # c_f depends only on {x_g : g ≠ f}, hence only on {z_g : g ≠ f}
        c = self.loo_attn(self.loo_norm(x), mask)  # [B, d, H]

        # Process through per-position FFN layers (with residual)
        # Since FFN is per-position, it doesn't mix positions, so blind-spot is preserved
        for layer in self.ffn_layers:
            ffn_out = layer['ffn'](layer['norm'](c))
            c = c + ffn_out

        # Output
        z_hat = self.output_head(c).squeeze(-1)  # [B, d]

        return z_hat
