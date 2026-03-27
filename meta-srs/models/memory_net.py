"""
Neural Memory Model — MemoryNet (Section 2.2).

Replaces FSRS-6's hand-crafted formulas with a small differentiable neural
network (~50K parameters) that predicts memory-state transitions.

The small footprint is intentional: fast-adaptation requires a network that
moves meaningfully in just 5 gradient steps without overfitting.

Input feature vector at each review event (dim = 113):
    x = concat([
        D_prev,          # current difficulty [1, 10]              → 1
        log(S_prev),     # log stability (numerical stability)     → 1
        R_at_review,     # retrievability at review time           → 1
        log(delta_t + 1),# log time since last review              → 1
        grade_onehot,    # [Again, Hard, Good, Easy]               → 4
        review_count,    # total reviews of this card              → 1
        card_embed,      # 64-dim projected BERT embedding         → 64
        user_stats,      # mean D, mean S, session length, etc.    → 8
        context_h,       # GRU hidden state from review history    → 32
    ])                                                       Total: 113

Output heads:
    S_next   = Softplus(linear) * S_prev   — stability grows on success
    D_next   = Sigmoid(linear) * 9 + 1     — difficulty in [1, 10]
    p_recall = Sigmoid(linear)             — recall probability (primary loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, NamedTuple

from .gru_encoder import GRUHistoryEncoder
from .card_embeddings import CardEmbeddingProjector


class MemoryState(NamedTuple):
    """Output of a MemoryNet forward pass."""
    S_next: torch.Tensor     # (batch,) predicted next stability
    D_next: torch.Tensor     # (batch,) predicted next difficulty
    p_recall: torch.Tensor   # (batch,) predicted recall probability


class MemoryNet(nn.Module):
    """
    Neural memory-state transition model.

    Architecture:
        Linear(113, 128) + LayerNorm + GELU
        Linear(128, 128) + LayerNorm + GELU
        Linear(128, 64)  + GELU
        → 3 output heads: S_next, D_next, p_recall
    """

    # Feature dimensions (must sum to input_dim=113)
    SCALAR_FEATURES = 4   # D_prev, log_S_prev, R_at_review, log_delta_t
    GRADE_DIM = 4          # one-hot [Again, Hard, Good, Easy]
    COUNT_DIM = 1          # review_count (log-scaled)

    def __init__(
        self,
        input_dim: int = 113,
        hidden_dim: int = 128,
        card_embed_dim: int = 64,
        card_raw_dim: int = 384,
        gru_hidden_dim: int = 32,
        user_stats_dim: int = 8,
        dropout: float = 0.1,
        history_len: int = 32,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.card_embed_dim = card_embed_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.user_stats_dim = user_stats_dim

        # Sub-modules (part of phi, updated in meta-training)
        self.card_projector = CardEmbeddingProjector(card_raw_dim, card_embed_dim)
        self.gru_encoder = GRUHistoryEncoder(gru_hidden_dim, max_len=history_len)

        # Main network
        self.h1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.h2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.h3 = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Output heads
        self.stability_head = nn.Linear(64, 1)
        self.difficulty_head = nn.Linear(64, 1)
        self.recall_head = nn.Linear(64, 1)

    def build_features(
        self,
        D_prev: torch.Tensor,
        S_prev: torch.Tensor,
        R_at_review: torch.Tensor,
        delta_t: torch.Tensor,
        grade: torch.Tensor,
        review_count: torch.Tensor,
        card_embedding_raw: torch.Tensor,
        user_stats: torch.Tensor,
        history_grades: Optional[torch.Tensor] = None,
        history_delta_ts: Optional[torch.Tensor] = None,
        history_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Assemble the 113-dim input feature vector.

        Args:
            D_prev:             (batch,) current difficulty [1, 10]
            S_prev:             (batch,) current stability in days
            R_at_review:        (batch,) retrievability at review time [0, 1]
            delta_t:            (batch,) days since last review
            grade:              (batch,) integer grade 1-4
            review_count:       (batch,) total reviews for this card
            card_embedding_raw: (batch, 384) raw BERT embedding
            user_stats:         (batch, 8) user-level statistics
            history_grades:     (batch, seq_len) grade history for GRU
            history_delta_ts:   (batch, seq_len) delta_t history for GRU
            history_lengths:    (batch,) actual sequence lengths

        Returns:
            x: (batch, 113) assembled feature vector
        """
        batch_size = D_prev.size(0)
        device = D_prev.device

        # Scalar features
        scalars = torch.stack([
            D_prev / 10.0,                              # normalise to ~[0, 1]
            torch.log(S_prev.clamp(min=1e-6)),          # log stability
            R_at_review,                                 # already [0, 1]
            torch.log(delta_t.clamp(min=0) + 1.0),      # log-scale
        ], dim=-1)  # (batch, 4)

        # Grade one-hot
        grade_oh = F.one_hot(
            (grade.long() - 1).clamp(0, 3), num_classes=4
        ).float()  # (batch, 4)

        # Review count (log-scaled)
        count_feat = torch.log(review_count.float().clamp(min=1)).unsqueeze(-1)  # (batch, 1)

        # Card embedding (trainable projection)
        card_embed = self.card_projector(card_embedding_raw)  # (batch, 64)

        # GRU context
        if history_grades is not None and history_delta_ts is not None:
            context_h = self.gru_encoder(
                history_grades.float(), history_delta_ts.float(), history_lengths
            )  # (batch, 32)
        else:
            context_h = self.gru_encoder.zero_state(batch_size, device)

        # Concatenate all features
        x = torch.cat([
            scalars,       # 4
            grade_oh,      # 4
            count_feat,    # 1
            card_embed,    # 64
            user_stats,    # 8
            context_h,     # 32
        ], dim=-1)  # Total: 113

        # Dynamic padding/truncation to match expected input_dim (safety net)
        if x.size(-1) < self.input_dim:
            pad = torch.zeros(batch_size, self.input_dim - x.size(-1), device=device)
            x = torch.cat([x, pad], dim=-1)
        elif x.size(-1) > self.input_dim:
            x = x[:, :self.input_dim]

        return x

    def forward_from_features(
        self, x: torch.Tensor, S_prev: torch.Tensor
    ) -> MemoryState:
        """
        Forward pass from pre-assembled feature vector.

        Args:
            x:      (batch, input_dim) — assembled features.
            S_prev: (batch,) — previous stability (for multiplicative head).

        Returns:
            MemoryState(S_next, D_next, p_recall)
        """
        h = self.h1(x)
        h = self.h2(h)
        h = self.h3(h)

        # Output heads
        S_next = F.softplus(self.stability_head(h).squeeze(-1)) * S_prev
        S_next = S_next.clamp(min=1e-3, max=36500.0)  # Stability: 0.001 to 100 years
        D_next = torch.sigmoid(self.difficulty_head(h).squeeze(-1)) * 9.0 + 1.0
        p_recall = torch.sigmoid(self.recall_head(h).squeeze(-1))

        return MemoryState(S_next=S_next, D_next=D_next, p_recall=p_recall)

    def forward(
        self,
        D_prev: torch.Tensor,
        S_prev: torch.Tensor,
        R_at_review: torch.Tensor,
        delta_t: torch.Tensor,
        grade: torch.Tensor,
        review_count: torch.Tensor,
        card_embedding_raw: torch.Tensor,
        user_stats: torch.Tensor,
        history_grades: Optional[torch.Tensor] = None,
        history_delta_ts: Optional[torch.Tensor] = None,
        history_lengths: Optional[torch.Tensor] = None,
    ) -> MemoryState:
        """
        Full forward pass: build features → network → predictions.
        """
        x = self.build_features(
            D_prev, S_prev, R_at_review, delta_t, grade,
            review_count, card_embedding_raw, user_stats,
            history_grades, history_delta_ts, history_lengths,
        )
        return self.forward_from_features(x, S_prev)

    def predict_recall(self, features: torch.Tensor, S_prev: torch.Tensor) -> torch.Tensor:
        """Convenience: return only p_recall from features."""
        return self.forward_from_features(features, S_prev).p_recall

    def predict_stability(self, features: torch.Tensor, S_prev: torch.Tensor) -> torch.Tensor:
        """Convenience: return only S_next from features."""
        return self.forward_from_features(features, S_prev).S_next

    def count_parameters(self) -> int:
        """Total trainable parameters (target: ~50K)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
