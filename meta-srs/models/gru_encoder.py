"""
GRU History Encoder (Section 2.4).

Captures per-card review trajectory patterns — repeated 'Again' presses,
irregular gaps, etc. — and produces a fixed-size context vector that is
appended to the MemoryNet input.

Input sequence per card:
    [(grade_1, delta_t_1), (grade_2, delta_t_2), ..., (grade_n, delta_t_n)]

Output:
    context_h  — final GRU hidden state (dim = gru_hidden_dim)
"""

import torch
import torch.nn as nn
from typing import Optional


class GRUHistoryEncoder(nn.Module):
    """1-layer GRU that encodes a card's review history into a context vector."""

    # Each timestep: grade (1-4 as float) + log(delta_t + 1)
    FEATURE_DIM = 2

    def __init__(self, hidden_dim: int = 32, max_len: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        self.gru = nn.GRU(
            input_size=self.FEATURE_DIM,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Small input projection to help with numerical scale
        self.input_proj = nn.Linear(self.FEATURE_DIM, self.FEATURE_DIM)

    def forward(
        self,
        grades: torch.Tensor,
        delta_ts: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            grades:   (batch, seq_len) — grade values 1–4 as float.
            delta_ts: (batch, seq_len) — days since previous review.
            lengths:  (batch,) — actual sequence lengths (for packing).

        Returns:
            context_h: (batch, hidden_dim) — final hidden state.
        """
        # Normalise inputs
        grade_norm = grades / 4.0                         # [0, 1]
        dt_norm = torch.log(delta_ts.clamp(min=0) + 1.0)  # log-scale

        # (batch, seq_len, 2)
        x = torch.stack([grade_norm, dt_norm], dim=-1)
        x = self.input_proj(x)

        # Truncate to max_len (keep most recent reviews)
        if x.size(1) > self.max_len:
            x = x[:, -self.max_len:, :]
            if lengths is not None:
                lengths = lengths.clamp(max=self.max_len)

        if lengths is not None:
            # Pack for variable-length sequences
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
            )
            _, h_n = self.gru(packed)
        else:
            _, h_n = self.gru(x)

        # h_n: (1, batch, hidden_dim) → (batch, hidden_dim)
        return h_n.squeeze(0)

    def zero_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return a zero context vector for cards with no prior history."""
        return torch.zeros(batch_size, self.hidden_dim, device=device)
