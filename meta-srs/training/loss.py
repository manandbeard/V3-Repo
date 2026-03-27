"""
Multi-Component Loss Function (Section 3.3).

Three loss components:

1. PRIMARY — Recall prediction (binary cross-entropy):
   BCELoss(p_pred, recalled)

2. AUXILIARY 1 — Stability-curve consistency:
   MSELoss(R_from_S, recalled)
   where R_from_S = (0.9^(1/S_pred))^(elapsed_days^w20)

3. AUXILIARY 2 — Monotonicity constraint:
   S cannot decrease on successful recall.
   ReLU(-(S_next - S_prev)[grade >= 2]).mean()

Combined: recall_L + 0.10 * stable_L + 0.01 * mono_L
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from config import TrainingConfig, FSRSConfig


class MetaSRSLoss(nn.Module):
    """
    Multi-component loss for MemoryNet training.
    Used in both warm-start pre-training and Reptile inner loop.
    """

    def __init__(
        self,
        stability_weight: float = 0.10,
        monotonicity_weight: float = 0.01,
        w20: float = 0.4665,
    ):
        super().__init__()
        self.stability_weight = stability_weight
        self.monotonicity_weight = monotonicity_weight
        self.w20 = w20

        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def forward(
        self,
        p_recall_pred: torch.Tensor,
        S_next_pred: torch.Tensor,
        recalled: torch.Tensor,
        elapsed_days: torch.Tensor,
        grade: torch.Tensor,
        S_prev: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the combined loss.

        Args:
            p_recall_pred: (batch,) predicted recall probabilities
            S_next_pred:   (batch,) predicted next stability values
            recalled:      (batch,) binary ground truth (grade >= 2)
            elapsed_days:  (batch,) days since last review
            grade:         (batch,) grade values 1-4
            S_prev:        (batch,) optional previous stability for monotonicity

        Returns:
            Dict with 'total', 'recall', 'stability', 'monotonicity' losses.
        """
        # PRIMARY: recall prediction (binary cross-entropy)
        # Convert probabilities to logits for numerical stability
        p_clamped = p_recall_pred.clamp(1e-6, 1 - 1e-6)
        logits = torch.log(p_clamped / (1 - p_clamped))
        recall_L = self.bce(logits, recalled.float())

        # AUXILIARY 1: stability-curve consistency
        # R_from_S = (0.9^(1/S_pred))^(elapsed_days^w20)
        # Computed in log-space for numerical stability:
        #   log(R) = (1/S) * log(0.9) * (t^w20)
        log_09 = -0.10536051565782628  # math.log(0.9)
        t_pow = elapsed_days.clamp(min=1e-6) ** self.w20
        log_R = log_09 * t_pow / S_next_pred.clamp(min=1e-3)
        R_from_S = torch.exp(log_R.clamp(min=-20, max=0))  # R ∈ (0, 1]
        stable_L = self.mse(R_from_S, recalled.float())

        # AUXILIARY 2: monotonicity (S cannot decrease on successful recall)
        if S_prev is not None and S_next_pred.size(0) > 1:
            # Compare consecutive predictions where recall was successful
            success_mask = (grade >= 2).float()
            delta_S = S_next_pred - S_prev
            # Penalise negative delta_S on successful reviews
            mono_L = (F.relu(-delta_S) * success_mask).mean()
        else:
            mono_L = torch.tensor(0.0, device=p_recall_pred.device)

        # Combined loss
        total = (
            recall_L
            + self.stability_weight * stable_L
            + self.monotonicity_weight * mono_L
        )

        # Guard against NaN (can occur from extreme S values during adaptation)
        if torch.isnan(total):
            total = recall_L  # Fall back to primary loss only

        return {
            "total": total,
            "recall": recall_L,
            "stability": stable_L,
            "monotonicity": mono_L,
        }


def compute_loss(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    loss_fn: MetaSRSLoss,
) -> Dict[str, torch.Tensor]:
    """
    Convenience function: run model forward + compute loss on a batch.

    Args:
        model: MemoryNet instance
        batch: Dict from reviews_to_batch() with keys:
            D_prev, S_prev, R_at_review, delta_t, grade,
            review_count, card_embedding_raw, user_stats, recalled
        loss_fn: MetaSRSLoss instance

    Returns:
        Dict with total loss and component breakdowns.
    """
    state = model(
        D_prev=batch["D_prev"],
        S_prev=batch["S_prev"],
        R_at_review=batch["R_at_review"],
        delta_t=batch["delta_t"],
        grade=batch["grade"],
        review_count=batch["review_count"],
        card_embedding_raw=batch["card_embedding_raw"],
        user_stats=batch["user_stats"],
        history_grades=batch.get("history_grades"),
        history_delta_ts=batch.get("history_delta_ts"),
        history_lengths=batch.get("history_lengths"),
    )

    return loss_fn(
        p_recall_pred=state.p_recall,
        S_next_pred=state.S_next,
        recalled=batch["recalled"],
        elapsed_days=batch["delta_t"],
        grade=batch["grade"],
        S_prev=batch["S_prev"],
    )
