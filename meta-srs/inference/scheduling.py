"""
Uncertainty-Aware Scheduling (Section 4.2).

Monte Carlo Dropout for uncertainty quantification:
    def predict_with_uncertainty(theta, x, n_samples=20):
        model.train()  # keep dropout active at inference
        preds = [model(x) for _ in range(n_samples)]
        mean  = torch.stack(preds).mean(0)  # expected recall probability
        sigma = torch.stack(preds).std(0)   # epistemic uncertainty
        return mean, sigma

Interval with uncertainty and difficulty discounting:
    def compute_interval(S_pred, D, sigma, desired_retention=0.90):
        base_interval = S_pred  # when DR=0.90, I ~ S
        confidence_factor = 1.0 - 0.5 * sigma     # range [0.5, 1.0]
        difficulty_factor = 1.0 - 0.05 * (D - 5)  # centre on D=5
        fuzz = uniform(0.95, 1.05)                 # prevent card clustering
        return max(1, round(base_interval * confidence_factor
                            * difficulty_factor * fuzz))
"""

import random
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from config import SchedulingConfig, ModelConfig
from models.memory_net import MemoryNet


@dataclass
class ScheduleResult:
    """Result of scheduling a single card."""
    card_id: str
    interval_days: int            # Recommended review interval
    p_recall_mean: float          # Expected recall probability
    p_recall_sigma: float         # Epistemic uncertainty
    S_pred: float                 # Predicted stability (days)
    D_pred: float                 # Predicted difficulty [1, 10]
    confidence_factor: float      # Uncertainty discount applied
    difficulty_factor: float      # Difficulty discount applied


class Scheduler:
    """
    Uncertainty-aware card scheduler.

    Uses Monte Carlo Dropout to estimate both the expected recall
    probability and its epistemic uncertainty, then computes review
    intervals with confidence and difficulty discounting.
    """

    def __init__(
        self,
        model: MemoryNet,
        config: Optional[SchedulingConfig] = None,
        mc_samples: int = 20,
    ):
        self.model = model
        self.cfg = config or SchedulingConfig()
        self.mc_samples = mc_samples

    def predict_with_uncertainty(
        self,
        features: torch.Tensor,
        S_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Monte Carlo Dropout: run the model n_samples times with dropout
        active to estimate prediction uncertainty.

        Args:
            features: (batch, input_dim) pre-assembled feature vectors.
            S_prev:   (batch,) previous stability values.

        Returns:
            p_recall_mean:  (batch,) expected recall probability
            p_recall_sigma: (batch,) epistemic uncertainty
            S_mean:         (batch,) expected next stability
            D_mean:         (batch,) expected next difficulty
        """
        self.model.train()  # Keep dropout active for MC sampling

        all_p_recall = []
        all_S = []
        all_D = []

        with torch.no_grad():
            for _ in range(self.mc_samples):
                state = self.model.forward_from_features(features, S_prev)
                all_p_recall.append(state.p_recall)
                all_S.append(state.S_next)
                all_D.append(state.D_next)

        p_stack = torch.stack(all_p_recall)  # (n_samples, batch)
        S_stack = torch.stack(all_S)
        D_stack = torch.stack(all_D)

        return (
            p_stack.mean(0),
            p_stack.std(0),
            S_stack.mean(0),
            D_stack.mean(0),
        )

    def compute_interval(
        self,
        S_pred: float,
        D: float,
        sigma: float,
    ) -> Tuple[int, float, float]:
        """
        Compute review interval with uncertainty and difficulty discounting.

        Args:
            S_pred: Predicted stability (days until R drops to 90%).
            D: Predicted difficulty [1, 10].
            sigma: Epistemic uncertainty of recall prediction.

        Returns:
            (interval_days, confidence_factor, difficulty_factor)
        """
        base_interval = S_pred  # When desired_retention=0.90, interval ~ S

        # Uncertainty discount: high uncertainty → shorter interval
        confidence_factor = 1.0 - self.cfg.uncertainty_discount * min(sigma, 1.0)
        confidence_factor = max(0.5, min(1.0, confidence_factor))

        # Difficulty discount: harder cards get slightly shorter intervals
        difficulty_factor = 1.0 - self.cfg.difficulty_slope * (
            D - self.cfg.difficulty_centre
        )
        difficulty_factor = max(0.5, min(1.5, difficulty_factor))

        # Fuzz to prevent card clustering
        fuzz_lo, fuzz_hi = self.cfg.fuzz_range
        fuzz = random.uniform(fuzz_lo, fuzz_hi)

        interval = base_interval * confidence_factor * difficulty_factor * fuzz
        interval = max(self.cfg.min_interval, min(self.cfg.max_interval, round(interval)))

        return int(interval), confidence_factor, difficulty_factor

    def schedule_card(
        self,
        card_id: str,
        features: torch.Tensor,
        S_prev: torch.Tensor,
    ) -> ScheduleResult:
        """
        Schedule a single card: predict memory state + compute interval.

        Args:
            card_id: Identifier for the card.
            features: (1, input_dim) feature vector for this card/review.
            S_prev: (1,) previous stability.

        Returns:
            ScheduleResult with interval and prediction details.
        """
        p_mean, p_sigma, S_mean, D_mean = self.predict_with_uncertainty(
            features, S_prev
        )

        interval, conf_factor, diff_factor = self.compute_interval(
            S_pred=S_mean.item(),
            D=D_mean.item(),
            sigma=p_sigma.item(),
        )

        return ScheduleResult(
            card_id=card_id,
            interval_days=interval,
            p_recall_mean=p_mean.item(),
            p_recall_sigma=p_sigma.item(),
            S_pred=S_mean.item(),
            D_pred=D_mean.item(),
            confidence_factor=conf_factor,
            difficulty_factor=diff_factor,
        )

    def schedule_deck(
        self,
        card_ids: List[str],
        features: torch.Tensor,
        S_prev: torch.Tensor,
    ) -> List[ScheduleResult]:
        """
        Schedule an entire deck: batch MC-Dropout + individual intervals.

        Args:
            card_ids: List of card identifiers.
            features: (n_cards, input_dim) feature vectors.
            S_prev: (n_cards,) previous stability values.

        Returns:
            List of ScheduleResult, one per card.
        """
        p_mean, p_sigma, S_mean, D_mean = self.predict_with_uncertainty(
            features, S_prev
        )

        results = []
        for i, card_id in enumerate(card_ids):
            interval, conf_factor, diff_factor = self.compute_interval(
                S_pred=S_mean[i].item(),
                D=D_mean[i].item(),
                sigma=p_sigma[i].item(),
            )
            results.append(ScheduleResult(
                card_id=card_id,
                interval_days=interval,
                p_recall_mean=p_mean[i].item(),
                p_recall_sigma=p_sigma[i].item(),
                S_pred=S_mean[i].item(),
                D_pred=D_mean[i].item(),
                confidence_factor=conf_factor,
                difficulty_factor=diff_factor,
            ))

        return results

    def select_next_card(
        self,
        results: List[ScheduleResult],
        strategy: str = "uncertainty_weighted",
    ) -> ScheduleResult:
        """
        Select which card to review next using uncertainty-aware exploration.

        Strategies:
            'most_due': Card with lowest interval (most overdue).
            'highest_uncertainty': Card with highest sigma.
            'uncertainty_weighted': Mix of due-ness and uncertainty
                (Phase 1 exploration: mix predicted-easy and uncertain cards).
        """
        if not results:
            raise ValueError("No cards to schedule")

        if strategy == "most_due":
            return min(results, key=lambda r: r.interval_days)

        elif strategy == "highest_uncertainty":
            return max(results, key=lambda r: r.p_recall_sigma)

        elif strategy == "uncertainty_weighted":
            # Score = (1 - p_recall) + 0.5 * sigma
            # High score → more urgent (forgotten or uncertain)
            scores = [
                (1.0 - r.p_recall_mean) + 0.5 * r.p_recall_sigma
                for r in results
            ]
            # Softmax selection for stochastic exploration
            scores_t = torch.tensor(scores)
            probs = torch.softmax(scores_t * 3.0, dim=0)  # temperature=1/3
            idx = torch.multinomial(probs, 1).item()
            return results[idx]

        else:
            raise ValueError(f"Unknown strategy: {strategy}")
