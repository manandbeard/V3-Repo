"""
FSRS-6 Baseline & Warm-Start (Sections 1.1 and 3.4 tip).

Implements the FSRS-6 forgetting curve and stability update formulas for:
  1. Generating synthetic training targets for MemoryNet warm-start.
  2. Serving as the evaluation baseline.

Warm-start tip from the research:
    Pre-train MemoryNet to reproduce FSRS-6 predictions on a large corpus
    before running Reptile. This grounds the meta-initialization in a
    cognitively-valid baseline and cuts Reptile training time by ~60%.
"""

import math
import torch
import torch.nn as nn
from copy import deepcopy
from typing import List, Optional, Tuple

from config import FSRSConfig, ModelConfig, TrainingConfig


class FSRS6:
    """
    Deterministic FSRS-6 model using the 21-parameter forgetting curve
    and stability update formulas from Section 1.1.
    """

    def __init__(self, config: Optional[FSRSConfig] = None):
        self.cfg = config or FSRSConfig()
        self.w = self.cfg.w

    # ----- Forgetting Curve (power-law) -----

    def retrievability(self, t: float, S: float) -> float:
        """
        R(t, S) = (0.9^(1/S))^(t^w20)

        Power-law beats exponential because the population average of two
        exponential forgetting curves is better approximated by a power function.
        """
        if S <= 0:
            return 0.0
        w20 = self.w[20]
        return (0.9 ** (1.0 / S)) ** (t ** w20)

    def retrievability_batch(self, t: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """Vectorised retrievability for PyTorch tensors."""
        w20 = self.w[20]
        return (0.9 ** (1.0 / S.clamp(min=1e-6))) ** (t.clamp(min=0) ** w20)

    # ----- Initial State -----

    def initial_stability(self, grade: int) -> float:
        """S_0 for the first review of a card.  grade: 1=Again, 2=Hard, 3=Good, 4=Easy."""
        return self.w[grade - 1]  # w0..w3

    def initial_difficulty(self, grade: int) -> float:
        """D_0 based on first-review grade."""
        return self.w[4] - (grade - 3) * self.w[5]

    # ----- Stability Update on Success (Hard / Good / Easy) -----

    def stability_after_success(
        self, S: float, D: float, R: float, grade: int
    ) -> float:
        """
        S' = S * (e^w8 * (11-D) * S^(-w9) * (e^(w10*(1-R)) - 1) * w15_grade * w16_grade + 1)
        """
        w = self.w
        grade_modifier = 1.0
        if grade == 2:  # Hard
            grade_modifier = w[15]
        elif grade == 4:  # Easy
            grade_modifier = w[16]

        inner = (
            math.exp(w[8])
            * (11 - D)
            * (S ** (-w[9]))
            * (math.exp(w[10] * (1 - R)) - 1)
            * grade_modifier
            + 1
        )
        return S * inner

    # ----- Stability Update on Lapse (Again) -----

    def stability_after_lapse(self, S: float, D: float, R: float) -> float:
        """
        S_f = w11 * D^(-w12) * ((S+1)^w13 - 1) * e^(w14*(1-R))
        """
        w = self.w
        return (
            w[11]
            * (D ** (-w[12]))
            * ((S + 1) ** w[13] - 1)
            * math.exp(w[14] * (1 - R))
        )

    # ----- Difficulty Update -----

    def update_difficulty(self, D: float, grade: int) -> float:
        """
        dD = -w6 * (grade - 3)
        D'' = D + dD * (10 - D) / 9       # damping toward D=10
        D'  = w5 * D0(Easy) + (1 - w5) * D''  # mean reversion to default
        """
        w = self.w
        dD = -w[6] * (grade - 3)
        D_double_prime = D + dD * (10 - D) / 9.0
        D0_easy = self.initial_difficulty(4)
        D_prime = w[5] * D0_easy + (1 - w[5]) * D_double_prime
        return max(1.0, min(10.0, D_prime))

    # ----- Full Step -----

    def step(
        self, S: float, D: float, elapsed_days: float, grade: int
    ) -> Tuple[float, float, float]:
        """
        Process one review event.

        Returns:
            (S_next, D_next, R_at_review)
        """
        R = self.retrievability(elapsed_days, S) if elapsed_days > 0 else 1.0

        # Update stability
        if grade == 1:  # Again (lapse)
            S_next = self.stability_after_lapse(S, D, R)
        else:  # Hard / Good / Easy (success)
            S_next = self.stability_after_success(S, D, R, grade)

        # Update difficulty
        D_next = self.update_difficulty(D, grade)

        return S_next, D_next, R

    def simulate_student(
        self, reviews: List[dict]
    ) -> List[dict]:
        """
        Run FSRS-6 on a student's full review history to generate
        ground-truth (S, D, R) trajectories for warm-start training.

        Args:
            reviews: List of dicts with 'card_id', 'elapsed_days', 'grade'.

        Returns:
            List of dicts augmented with 'S', 'D', 'R' values.
        """
        card_states = {}  # card_id → (S, D)
        results = []

        for review in reviews:
            cid = review["card_id"]
            grade = review["grade"]
            elapsed = review["elapsed_days"]

            if cid not in card_states:
                # First review of this card
                S = self.initial_stability(grade)
                D = self.initial_difficulty(grade)
                R = 1.0  # first review: no forgetting yet
            else:
                S_prev, D_prev = card_states[cid]
                S, D, R = self.step(S_prev, D_prev, elapsed, grade)

            card_states[cid] = (S, D)
            results.append({
                **review,
                "S": S,
                "D": D,
                "R": R,
                "recalled": grade >= 2,
            })

        return results


def warm_start_from_fsrs6(
    model: nn.Module,
    dataset,
    config: Optional[TrainingConfig] = None,
    device: str = "cpu",
) -> nn.Module:
    """
    Pre-train MemoryNet to reproduce FSRS-6 predictions on a large corpus.

    This grounds the meta-initialization in a cognitively-valid baseline
    and cuts Reptile training time by ~60%.

    Args:
        model: MemoryNet instance.
        dataset: Iterable yielding batches of (features, S_prev, fsrs_targets).
            - features: (batch, input_dim)
            - S_prev: (batch,) previous stability
            - fsrs_targets: dict with 'S_target', 'D_target', 'R_target' tensors
        config: Training config for warmstart params.
        device: Device string.

    Returns:
        Pre-trained model (modified in-place).
    """
    cfg = config or TrainingConfig()
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.warmstart_lr)
    mse = nn.MSELoss()
    bce = nn.BCELoss()

    for epoch in range(cfg.warmstart_epochs):
        total_loss = 0.0
        n_batches = 0

        for features, S_prev, targets in dataset:
            features = features.to(device)
            S_prev = S_prev.to(device)
            S_target = targets["S_target"].to(device)
            D_target = targets["D_target"].to(device)
            R_target = targets["R_target"].to(device)

            state = model.forward_from_features(features, S_prev)

            loss = (
                bce(state.p_recall, R_target)
                + 0.5 * mse(torch.log(state.S_next + 1), torch.log(S_target + 1))
                + 0.5 * mse(state.D_next, D_target)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg = total_loss / max(n_batches, 1)
        print(f"  Warm-start epoch {epoch+1}/{cfg.warmstart_epochs}  loss={avg:.4f}")

    return model
