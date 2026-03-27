"""
Fast Adaptation for New Students (Section 4.1).

Three-Phase Onboarding Protocol:

Phase 1 — Zero-shot (0 reviews):
    Use phi* directly. Schedule with meta-parameter predictions.
    Apply uncertainty-aware exploration — mix predicted-easy and
    uncertain cards to gather calibration data quickly.

Phase 2 — Rapid adapt (5–50 reviews):
    Every 5 new reviews: run k=5 gradient steps from phi* on student
    history. Switch scheduling from phi* to theta_student. AUC
    already exceeds population-average FSRS-6.

Phase 3 — Full personal (50+ reviews):
    Run k=20 gradient steps at higher inner learning rate. Enable
    online streaming updates (gradient step after every review).
    Reach full personalization within the first study session.
"""

import torch
import torch.nn as nn
from copy import deepcopy
from enum import Enum, auto
from typing import Dict, List, Optional
from collections import OrderedDict

from config import AdaptationConfig, MetaSRSConfig
from models.memory_net import MemoryNet
from training.loss import MetaSRSLoss, compute_loss
from data.task_sampler import Review, reviews_to_batch


class AdaptationPhase(Enum):
    """The three onboarding phases from Section 4.1."""
    ZERO_SHOT = auto()    # Phase 1: 0 reviews, use phi* directly
    RAPID_ADAPT = auto()  # Phase 2: 5–50 reviews, periodic adaptation
    FULL_PERSONAL = auto() # Phase 3: 50+ reviews, streaming updates


class FastAdapter:
    """
    Online fast-adaptation engine for a single student.

    Manages the transition through the three onboarding phases and
    maintains the student's personalised parameters theta_student.

    Usage:
        adapter = FastAdapter(model, phi_star, config)
        # Student reviews a card:
        adapter.add_review(review)
        # Get current model for scheduling:
        model = adapter.get_model()
    """

    def __init__(
        self,
        model: MemoryNet,
        phi_star: OrderedDict,
        config: Optional[MetaSRSConfig] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model.to(device)
        self.phi_star = deepcopy(phi_star)
        self.cfg = (config or MetaSRSConfig()).adaptation
        self.fsrs_cfg = (config or MetaSRSConfig()).fsrs
        self.device = device

        # Student's personalised parameters
        self.theta_student = deepcopy(phi_star)
        self.reviews: List[Review] = []
        self.card_embeddings: Dict[str, any] = {}
        self._reviews_since_last_adapt = 0

        self.loss_fn = MetaSRSLoss(w20=self.fsrs_cfg.w[20])

    @property
    def phase(self) -> AdaptationPhase:
        """Determine current onboarding phase based on review count."""
        n = len(self.reviews)
        if n < self.cfg.phase1_threshold:
            return AdaptationPhase.ZERO_SHOT
        elif n < self.cfg.phase2_threshold:
            return AdaptationPhase.RAPID_ADAPT
        else:
            return AdaptationPhase.FULL_PERSONAL

    @property
    def n_reviews(self) -> int:
        return len(self.reviews)

    def add_review(self, review: Review):
        """
        Process a new review event. Triggers adaptation if appropriate
        based on the current phase.
        """
        self.reviews.append(review)
        self._reviews_since_last_adapt += 1

        phase = self.phase

        if phase == AdaptationPhase.ZERO_SHOT:
            # No adaptation yet — use phi* directly
            pass

        elif phase == AdaptationPhase.RAPID_ADAPT:
            # Every N reviews: run k=5 gradient steps from phi*
            if self._reviews_since_last_adapt >= self.cfg.adapt_every_n_reviews:
                self._adapt(
                    k_steps=self.cfg.phase1_k_steps,
                    inner_lr=0.01,
                    from_phi=True,  # Always adapt from phi*, not theta
                )
                self._reviews_since_last_adapt = 0

        elif phase == AdaptationPhase.FULL_PERSONAL:
            if self._reviews_since_last_adapt >= self.cfg.adapt_every_n_reviews:
                # Run k=20 gradient steps at higher LR
                self._adapt(
                    k_steps=self.cfg.phase3_k_steps,
                    inner_lr=self.cfg.phase3_inner_lr,
                    from_phi=False,  # Fine-tune from current theta
                )
                self._reviews_since_last_adapt = 0
            elif len(self.reviews) >= self.cfg.streaming_after:
                # Streaming: single gradient step after every review
                self._streaming_step()

    def _adapt(
        self,
        k_steps: int,
        inner_lr: float,
        from_phi: bool = True,
    ):
        """Run k gradient steps for adaptation."""
        # Start from phi* or current theta
        start_params = self.phi_star if from_phi else self.theta_student
        self.model.load_state_dict(deepcopy(start_params))
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=inner_lr)

        # Use all available reviews as training data
        batch = reviews_to_batch(
            self.reviews, self.card_embeddings, self.device
        )

        for step in range(k_steps):
            losses = compute_loss(self.model, batch, self.loss_fn)
            loss = losses["total"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

        self.theta_student = deepcopy(self.model.state_dict())

    def _streaming_step(self):
        """Single gradient step on the most recent review (online learning)."""
        if not self.reviews:
            return

        self.model.load_state_dict(deepcopy(self.theta_student))
        self.model.train()

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.cfg.phase3_inner_lr
        )

        # Use last few reviews as a mini-batch
        recent = self.reviews[-min(8, len(self.reviews)):]
        batch = reviews_to_batch(recent, self.card_embeddings, self.device)

        losses = compute_loss(self.model, batch, self.loss_fn)
        loss = losses["total"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()

        self.theta_student = deepcopy(self.model.state_dict())

    def get_model(self) -> MemoryNet:
        """
        Get the current model with appropriate parameters for scheduling.

        Phase 1: Returns model with phi* (meta-parameters).
        Phase 2–3: Returns model with theta_student (adapted).
        """
        if self.phase == AdaptationPhase.ZERO_SHOT:
            self.model.load_state_dict(self.phi_star)
        else:
            self.model.load_state_dict(self.theta_student)
        self.model.eval()
        return self.model

    def get_state(self) -> dict:
        """Serialise adapter state for persistence."""
        return {
            "theta_student": self.theta_student,
            "reviews": self.reviews,
            "n_reviews": len(self.reviews),
            "phase": self.phase.name,
            "reviews_since_adapt": self._reviews_since_last_adapt,
        }

    def load_state(self, state: dict):
        """Restore adapter state."""
        self.theta_student = state["theta_student"]
        self.reviews = state["reviews"]
        self._reviews_since_last_adapt = state.get("reviews_since_adapt", 0)
