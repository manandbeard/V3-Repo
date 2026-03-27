"""
Tests for FastAdapter and AdaptationPhase (inference/adaptation.py).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import torch
from copy import deepcopy

from inference.adaptation import FastAdapter, AdaptationPhase
from models.memory_net import MemoryNet
from data.task_sampler import Review
from config import MetaSRSConfig, AdaptationConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_review(card_id="A", grade=3, elapsed=1.0, i=0):
    return Review(
        card_id=card_id,
        timestamp=i * 1000,
        elapsed_days=elapsed,
        grade=grade,
        recalled=grade >= 2,
        S_prev=1.0 + i * 0.1,
        D_prev=5.0,
        R_at_review=0.9,
        S_target=2.0,
        D_target=5.0,
    )


def make_adapter(n_reviews=0, config=None):
    torch.manual_seed(0)
    model = MemoryNet()
    phi = deepcopy(model.state_dict())
    cfg = config or MetaSRSConfig()
    adapter = FastAdapter(model, phi, cfg, device=torch.device("cpu"))
    # Pre-populate with n_reviews (bypasses triggers)
    card_embs = {
        f"card_{i % 5}": np.random.randn(384).astype(np.float32)
        for i in range(max(n_reviews, 5))
    }
    adapter.card_embeddings = card_embs
    for i in range(n_reviews):
        r = make_review(f"card_{i % 5}", grade=(i % 4) + 1, elapsed=1.0, i=i)
        adapter.reviews.append(r)
        adapter._reviews_since_last_adapt += 1
    return adapter


# ---------------------------------------------------------------------------
# AdaptationPhase transitions
# ---------------------------------------------------------------------------

class TestAdaptationPhase:
    @pytest.fixture
    def cfg(self):
        return MetaSRSConfig()

    def test_zero_reviews_is_zero_shot(self, cfg):
        adapter = make_adapter(n_reviews=0, config=cfg)
        assert adapter.phase == AdaptationPhase.ZERO_SHOT

    def test_just_below_phase1_threshold_is_zero_shot(self, cfg):
        n = cfg.adaptation.phase1_threshold - 1
        adapter = make_adapter(n_reviews=n, config=cfg)
        assert adapter.phase == AdaptationPhase.ZERO_SHOT

    def test_at_phase1_threshold_is_rapid_adapt(self, cfg):
        n = cfg.adaptation.phase1_threshold
        adapter = make_adapter(n_reviews=n, config=cfg)
        assert adapter.phase == AdaptationPhase.RAPID_ADAPT

    def test_just_below_phase2_threshold_is_rapid_adapt(self, cfg):
        n = cfg.adaptation.phase2_threshold - 1
        adapter = make_adapter(n_reviews=n, config=cfg)
        assert adapter.phase == AdaptationPhase.RAPID_ADAPT

    def test_at_phase2_threshold_is_full_personal(self, cfg):
        n = cfg.adaptation.phase2_threshold
        adapter = make_adapter(n_reviews=n, config=cfg)
        assert adapter.phase == AdaptationPhase.FULL_PERSONAL

    def test_well_above_phase2_threshold_is_full_personal(self, cfg):
        adapter = make_adapter(n_reviews=200, config=cfg)
        assert adapter.phase == AdaptationPhase.FULL_PERSONAL


# ---------------------------------------------------------------------------
# n_reviews property
# ---------------------------------------------------------------------------

class TestNReviews:
    def test_n_reviews_matches_list_length(self):
        adapter = make_adapter(n_reviews=7)
        assert adapter.n_reviews == 7


# ---------------------------------------------------------------------------
# add_review
# ---------------------------------------------------------------------------

class TestAddReview:
    def test_review_is_appended(self):
        adapter = make_adapter()
        r = make_review("card_A", grade=3, i=0)
        adapter.add_review(r)
        assert adapter.n_reviews == 1

    def test_counter_increments(self):
        adapter = make_adapter()
        before = adapter._reviews_since_last_adapt
        r = make_review("card_A", grade=3, i=0)
        adapter.add_review(r)
        assert adapter._reviews_since_last_adapt == before + 1

    def test_adaptation_triggered_after_enough_reviews_phase2(self):
        """Filling up to phase2 threshold and adding N reviews triggers adaptation."""
        cfg = MetaSRSConfig()
        # Start just at phase1 threshold (rapid adapt phase)
        adapter = make_adapter(n_reviews=cfg.adaptation.phase1_threshold, config=cfg)
        # Reset counter
        adapter._reviews_since_last_adapt = 0
        # Add enough reviews to trigger adaptation
        for i in range(cfg.adaptation.adapt_every_n_reviews):
            r = make_review("card_A", grade=3, elapsed=1.0, i=i)
            adapter.add_review(r)
        # Counter should have been reset
        assert adapter._reviews_since_last_adapt == 0

    def test_theta_updated_after_adaptation(self):
        cfg = MetaSRSConfig()
        adapter = make_adapter(n_reviews=cfg.adaptation.phase1_threshold, config=cfg)
        theta_before = deepcopy(adapter.theta_student)
        adapter._reviews_since_last_adapt = 0
        for i in range(cfg.adaptation.adapt_every_n_reviews):
            r = make_review("card_A", grade=3, elapsed=1.0, i=i)
            adapter.add_review(r)
        theta_after = adapter.theta_student
        # At least one param should change
        any_changed = any(
            not torch.allclose(theta_before[k], theta_after[k])
            for k in theta_before
        )
        assert any_changed


# ---------------------------------------------------------------------------
# get_model
# ---------------------------------------------------------------------------

class TestGetModel:
    def test_returns_memory_net(self):
        adapter = make_adapter()
        m = adapter.get_model()
        assert isinstance(m, MemoryNet)

    def test_phase1_model_uses_phi_star(self):
        adapter = make_adapter(n_reviews=0)
        m = adapter.get_model()
        # In zero-shot phase, model should have phi_star weights
        for key in adapter.phi_star:
            assert torch.allclose(m.state_dict()[key], adapter.phi_star[key])

    def test_model_in_eval_mode(self):
        adapter = make_adapter(n_reviews=0)
        m = adapter.get_model()
        assert not m.training

    def test_phase2_model_uses_theta_student(self):
        cfg = MetaSRSConfig()
        adapter = make_adapter(n_reviews=cfg.adaptation.phase1_threshold, config=cfg)
        # Force a parameter change in theta_student
        adapter.theta_student["h1.0.weight"] = adapter.theta_student["h1.0.weight"] + 1.0
        m = adapter.get_model()
        # The model should reflect theta_student, not phi_star
        with_theta = m.state_dict()["h1.0.weight"]
        phi = adapter.phi_star["h1.0.weight"]
        assert not torch.allclose(with_theta, phi)


# ---------------------------------------------------------------------------
# get_state / load_state
# ---------------------------------------------------------------------------

class TestStateSerialization:
    def test_get_state_has_required_keys(self):
        adapter = make_adapter(n_reviews=3)
        state = adapter.get_state()
        assert "theta_student" in state
        assert "reviews" in state
        assert "n_reviews" in state
        assert "phase" in state

    def test_load_state_restores_reviews(self):
        adapter = make_adapter(n_reviews=5)
        state = adapter.get_state()

        fresh = make_adapter(n_reviews=0)
        fresh.load_state(state)
        assert len(fresh.reviews) == 5

    def test_load_state_restores_theta(self):
        adapter = make_adapter(n_reviews=5)
        adapter.theta_student["h1.0.weight"] = torch.zeros_like(
            adapter.theta_student["h1.0.weight"]
        )
        state = adapter.get_state()

        fresh = make_adapter(n_reviews=0)
        fresh.load_state(state)
        assert torch.allclose(
            fresh.theta_student["h1.0.weight"],
            torch.zeros_like(adapter.theta_student["h1.0.weight"])
        )

    def test_round_trip_preserves_n_reviews(self):
        adapter = make_adapter(n_reviews=7)
        state = adapter.get_state()
        fresh = make_adapter(n_reviews=0)
        fresh.load_state(state)
        assert fresh.n_reviews == 7
