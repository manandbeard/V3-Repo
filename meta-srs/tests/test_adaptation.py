"""Tests for FastAdapter (three-phase adaptation)."""

import sys
import os
import torch
import numpy as np
import pytest
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import MetaSRSConfig
from models.memory_net import MemoryNet
from inference.adaptation import FastAdapter, AdaptationPhase
from tests.conftest import create_model
from data.task_sampler import Review


def _make_review(card_id="c1", grade=3, elapsed=1.0, timestamp=0):
    return Review(
        card_id=card_id,
        timestamp=timestamp,
        elapsed_days=elapsed,
        grade=grade,
        recalled=grade >= 2,
        S_prev=3.0,
        D_prev=5.0,
        R_at_review=0.9,
        S_target=5.0,
        D_target=4.8,
    )


@pytest.fixture
def adapter():
    config = MetaSRSConfig()
    model = create_model(config.model)
    phi_star = deepcopy(model.state_dict())
    return FastAdapter(model, phi_star, config)


class TestAdaptationPhases:
    def test_initial_phase_is_zero_shot(self, adapter):
        assert adapter.phase == AdaptationPhase.ZERO_SHOT

    def test_phase_transitions(self, adapter):
        # Phase 1: 0-4 reviews → ZERO_SHOT
        for i in range(4):
            adapter.add_review(_make_review(timestamp=i))
        assert adapter.phase == AdaptationPhase.ZERO_SHOT

        # Phase 2: 5-49 reviews → RAPID_ADAPT
        adapter.add_review(_make_review(timestamp=5))
        assert adapter.phase == AdaptationPhase.RAPID_ADAPT

        # Add more to reach phase 3 (need >= 50 total)
        for i in range(6, 51):
            adapter.add_review(_make_review(timestamp=i))
        assert adapter.phase == AdaptationPhase.FULL_PERSONAL

    def test_n_reviews_property(self, adapter):
        assert adapter.n_reviews == 0
        adapter.add_review(_make_review())
        assert adapter.n_reviews == 1


class TestGetModel:
    def test_get_model_zero_shot_uses_phi(self, adapter):
        """In zero-shot phase, get_model should return model with phi_star."""
        model = adapter.get_model()
        assert isinstance(model, MemoryNet)

    def test_get_model_returns_eval_mode(self, adapter):
        model = adapter.get_model()
        assert not model.training

    def test_get_model_after_adaptation(self, adapter):
        """After adaptation, model should use theta_student."""
        for i in range(6):
            adapter.add_review(_make_review(
                card_id=f"c{i%3}", timestamp=i, elapsed=float(i+1)
            ))
        assert adapter.phase == AdaptationPhase.RAPID_ADAPT
        model = adapter.get_model()
        assert isinstance(model, MemoryNet)


class TestStateManagement:
    def test_get_state(self, adapter):
        adapter.add_review(_make_review())
        state = adapter.get_state()
        assert "theta_student" in state
        assert "reviews" in state
        assert "n_reviews" in state
        assert state["n_reviews"] == 1
        assert "phase" in state

    def test_load_state_roundtrip(self, adapter):
        for i in range(3):
            adapter.add_review(_make_review(timestamp=i))

        state = adapter.get_state()

        # Create new adapter and load state
        config = MetaSRSConfig()
        model = create_model(config.model)
        phi_star = deepcopy(model.state_dict())
        new_adapter = FastAdapter(model, phi_star, config)
        new_adapter.load_state(state)

        assert new_adapter.n_reviews == 3
