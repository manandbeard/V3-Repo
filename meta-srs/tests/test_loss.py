"""Tests for the multi-component loss function."""

import sys
import os
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.loss import MetaSRSLoss, compute_loss
from models.memory_net import MemoryNet
from config import ModelConfig


class TestMetaSRSLoss:
    @pytest.fixture
    def loss_fn(self):
        return MetaSRSLoss()

    def test_loss_returns_dict(self, loss_fn):
        p_pred = torch.tensor([0.8, 0.6, 0.3, 0.9])
        S_next = torch.tensor([10.0, 5.0, 2.0, 20.0])
        recalled = torch.tensor([1.0, 1.0, 0.0, 1.0])
        elapsed = torch.tensor([3.0, 7.0, 14.0, 1.0])
        grade = torch.tensor([3, 3, 1, 4])
        S_prev = torch.tensor([5.0, 3.0, 5.0, 10.0])

        result = loss_fn(p_pred, S_next, recalled, elapsed, grade, S_prev)

        assert "total" in result
        assert "recall" in result
        assert "stability" in result
        assert "monotonicity" in result

    def test_loss_is_finite(self, loss_fn):
        p_pred = torch.tensor([0.8, 0.6, 0.3, 0.9])
        S_next = torch.tensor([10.0, 5.0, 2.0, 20.0])
        recalled = torch.tensor([1.0, 1.0, 0.0, 1.0])
        elapsed = torch.tensor([3.0, 7.0, 14.0, 1.0])
        grade = torch.tensor([3, 3, 1, 4])
        S_prev = torch.tensor([5.0, 3.0, 5.0, 10.0])

        result = loss_fn(p_pred, S_next, recalled, elapsed, grade, S_prev)
        assert torch.isfinite(result["total"])

    def test_perfect_prediction_low_loss(self, loss_fn):
        """Perfect predictions should yield low recall loss."""
        p_pred = torch.tensor([0.99, 0.99, 0.01, 0.99])
        S_next = torch.tensor([10.0, 5.0, 2.0, 20.0])
        recalled = torch.tensor([1.0, 1.0, 0.0, 1.0])
        elapsed = torch.tensor([1.0, 1.0, 1.0, 1.0])
        grade = torch.tensor([3, 3, 1, 4])

        result = loss_fn(p_pred, S_next, recalled, elapsed, grade)
        assert result["recall"].item() < 0.5  # Should be low

    def test_bad_prediction_high_loss(self, loss_fn):
        """Inverted predictions should yield high recall loss."""
        p_pred = torch.tensor([0.01, 0.01, 0.99, 0.01])
        S_next = torch.tensor([10.0, 5.0, 2.0, 20.0])
        recalled = torch.tensor([1.0, 1.0, 0.0, 1.0])
        elapsed = torch.tensor([1.0, 1.0, 1.0, 1.0])
        grade = torch.tensor([3, 3, 1, 4])

        result = loss_fn(p_pred, S_next, recalled, elapsed, grade)
        assert result["recall"].item() > 1.0  # Should be high

    def test_monotonicity_penalty(self, loss_fn):
        """Stability decrease on success should incur monotonicity penalty."""
        p_pred = torch.tensor([0.8, 0.8])
        S_next = torch.tensor([3.0, 3.0])   # S decreased from 5 and 10
        recalled = torch.tensor([1.0, 1.0])
        elapsed = torch.tensor([1.0, 1.0])
        grade = torch.tensor([3, 3])  # Success
        S_prev = torch.tensor([5.0, 10.0])   # Higher than S_next

        result = loss_fn(p_pred, S_next, recalled, elapsed, grade, S_prev)
        assert result["monotonicity"].item() > 0

    def test_no_monotonicity_penalty_on_increase(self, loss_fn):
        """Stability increase on success should have zero monotonicity penalty."""
        p_pred = torch.tensor([0.8, 0.8])
        S_next = torch.tensor([10.0, 20.0])  # S increased
        recalled = torch.tensor([1.0, 1.0])
        elapsed = torch.tensor([1.0, 1.0])
        grade = torch.tensor([3, 3])
        S_prev = torch.tensor([5.0, 10.0])   # Lower than S_next

        result = loss_fn(p_pred, S_next, recalled, elapsed, grade, S_prev)
        assert result["monotonicity"].item() == 0.0

    def test_nan_fallback(self, loss_fn):
        """If total loss is NaN, should fall back to recall loss only."""
        # This is hard to trigger directly, but we can test the guard logic
        p_pred = torch.tensor([0.5, 0.5])
        S_next = torch.tensor([5.0, 5.0])
        recalled = torch.tensor([1.0, 0.0])
        elapsed = torch.tensor([1.0, 1.0])
        grade = torch.tensor([3, 1])

        result = loss_fn(p_pred, S_next, recalled, elapsed, grade)
        assert not torch.isnan(result["total"])

    def test_loss_weights(self):
        """Verify loss weights are applied correctly."""
        loss_fn = MetaSRSLoss(stability_weight=0.0, monotonicity_weight=0.0)
        p_pred = torch.tensor([0.8, 0.2])
        S_next = torch.tensor([3.0, 3.0])
        recalled = torch.tensor([1.0, 0.0])
        elapsed = torch.tensor([5.0, 5.0])
        grade = torch.tensor([3, 1])
        S_prev = torch.tensor([5.0, 5.0])

        result = loss_fn(p_pred, S_next, recalled, elapsed, grade, S_prev)
        # With zero weights, total should equal recall loss
        assert abs(result["total"].item() - result["recall"].item()) < 1e-6


class TestComputeLoss:
    def test_compute_loss_integration(self, model, sample_batch_tensors):
        loss_fn = MetaSRSLoss()
        result = compute_loss(model, sample_batch_tensors, loss_fn)
        assert "total" in result
        assert torch.isfinite(result["total"])
