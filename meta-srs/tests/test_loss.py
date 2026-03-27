"""
Tests for MetaSRSLoss and the compute_loss helper (training/loss.py).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

from training.loss import MetaSRSLoss, compute_loss
from models.memory_net import MemoryNet


BATCH = 16


def _tensors(batch=BATCH, all_recalled=None, all_success=True):
    torch.manual_seed(42)
    p_recall = torch.rand(batch) * 0.4 + 0.5        # realistic probabilities
    S_next = torch.rand(batch) * 10.0 + 1.0
    S_prev = S_next / 2.0                            # smaller than S_next
    elapsed = torch.rand(batch) * 10.0 + 0.1
    if all_recalled is True:
        recalled = torch.ones(batch)
    elif all_recalled is False:
        recalled = torch.zeros(batch)
    else:
        recalled = (torch.rand(batch) > 0.3).float()
    grade = torch.randint(2, 5, (batch,))  # all successful
    return p_recall, S_next, recalled, elapsed, grade, S_prev


class TestMetaSRSLossComponents:
    @pytest.fixture
    def loss_fn(self):
        return MetaSRSLoss(stability_weight=0.10, monotonicity_weight=0.01)

    def test_returns_dict_with_all_keys(self, loss_fn):
        p, S_next, rec, el, gr, S_prev = _tensors()
        out = loss_fn(p, S_next, rec, el, gr, S_prev)
        assert "total" in out
        assert "recall" in out
        assert "stability" in out
        assert "monotonicity" in out

    def test_total_is_non_negative(self, loss_fn):
        p, S_next, rec, el, gr, S_prev = _tensors()
        out = loss_fn(p, S_next, rec, el, gr, S_prev)
        assert out["total"].item() >= 0.0

    def test_recall_loss_non_negative(self, loss_fn):
        p, S_next, rec, el, gr, S_prev = _tensors()
        out = loss_fn(p, S_next, rec, el, gr, S_prev)
        assert out["recall"].item() >= 0.0

    def test_stability_loss_non_negative(self, loss_fn):
        p, S_next, rec, el, gr, S_prev = _tensors()
        out = loss_fn(p, S_next, rec, el, gr, S_prev)
        assert out["stability"].item() >= 0.0

    def test_monotonicity_loss_non_negative(self, loss_fn):
        p, S_next, rec, el, gr, S_prev = _tensors()
        out = loss_fn(p, S_next, rec, el, gr, S_prev)
        assert out["monotonicity"].item() >= 0.0

    def test_monotonicity_zero_when_S_increases(self, loss_fn):
        p, S_next, rec, el, gr, _ = _tensors()
        # S_prev < S_next always → monotonicity loss should be 0
        S_prev = S_next * 0.5
        out = loss_fn(p, S_next, rec, el, gr, S_prev)
        assert out["monotonicity"].item() == pytest.approx(0.0, abs=1e-6)

    def test_monotonicity_positive_when_S_decreases(self, loss_fn):
        p, S_next, rec, el, _, _ = _tensors()
        grade = torch.full((BATCH,), 3)  # all "Good" (>=2, i.e. successful)
        # S_prev > S_next → violation → positive monotonicity loss
        S_prev = S_next * 2.0
        out = loss_fn(p, S_next, rec, el, grade, S_prev)
        assert out["monotonicity"].item() > 0.0

    def test_no_s_prev_sets_mono_loss_to_zero(self, loss_fn):
        p, S_next, rec, el, gr, _ = _tensors()
        out = loss_fn(p, S_next, rec, el, gr, S_prev=None)
        assert out["monotonicity"].item() == pytest.approx(0.0, abs=1e-6)

    def test_total_equals_weighted_sum(self, loss_fn):
        p, S_next, rec, el, gr, S_prev = _tensors()
        out = loss_fn(p, S_next, rec, el, gr, S_prev)
        expected = (
            out["recall"]
            + 0.10 * out["stability"]
            + 0.01 * out["monotonicity"]
        )
        assert torch.allclose(out["total"], expected, atol=1e-5)

    def test_perfect_recall_predictions_low_loss(self, loss_fn):
        # p close to 1 and all recalled → lower loss than random
        p = torch.ones(BATCH) * 0.99
        S_next = torch.ones(BATCH) * 5.0
        recalled = torch.ones(BATCH)
        elapsed = torch.ones(BATCH) * 3.0
        grade = torch.full((BATCH,), 3)
        out = loss_fn(p, S_next, recalled, elapsed, grade)
        assert out["recall"].item() < 0.2

    def test_nan_guard_falls_back_to_recall_loss(self, loss_fn):
        # Force NaN by using near-zero S_next (overflow in exp)
        p = torch.ones(BATCH) * 0.5
        S_next = torch.zeros(BATCH) * float("nan")   # NaN inputs
        recalled = torch.ones(BATCH) * 0.5
        elapsed = torch.ones(BATCH)
        grade = torch.full((BATCH,), 3)
        # Should not raise; NaN guard falls back to recall loss
        out = loss_fn(p, S_next, recalled, elapsed, grade)
        assert not torch.isnan(out["total"])

    def test_gradients_flow_through_loss(self, loss_fn):
        p_base = torch.rand(BATCH) * 0.4 + 0.5
        p_base.requires_grad_(True)
        S_base = torch.rand(BATCH) * 10.0 + 1.0
        S_base.requires_grad_(True)
        # These are leaf tensors, so .grad will be populated
        recalled = (torch.rand(BATCH) > 0.3).float()
        elapsed = torch.rand(BATCH) * 10.0
        grade = torch.randint(2, 5, (BATCH,))
        out = loss_fn(p_base, S_base, recalled, elapsed, grade)
        out["total"].backward()
        assert p_base.grad is not None
        assert S_base.grad is not None


class TestComputeLoss:
    @pytest.fixture
    def model(self):
        torch.manual_seed(5)
        return MemoryNet()

    @pytest.fixture
    def loss_fn(self):
        return MetaSRSLoss()

    def test_compute_loss_returns_dict(self, model, loss_fn):
        from data.task_sampler import Review, reviews_to_batch
        reviews = [
            Review("c1", 0, 1.0, 3, True, S_prev=1.0, D_prev=5.0),
            Review("c2", 86400, 2.0, 2, True, S_prev=2.0, D_prev=4.5),
        ]
        batch = reviews_to_batch(reviews, {}, torch.device("cpu"))
        out = compute_loss(model, batch, loss_fn)
        assert "total" in out
        assert out["total"].item() >= 0.0

    def test_compute_loss_no_nan(self, model, loss_fn):
        from data.task_sampler import Review, reviews_to_batch
        reviews = [
            Review(f"c{i}", i * 1000, float(i), (i % 4) + 1, (i % 4) >= 1)
            for i in range(10)
        ]
        batch = reviews_to_batch(reviews, {}, torch.device("cpu"))
        out = compute_loss(model, batch, loss_fn)
        assert not torch.isnan(out["total"])
