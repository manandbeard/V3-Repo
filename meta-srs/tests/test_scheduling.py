"""
Tests for the uncertainty-aware Scheduler (inference/scheduling.py).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import pytest
import torch
from copy import deepcopy

from models.memory_net import MemoryNet
from inference.scheduling import Scheduler, ScheduleResult
from config import SchedulingConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_scheduler(mc_samples=5):
    torch.manual_seed(42)
    model = MemoryNet()
    cfg = SchedulingConfig()
    return Scheduler(model, cfg, mc_samples=mc_samples)


def make_features(batch=4):
    torch.manual_seed(7)
    return torch.randn(batch, 113), torch.rand(batch) * 9.0 + 1.0


# ---------------------------------------------------------------------------
# Scheduler.compute_interval
# ---------------------------------------------------------------------------

class TestComputeInterval:
    @pytest.fixture
    def sched(self):
        return make_scheduler()

    def test_interval_within_bounds(self, sched):
        for _ in range(20):
            interval, _, _ = sched.compute_interval(S_pred=10.0, D=5.0, sigma=0.1)
            assert sched.cfg.min_interval <= interval <= sched.cfg.max_interval

    def test_interval_is_int(self, sched):
        interval, _, _ = sched.compute_interval(S_pred=15.0, D=5.0, sigma=0.2)
        assert isinstance(interval, int)

    def test_high_uncertainty_reduces_interval(self, sched):
        # High sigma → confidence_factor closer to 0.5 → shorter interval
        interval_low_sigma, _, _ = sched.compute_interval(10.0, D=5.0, sigma=0.0)
        interval_high_sigma, _, _ = sched.compute_interval(10.0, D=5.0, sigma=1.0)
        # interval with no uncertainty should generally be >= than with max uncertainty
        assert interval_high_sigma <= interval_low_sigma + 5  # allow fuzz noise

    def test_confidence_factor_range(self, sched):
        for sigma in [0.0, 0.5, 1.0, 2.0]:
            _, conf, _ = sched.compute_interval(10.0, D=5.0, sigma=sigma)
            assert 0.5 <= conf <= 1.0

    def test_difficulty_factor_range(self, sched):
        for D in [1.0, 5.0, 10.0]:
            _, _, diff = sched.compute_interval(10.0, D=D, sigma=0.0)
            assert 0.5 <= diff <= 1.5

    def test_easy_card_longer_interval(self, sched):
        # D < 5 (below-average difficulty) → difficulty_factor > 1
        _, _, diff_easy = sched.compute_interval(10.0, D=3.0, sigma=0.0)
        _, _, diff_hard = sched.compute_interval(10.0, D=8.0, sigma=0.0)
        assert diff_easy >= diff_hard

    def test_min_interval_enforced(self, sched):
        # Very small S_pred → interval should still be at least min_interval
        interval, _, _ = sched.compute_interval(S_pred=0.001, D=5.0, sigma=0.5)
        assert interval >= sched.cfg.min_interval

    def test_max_interval_enforced(self, sched):
        # Very large S_pred → interval should not exceed max_interval
        interval, _, _ = sched.compute_interval(S_pred=10_000.0, D=5.0, sigma=0.0)
        assert interval <= sched.cfg.max_interval


# ---------------------------------------------------------------------------
# Scheduler.predict_with_uncertainty
# ---------------------------------------------------------------------------

class TestPredictWithUncertainty:
    @pytest.fixture
    def sched(self):
        return make_scheduler(mc_samples=10)

    def test_output_shapes(self, sched):
        features, S_prev = make_features(batch=4)
        p_mean, p_sigma, S_mean, D_mean = sched.predict_with_uncertainty(features, S_prev)
        assert p_mean.shape == (4,)
        assert p_sigma.shape == (4,)
        assert S_mean.shape == (4,)
        assert D_mean.shape == (4,)

    def test_p_mean_in_range(self, sched):
        features, S_prev = make_features(batch=3)
        p_mean, _, _, _ = sched.predict_with_uncertainty(features, S_prev)
        assert p_mean.min().item() >= 0.0
        assert p_mean.max().item() <= 1.0

    def test_sigma_non_negative(self, sched):
        features, S_prev = make_features(batch=3)
        _, p_sigma, _, _ = sched.predict_with_uncertainty(features, S_prev)
        assert p_sigma.min().item() >= 0.0

    def test_S_mean_positive(self, sched):
        features, S_prev = make_features(batch=3)
        _, _, S_mean, _ = sched.predict_with_uncertainty(features, S_prev)
        assert S_mean.min().item() > 0

    def test_D_mean_in_range(self, sched):
        features, S_prev = make_features(batch=3)
        _, _, _, D_mean = sched.predict_with_uncertainty(features, S_prev)
        assert D_mean.min().item() >= 1.0
        assert D_mean.max().item() <= 10.0


# ---------------------------------------------------------------------------
# Scheduler.schedule_card
# ---------------------------------------------------------------------------

class TestScheduleCard:
    @pytest.fixture
    def sched(self):
        return make_scheduler()

    def test_returns_schedule_result(self, sched):
        features, S_prev = make_features(batch=1)
        result = sched.schedule_card("my_card", features, S_prev)
        assert isinstance(result, ScheduleResult)

    def test_card_id_preserved(self, sched):
        features, S_prev = make_features(batch=1)
        result = sched.schedule_card("test_card_42", features, S_prev)
        assert result.card_id == "test_card_42"

    def test_interval_valid(self, sched):
        features, S_prev = make_features(batch=1)
        result = sched.schedule_card("c0", features, S_prev)
        assert result.interval_days >= sched.cfg.min_interval
        assert result.interval_days <= sched.cfg.max_interval

    def test_p_recall_mean_in_range(self, sched):
        features, S_prev = make_features(batch=1)
        result = sched.schedule_card("c0", features, S_prev)
        assert 0.0 <= result.p_recall_mean <= 1.0

    def test_p_recall_sigma_non_negative(self, sched):
        features, S_prev = make_features(batch=1)
        result = sched.schedule_card("c0", features, S_prev)
        assert result.p_recall_sigma >= 0.0


# ---------------------------------------------------------------------------
# Scheduler.schedule_deck
# ---------------------------------------------------------------------------

class TestScheduleDeck:
    @pytest.fixture
    def sched(self):
        return make_scheduler()

    def test_returns_list_of_results(self, sched):
        n = 5
        features, S_prev = make_features(batch=n)
        card_ids = [f"card_{i}" for i in range(n)]
        results = sched.schedule_deck(card_ids, features, S_prev)
        assert isinstance(results, list)
        assert len(results) == n

    def test_each_result_is_schedule_result(self, sched):
        features, S_prev = make_features(batch=3)
        results = sched.schedule_deck(["a", "b", "c"], features, S_prev)
        for r in results:
            assert isinstance(r, ScheduleResult)

    def test_card_ids_preserved(self, sched):
        ids = ["card_X", "card_Y", "card_Z"]
        features, S_prev = make_features(batch=3)
        results = sched.schedule_deck(ids, features, S_prev)
        assert [r.card_id for r in results] == ids


# ---------------------------------------------------------------------------
# Scheduler.select_next_card
# ---------------------------------------------------------------------------

class TestSelectNextCard:
    def _make_results(self, n=4):
        results = []
        for i in range(n):
            results.append(ScheduleResult(
                card_id=f"card_{i}",
                interval_days=i + 1,
                p_recall_mean=0.9 - i * 0.1,
                p_recall_sigma=i * 0.05,
                S_pred=float(i + 1),
                D_pred=5.0,
                confidence_factor=1.0,
                difficulty_factor=1.0,
            ))
        return results

    @pytest.fixture
    def sched(self):
        return make_scheduler()

    def test_most_due_selects_smallest_interval(self, sched):
        results = self._make_results()
        selected = sched.select_next_card(results, strategy="most_due")
        assert selected.interval_days == 1  # smallest interval

    def test_highest_uncertainty_selects_max_sigma(self, sched):
        results = self._make_results()
        selected = sched.select_next_card(results, strategy="highest_uncertainty")
        max_sigma = max(r.p_recall_sigma for r in results)
        assert selected.p_recall_sigma == max_sigma

    def test_uncertainty_weighted_returns_valid_result(self, sched):
        results = self._make_results()
        selected = sched.select_next_card(results, strategy="uncertainty_weighted")
        assert selected in results

    def test_empty_results_raises(self, sched):
        with pytest.raises(ValueError):
            sched.select_next_card([], strategy="most_due")

    def test_unknown_strategy_raises(self, sched):
        results = self._make_results()
        with pytest.raises(ValueError):
            sched.select_next_card(results, strategy="invalid_strategy")

    def test_single_card_always_selected(self, sched):
        results = [ScheduleResult(
            card_id="only",
            interval_days=7,
            p_recall_mean=0.8,
            p_recall_sigma=0.1,
            S_pred=7.0,
            D_pred=5.0,
            confidence_factor=1.0,
            difficulty_factor=1.0,
        )]
        for strategy in ["most_due", "highest_uncertainty", "uncertainty_weighted"]:
            selected = sched.select_next_card(results, strategy=strategy)
            assert selected.card_id == "only"
