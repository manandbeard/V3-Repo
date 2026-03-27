"""Tests for uncertainty-aware scheduling."""

import sys
import os
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import MetaSRSConfig, SchedulingConfig
from models.memory_net import MemoryNet
from inference.scheduling import Scheduler, ScheduleResult
from tests.conftest import create_model


@pytest.fixture
def scheduler():
    config = MetaSRSConfig()
    model = create_model(config.model)
    return Scheduler(model, config.scheduling, mc_samples=5)


@pytest.fixture
def single_features():
    """Single card features for scheduling."""
    return torch.randn(1, 113), torch.tensor([10.0])


class TestComputeInterval:
    def test_basic_interval(self, scheduler):
        interval, conf, diff = scheduler.compute_interval(
            S_pred=10.0, D=5.0, sigma=0.0
        )
        assert isinstance(interval, int)
        assert 1 <= interval <= 365

    def test_min_interval(self, scheduler):
        interval, _, _ = scheduler.compute_interval(
            S_pred=0.1, D=5.0, sigma=0.0
        )
        assert interval >= scheduler.cfg.min_interval

    def test_max_interval(self, scheduler):
        interval, _, _ = scheduler.compute_interval(
            S_pred=1000.0, D=5.0, sigma=0.0
        )
        assert interval <= scheduler.cfg.max_interval

    def test_high_uncertainty_shortens_interval(self, scheduler):
        interval_low_sigma, _, _ = scheduler.compute_interval(
            S_pred=20.0, D=5.0, sigma=0.01
        )
        interval_high_sigma, _, _ = scheduler.compute_interval(
            S_pred=20.0, D=5.0, sigma=0.99
        )
        # High uncertainty should generally give shorter or equal interval
        # (due to fuzz, we check average behavior)
        assert interval_high_sigma <= interval_low_sigma + 5

    def test_confidence_factor_range(self, scheduler):
        _, conf, _ = scheduler.compute_interval(
            S_pred=10.0, D=5.0, sigma=0.5
        )
        assert 0.5 <= conf <= 1.0

    def test_difficulty_factor_range(self, scheduler):
        _, _, diff = scheduler.compute_interval(
            S_pred=10.0, D=5.0, sigma=0.0
        )
        assert 0.5 <= diff <= 1.5

    def test_hard_card_shorter_interval(self, scheduler):
        """Harder cards (D>5) should get shorter intervals on average."""
        # Run multiple times to account for fuzz randomness
        easy_intervals = []
        hard_intervals = []
        for _ in range(100):
            interval_easy, _, _ = scheduler.compute_interval(
                S_pred=20.0, D=2.0, sigma=0.0
            )
            interval_hard, _, _ = scheduler.compute_interval(
                S_pred=20.0, D=9.0, sigma=0.0
            )
            easy_intervals.append(interval_easy)
            hard_intervals.append(interval_hard)

        avg_easy = sum(easy_intervals) / len(easy_intervals)
        avg_hard = sum(hard_intervals) / len(hard_intervals)
        assert avg_hard < avg_easy


class TestPredictWithUncertainty:
    def test_returns_four_tensors(self, scheduler, single_features):
        features, S_prev = single_features
        p_mean, p_sigma, S_mean, D_mean = scheduler.predict_with_uncertainty(
            features, S_prev
        )
        assert p_mean.shape == (1,)
        assert p_sigma.shape == (1,)
        assert S_mean.shape == (1,)
        assert D_mean.shape == (1,)

    def test_sigma_non_negative(self, scheduler, single_features):
        features, S_prev = single_features
        _, p_sigma, _, _ = scheduler.predict_with_uncertainty(features, S_prev)
        assert (p_sigma >= 0).all()


class TestScheduleCard:
    def test_returns_schedule_result(self, scheduler, single_features):
        features, S_prev = single_features
        result = scheduler.schedule_card("card_001", features, S_prev)
        assert isinstance(result, ScheduleResult)
        assert result.card_id == "card_001"
        assert 1 <= result.interval_days <= 365
        assert 0.0 <= result.p_recall_mean <= 1.0
        assert result.p_recall_sigma >= 0.0


class TestScheduleDeck:
    def test_returns_list(self, scheduler):
        features = torch.randn(3, 113)
        S_prev = torch.tensor([5.0, 10.0, 20.0])
        results = scheduler.schedule_deck(
            ["c1", "c2", "c3"], features, S_prev
        )
        assert len(results) == 3
        assert all(isinstance(r, ScheduleResult) for r in results)


class TestSelectNextCard:
    def test_most_due(self, scheduler):
        results = [
            ScheduleResult("c1", 5, 0.8, 0.1, 10, 5, 1.0, 1.0),
            ScheduleResult("c2", 1, 0.3, 0.2, 3, 7, 0.9, 0.9),
            ScheduleResult("c3", 10, 0.9, 0.05, 20, 3, 1.0, 1.0),
        ]
        selected = scheduler.select_next_card(results, strategy="most_due")
        assert selected.card_id == "c2"  # Lowest interval

    def test_highest_uncertainty(self, scheduler):
        results = [
            ScheduleResult("c1", 5, 0.8, 0.1, 10, 5, 1.0, 1.0),
            ScheduleResult("c2", 1, 0.3, 0.3, 3, 7, 0.9, 0.9),
            ScheduleResult("c3", 10, 0.9, 0.05, 20, 3, 1.0, 1.0),
        ]
        selected = scheduler.select_next_card(results, strategy="highest_uncertainty")
        assert selected.card_id == "c2"  # Highest sigma

    def test_uncertainty_weighted(self, scheduler):
        results = [
            ScheduleResult("c1", 5, 0.8, 0.1, 10, 5, 1.0, 1.0),
            ScheduleResult("c2", 1, 0.3, 0.3, 3, 7, 0.9, 0.9),
        ]
        selected = scheduler.select_next_card(results, strategy="uncertainty_weighted")
        assert isinstance(selected, ScheduleResult)

    def test_invalid_strategy_raises(self, scheduler):
        results = [ScheduleResult("c1", 5, 0.8, 0.1, 10, 5, 1.0, 1.0)]
        with pytest.raises(ValueError):
            scheduler.select_next_card(results, strategy="invalid")

    def test_empty_results_raises(self, scheduler):
        with pytest.raises(ValueError):
            scheduler.select_next_card([], strategy="most_due")
