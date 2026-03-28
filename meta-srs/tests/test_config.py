"""Tests for MetaSRS configuration."""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    MetaSRSConfig, ModelConfig, FSRSConfig,
    TrainingConfig, AdaptationConfig, SchedulingConfig,
)


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.input_dim == 49
        assert cfg.hidden_dim == 128
        assert cfg.gru_hidden_dim == 32
        assert cfg.history_len == 32
        assert cfg.user_stats_dim == 8
        assert cfg.dropout == 0.1
        assert cfg.mc_samples == 20

    def test_feature_dim_sum(self):
        """Verify input_dim = scalars(4) + grade(4) + count(1) + user(8) + gru(32)."""
        cfg = ModelConfig()
        expected = 4 + 4 + 1 + cfg.user_stats_dim + cfg.gru_hidden_dim
        assert cfg.input_dim == expected


class TestFSRSConfig:
    def test_default_weights(self):
        cfg = FSRSConfig()
        assert len(cfg.w) == 21
        assert cfg.w[0] == 0.40255   # initial stability (Again)
        assert cfg.w[20] == 0.4665   # power-law exponent

    def test_weights_are_positive(self):
        cfg = FSRSConfig()
        # All weights should be non-negative (w7 is a placeholder)
        for i, w in enumerate(cfg.w):
            assert w >= 0, f"w[{i}] = {w} is negative"


class TestTrainingConfig:
    def test_epsilon_schedule_boundaries(self):
        cfg = TrainingConfig()
        assert cfg.epsilon_schedule(0) == cfg.epsilon_start
        assert abs(cfg.epsilon_schedule(cfg.n_iters) - cfg.epsilon_end) < 1e-8

    def test_epsilon_schedule_monotonic_decrease(self):
        cfg = TrainingConfig()
        prev = cfg.epsilon_schedule(0)
        for i in range(1, 100):
            cur = cfg.epsilon_schedule(i * 500)
            assert cur <= prev + 1e-12, f"epsilon increased at iter {i*500}"
            prev = cur

    def test_outer_lr_schedule_boundaries(self):
        cfg = TrainingConfig()
        # At iter 0, cosine gives cos(0) = 1 → lr = lr_start
        assert abs(cfg.outer_lr_schedule(0) - cfg.outer_lr_start) < 1e-8
        # At iter n_iters, cosine gives cos(pi) = -1 → lr = lr_end
        assert abs(cfg.outer_lr_schedule(cfg.n_iters) - cfg.outer_lr_end) < 1e-8

    def test_outer_lr_schedule_monotonic_decrease(self):
        cfg = TrainingConfig()
        prev = cfg.outer_lr_schedule(0)
        for i in range(1, 100):
            cur = cfg.outer_lr_schedule(i * 500)
            assert cur <= prev + 1e-12, f"outer_lr increased at iter {i*500}"
            prev = cur


class TestAdaptationConfig:
    def test_phase_thresholds_ordered(self):
        cfg = AdaptationConfig()
        assert cfg.phase1_threshold < cfg.phase2_threshold

    def test_k_steps_increasing(self):
        cfg = AdaptationConfig()
        assert cfg.phase1_k_steps <= cfg.phase2_k_steps <= cfg.phase3_k_steps


class TestSchedulingConfig:
    def test_defaults(self):
        cfg = SchedulingConfig()
        assert cfg.desired_retention == 0.90
        assert cfg.min_interval == 1
        assert cfg.max_interval == 365
        assert cfg.fuzz_range[0] < cfg.fuzz_range[1]


class TestMetaSRSConfig:
    def test_aggregation(self):
        cfg = MetaSRSConfig()
        assert isinstance(cfg.model, ModelConfig)
        assert isinstance(cfg.fsrs, FSRSConfig)
        assert isinstance(cfg.training, TrainingConfig)
        assert isinstance(cfg.adaptation, AdaptationConfig)
        assert isinstance(cfg.scheduling, SchedulingConfig)
