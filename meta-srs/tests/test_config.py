"""
Tests for MetaSRS configuration classes (config.py).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import pytest

from config import (
    ModelConfig,
    FSRSConfig,
    TrainingConfig,
    AdaptationConfig,
    SchedulingConfig,
    MetaSRSConfig,
)


class TestModelConfig:
    def test_default_input_dim(self):
        cfg = ModelConfig()
        assert cfg.input_dim == 113

    def test_feature_dim_sum(self):
        # 4 scalars + 4 grade_onehot + 1 count + 64 card_embed + 8 user_stats + 32 gru = 113
        cfg = ModelConfig()
        total = (
            4             # scalars
            + 4           # grade one-hot
            + 1           # review count
            + cfg.card_embed_dim   # 64
            + cfg.user_stats_dim   # 8
            + cfg.gru_hidden_dim   # 32
        )
        assert total == cfg.input_dim

    def test_custom_values(self):
        cfg = ModelConfig(hidden_dim=64, dropout=0.2)
        assert cfg.hidden_dim == 64
        assert cfg.dropout == 0.2


class TestFSRSConfig:
    def test_weight_count(self):
        cfg = FSRSConfig()
        assert len(cfg.w) == 21  # w0..w20

    def test_w20_positive(self):
        cfg = FSRSConfig()
        assert cfg.w[20] > 0  # power-law exponent must be positive

    def test_initial_stabilities_increasing_with_grade(self):
        cfg = FSRSConfig()
        # w0 < w1 < w2 < w3 (Again → Easy: increasing initial stability)
        assert cfg.w[0] < cfg.w[1] < cfg.w[2] < cfg.w[3]


class TestTrainingConfig:
    def test_epsilon_schedule_start(self):
        cfg = TrainingConfig()
        eps = cfg.epsilon_schedule(0)
        assert abs(eps - cfg.epsilon_start) < 1e-9

    def test_epsilon_schedule_end(self):
        cfg = TrainingConfig()
        eps = cfg.epsilon_schedule(cfg.n_iters)
        assert abs(eps - cfg.epsilon_end) < 1e-9

    def test_epsilon_schedule_monotone_decreasing(self):
        cfg = TrainingConfig()
        prev = cfg.epsilon_schedule(0)
        for it in range(1000, cfg.n_iters + 1, 1000):
            cur = cfg.epsilon_schedule(it)
            assert cur <= prev + 1e-12  # non-increasing
            prev = cur

    def test_epsilon_schedule_beyond_n_iters(self):
        cfg = TrainingConfig()
        eps = cfg.epsilon_schedule(cfg.n_iters * 10)
        assert abs(eps - cfg.epsilon_end) < 1e-9

    def test_outer_lr_schedule_start(self):
        cfg = TrainingConfig()
        lr = cfg.outer_lr_schedule(0)
        assert abs(lr - cfg.outer_lr_start) < 1e-9

    def test_outer_lr_schedule_end(self):
        cfg = TrainingConfig()
        lr = cfg.outer_lr_schedule(cfg.n_iters)
        assert abs(lr - cfg.outer_lr_end) < 1e-9

    def test_outer_lr_schedule_monotone_decreasing(self):
        cfg = TrainingConfig()
        prev = cfg.outer_lr_schedule(0)
        for it in range(1000, cfg.n_iters + 1, 1000):
            cur = cfg.outer_lr_schedule(it)
            assert cur <= prev + 1e-12
            prev = cur

    def test_outer_lr_is_cosine(self):
        cfg = TrainingConfig()
        half = cfg.n_iters // 2
        lr_half = cfg.outer_lr_schedule(half)
        # At midpoint, cosine factor = 0.5, so lr ~ midpoint of range
        expected = cfg.outer_lr_end + (cfg.outer_lr_start - cfg.outer_lr_end) * 0.5
        assert abs(lr_half - expected) < 1e-6


class TestAdaptationConfig:
    def test_phase_thresholds_ordered(self):
        cfg = AdaptationConfig()
        assert cfg.phase1_threshold < cfg.phase2_threshold

    def test_default_values(self):
        cfg = AdaptationConfig()
        assert cfg.phase1_threshold == 5
        assert cfg.phase2_threshold == 50
        assert cfg.adapt_every_n_reviews > 0


class TestSchedulingConfig:
    def test_desired_retention_valid(self):
        cfg = SchedulingConfig()
        assert 0.0 < cfg.desired_retention < 1.0

    def test_interval_bounds(self):
        cfg = SchedulingConfig()
        assert cfg.min_interval >= 1
        assert cfg.max_interval > cfg.min_interval

    def test_fuzz_range_valid(self):
        cfg = SchedulingConfig()
        lo, hi = cfg.fuzz_range
        assert lo < hi and lo > 0


class TestMetaSRSConfig:
    def test_aggregate_has_all_sub_configs(self):
        cfg = MetaSRSConfig()
        assert isinstance(cfg.model, ModelConfig)
        assert isinstance(cfg.fsrs, FSRSConfig)
        assert isinstance(cfg.training, TrainingConfig)
        assert isinstance(cfg.adaptation, AdaptationConfig)
        assert isinstance(cfg.scheduling, SchedulingConfig)

    def test_w20_consistent(self):
        cfg = MetaSRSConfig()
        # w20 is used in loss; make sure it's accessible via cfg.fsrs
        assert cfg.fsrs.w[20] > 0
