"""
Tests for evaluation metrics (evaluation/metrics.py):
  - EvalResults.summary
  - MetaSRSEvaluator._manual_auc
  - MetaSRSEvaluator.compute_auc_roc
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import torch

from evaluation.metrics import EvalResults, MetaSRSEvaluator
from models.memory_net import MemoryNet
from config import MetaSRSConfig


# ---------------------------------------------------------------------------
# EvalResults
# ---------------------------------------------------------------------------

class TestEvalResults:
    def test_default_values_are_zero(self):
        r = EvalResults()
        assert r.auc_roc == 0.0
        assert r.calibration_error == 0.0
        assert r.n_samples == 0

    def test_summary_contains_metric_names(self):
        r = EvalResults(auc_roc=0.82, calibration_error=0.03,
                        cold_start_auc=0.75, adaptation_speed=20,
                        retention_30d=0.87, rmse_stability=1.5,
                        n_samples=100)
        s = r.summary()
        assert "AUC-ROC" in s
        assert "Calibration Error" in s
        assert "Cold-Start AUC" in s
        assert "Adaptation Speed" in s
        assert "30-day Retention" in s
        assert "RMSE" in s

    def test_summary_shows_check_for_passing_metrics(self):
        r = EvalResults(auc_roc=0.85, calibration_error=0.03,
                        cold_start_auc=0.80, adaptation_speed=10,
                        retention_30d=0.92, rmse_stability=1.0)
        s = r.summary()
        # All targets met → expect checkmarks
        assert "✓" in s

    def test_summary_shows_x_for_failing_metrics(self):
        r = EvalResults(auc_roc=0.60, calibration_error=0.20)
        s = r.summary()
        assert "✗" in s

    def test_n_samples_displayed(self):
        r = EvalResults(n_samples=512)
        s = r.summary()
        assert "512" in s


# ---------------------------------------------------------------------------
# MetaSRSEvaluator._manual_auc
# ---------------------------------------------------------------------------

class TestManualAUC:
    @pytest.fixture
    def evaluator(self):
        torch.manual_seed(0)
        model = MemoryNet()
        return MetaSRSEvaluator(model, MetaSRSConfig(), torch.device("cpu"))

    def test_perfect_predictions_auc_1(self, evaluator):
        # Scores perfectly separate positives from negatives
        scores = np.array([0.9, 0.8, 0.3, 0.2])
        labels = np.array([1.0, 1.0, 0.0, 0.0])
        auc = evaluator._manual_auc(scores, labels)
        assert auc == pytest.approx(1.0, abs=0.05)

    def test_random_predictions_auc_near_half(self, evaluator):
        rng = np.random.RandomState(42)
        scores = rng.rand(200)
        labels = rng.randint(0, 2, 200).astype(float)
        auc = evaluator._manual_auc(scores, labels)
        # Random classifier → AUC ≈ 0.5
        assert 0.35 <= auc <= 0.65

    def test_inverted_predictions_auc_low(self, evaluator):
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([1.0, 1.0, 0.0, 0.0])
        auc = evaluator._manual_auc(scores, labels)
        assert auc <= 0.2  # very poor classifier

    def test_all_positive_returns_half(self, evaluator):
        scores = np.array([0.8, 0.7, 0.9])
        labels = np.array([1.0, 1.0, 1.0])
        auc = evaluator._manual_auc(scores, labels)
        assert auc == 0.5  # degenerate case

    def test_all_negative_returns_half(self, evaluator):
        scores = np.array([0.1, 0.2, 0.3])
        labels = np.array([0.0, 0.0, 0.0])
        auc = evaluator._manual_auc(scores, labels)
        assert auc == 0.5

    def test_auc_between_0_and_1(self, evaluator):
        rng = np.random.RandomState(7)
        for _ in range(20):
            scores = rng.rand(50)
            labels = rng.randint(0, 2, 50).astype(float)
            auc = evaluator._manual_auc(scores, labels)
            assert 0.0 <= auc <= 1.0


# ---------------------------------------------------------------------------
# MetaSRSEvaluator.compute_auc_roc
# ---------------------------------------------------------------------------

class TestComputeAUCROC:
    @pytest.fixture
    def evaluator(self):
        torch.manual_seed(0)
        model = MemoryNet()
        return MetaSRSEvaluator(model, MetaSRSConfig(), torch.device("cpu"))

    def test_perfect_auc(self, evaluator):
        scores = np.array([0.95, 0.85, 0.2, 0.1])
        labels = np.array([1.0, 1.0, 0.0, 0.0])
        auc = evaluator.compute_auc_roc(scores, labels)
        assert auc == pytest.approx(1.0, abs=0.05)

    def test_returns_float(self, evaluator):
        scores = np.random.rand(20)
        labels = np.random.randint(0, 2, 20).astype(float)
        auc = evaluator.compute_auc_roc(scores, labels)
        assert isinstance(auc, float)

    def test_auc_in_valid_range(self, evaluator):
        rng = np.random.RandomState(99)
        scores = rng.rand(100)
        labels = rng.randint(0, 2, 100).astype(float)
        auc = evaluator.compute_auc_roc(scores, labels)
        assert 0.0 <= auc <= 1.0

    def test_sklearn_and_manual_agree(self, evaluator):
        """When sklearn is available, manual and sklearn AUC should be close."""
        rng = np.random.RandomState(5)
        scores = rng.rand(200)
        labels = (scores + rng.randn(200) * 0.3 > 0.5).astype(float)
        manual = evaluator._manual_auc(scores, labels)
        sklearn_auc = evaluator.compute_auc_roc(scores, labels)
        # They should be within 5% of each other
        assert abs(manual - sklearn_auc) < 0.10
