"""Tests for the evaluation framework."""

import sys
import os
import numpy as np
import torch
import pytest
from collections import OrderedDict
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import MetaSRSConfig
from models.memory_net import MemoryNet
from evaluation.metrics import MetaSRSEvaluator, EvalResults
from tests.conftest import create_model
from data.task_sampler import ReviewDataset, TaskSampler


class TestEvalResults:
    def test_default_values(self):
        results = EvalResults()
        assert results.auc_roc == 0.0
        assert results.n_samples == 0

    def test_summary_string(self):
        results = EvalResults(auc_roc=0.85, calibration_error=0.03,
                              cold_start_auc=0.75, n_samples=100)
        s = results.summary()
        assert "AUC-ROC" in s
        assert "0.8500" in s
        assert "✓" in s  # AUC > 0.80 should pass


class TestAUCComputation:
    @pytest.fixture
    def evaluator(self):
        config = MetaSRSConfig()
        model = create_model(config.model)
        return MetaSRSEvaluator(model, config)

    def test_perfect_auc(self, evaluator):
        preds = np.array([0.9, 0.8, 0.2, 0.1])
        labels = np.array([1, 1, 0, 0])
        auc = evaluator.compute_auc_roc(preds, labels)
        assert auc == 1.0

    def test_random_auc(self, evaluator):
        """Random predictions should give AUC ≈ 0.5."""
        np.random.seed(42)
        preds = np.random.rand(1000)
        labels = (np.random.rand(1000) > 0.5).astype(float)
        auc = evaluator.compute_auc_roc(preds, labels)
        assert 0.4 < auc < 0.6

    def test_auc_with_all_same_label(self, evaluator):
        """AUC should handle single-class labels gracefully (NaN from sklearn)."""
        preds = np.array([0.9, 0.8, 0.7])
        labels = np.array([1, 1, 1])
        auc = evaluator.compute_auc_roc(preds, labels)
        # sklearn returns NaN for single-class; manual returns 0.5
        assert np.isnan(auc) or auc == 0.5

    def test_manual_auc_implementation(self, evaluator):
        """Test the fallback manual AUC implementation."""
        preds = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
        labels = np.array([1, 1, 1, 0, 0])
        auc = evaluator._manual_auc(preds, labels)
        assert auc == 1.0


class TestEvaluateOnTasks:
    @pytest.mark.timeout(60)
    def test_evaluate_returns_results(self):
        config = MetaSRSConfig()
        model = create_model(config.model)
        evaluator = MetaSRSEvaluator(model, config)

        tasks = ReviewDataset.generate_synthetic(
            n_students=10, reviews_per_student=30, n_cards=10, seed=42
        )
        # Split tasks
        for task in tasks:
            task.split(support_ratio=0.70)

        phi = deepcopy(model.state_dict())
        results = evaluator.evaluate_on_tasks(phi, tasks[:5], k_steps=2)

        assert isinstance(results, EvalResults)
        assert results.n_samples > 0
        assert 0.0 <= results.auc_roc <= 1.0

    @pytest.mark.timeout(60)
    def test_cold_start_curve(self):
        config = MetaSRSConfig()
        model = create_model(config.model)
        evaluator = MetaSRSEvaluator(model, config)

        tasks = ReviewDataset.generate_synthetic(
            n_students=10, reviews_per_student=30, n_cards=10, seed=42
        )
        for task in tasks:
            task.split(support_ratio=0.70)

        phi = deepcopy(model.state_dict())
        curve = evaluator.cold_start_curve(phi, tasks[:3],
                                           review_counts=[0, 5])

        assert isinstance(curve, dict)
        assert 0 in curve
        assert 0.0 <= curve[0] <= 1.0
