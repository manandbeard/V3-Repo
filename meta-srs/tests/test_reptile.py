"""Tests for Reptile meta-training."""

import sys
import os
import torch
import pytest
from collections import OrderedDict
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import MetaSRSConfig
from models.memory_net import MemoryNet
from training.reptile import inner_loop, reptile_update, sample_batch, ReptileTrainer
from tests.conftest import create_model
from training.loss import MetaSRSLoss
from data.task_sampler import Task, Review, TaskSampler, ReviewDataset


@pytest.fixture
def phi_and_model():
    """Create a model and its initial phi state dict."""
    config = MetaSRSConfig()
    model = create_model(config.model)
    phi = deepcopy(model.state_dict())
    return phi, model, config


@pytest.fixture
def simple_task():
    """Create a simple task for testing inner loop."""
    import numpy as np
    np.random.seed(42)
    card_embeddings = {
        f"c{i}": np.random.randn(384).astype(np.float32) for i in range(5)
    }
    reviews = [
        Review(card_id=f"c{i%5}", timestamp=i * 86400,
               elapsed_days=float(i), grade=3, recalled=True,
               S_prev=3.0 + i, D_prev=5.0, R_at_review=0.9,
               S_target=5.0 + i, D_target=4.8)
        for i in range(30)
    ]
    task = Task(student_id="test_student", reviews=reviews,
                card_embeddings=card_embeddings)
    task.split(support_ratio=0.70)
    return task


class TestSampleBatch:
    def test_returns_dict(self, simple_task):
        batch = sample_batch(
            simple_task.support_set, 8,
            simple_task.card_embeddings, torch.device("cpu")
        )
        assert isinstance(batch, dict)
        assert "D_prev" in batch
        assert "recalled" in batch

    def test_respects_size(self, simple_task):
        batch = sample_batch(
            simple_task.support_set, 5,
            simple_task.card_embeddings, torch.device("cpu")
        )
        assert batch["D_prev"].shape[0] <= max(5, len(simple_task.support_set))


class TestInnerLoop:
    def test_inner_loop_changes_weights(self, phi_and_model, simple_task):
        phi, model, config = phi_and_model
        loss_fn = MetaSRSLoss(w20=config.fsrs.w[20])

        adapted = inner_loop(
            phi_state_dict=phi,
            model=model,
            task=simple_task,
            loss_fn=loss_fn,
            k_steps=3,
            inner_lr=0.01,
            batch_size=16,
            device=torch.device("cpu"),
        )

        # Adapted weights should differ from phi
        different = False
        for key in phi:
            if not torch.equal(phi[key], adapted[key]):
                different = True
                break
        assert different, "Inner loop did not change weights"

    def test_inner_loop_does_not_modify_phi(self, phi_and_model, simple_task):
        phi, model, config = phi_and_model
        phi_copy = deepcopy(phi)
        loss_fn = MetaSRSLoss(w20=config.fsrs.w[20])

        inner_loop(
            phi_state_dict=phi,
            model=model,
            task=simple_task,
            loss_fn=loss_fn,
            k_steps=3,
            inner_lr=0.01,
            batch_size=16,
            device=torch.device("cpu"),
        )

        # phi should be unchanged
        for key in phi:
            assert torch.equal(phi[key], phi_copy[key]), \
                f"phi was modified at key {key}"


class TestReptileUpdate:
    def test_update_moves_phi(self, phi_and_model):
        phi, model, _ = phi_and_model
        # Create two "adapted" weight sets that differ from phi
        adapted1 = OrderedDict()
        adapted2 = OrderedDict()
        for key in phi:
            adapted1[key] = phi[key] + 0.1 * torch.randn_like(phi[key])
            adapted2[key] = phi[key] + 0.1 * torch.randn_like(phi[key])

        new_phi = reptile_update(phi, [adapted1, adapted2], epsilon=0.1)

        # New phi should differ from old phi
        different = False
        for key in phi:
            if not torch.equal(phi[key], new_phi[key]):
                different = True
                break
        assert different, "Reptile update did not change phi"

    def test_update_direction(self, phi_and_model):
        """Phi should move toward the adapted weights."""
        phi, model, _ = phi_and_model
        # All adapted weights are the same: phi + delta
        delta = OrderedDict()
        adapted = OrderedDict()
        for key in phi:
            delta[key] = torch.ones_like(phi[key])
            adapted[key] = phi[key] + delta[key]

        new_phi = reptile_update(phi, [adapted], epsilon=0.5)

        # new_phi should be phi + 0.5 * delta
        for key in phi:
            expected = phi[key] + 0.5 * delta[key]
            assert torch.allclose(new_phi[key], expected, atol=1e-6), \
                f"Unexpected update at key {key}"

    def test_epsilon_zero_no_change(self, phi_and_model):
        """With epsilon=0, phi should not change."""
        phi, model, _ = phi_and_model
        adapted = OrderedDict()
        for key in phi:
            adapted[key] = phi[key] + torch.randn_like(phi[key])

        new_phi = reptile_update(phi, [adapted], epsilon=0.0)
        for key in phi:
            assert torch.equal(phi[key], new_phi[key])


class TestReptileTrainer:
    @pytest.mark.timeout(60)
    def test_trainer_runs(self):
        """Test that ReptileTrainer runs for a few iterations without crashing."""
        config = MetaSRSConfig()
        config.training.log_dir = "/tmp/meta-srs-test-logs"
        config.training.checkpoint_dir = "/tmp/meta-srs-test-checkpoints"
        model = create_model(config.model)
        trainer = ReptileTrainer(model, config)

        tasks = ReviewDataset.generate_synthetic(
            n_students=10, reviews_per_student=30, n_cards=10, seed=42
        )
        sampler = TaskSampler(tasks, min_reviews=10)

        phi = trainer.train(sampler, n_iters=3)
        assert isinstance(phi, OrderedDict)
        assert len(phi) > 0
