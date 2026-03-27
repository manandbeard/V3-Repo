"""Integration test: full train→eval pipeline with synthetic data."""

import sys
import os
import pytest
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from config import MetaSRSConfig
from models.memory_net import MemoryNet
from data.task_sampler import ReviewDataset, TaskSampler
from tests.conftest import create_model
from training.reptile import ReptileTrainer
from training.loss import MetaSRSLoss, compute_loss
from evaluation.metrics import MetaSRSEvaluator


class TestEndToEnd:
    @pytest.mark.timeout(90)
    def test_synthetic_pipeline(self):
        """Test the complete train→eval pipeline with minimal synthetic data."""
        config = MetaSRSConfig()
        config.training.checkpoint_dir = "/tmp/meta-srs-e2e-checkpoints"
        config.training.log_dir = "/tmp/meta-srs-e2e-logs"

        # Step 1: Generate synthetic data
        all_tasks = ReviewDataset.generate_synthetic(
            n_students=15, reviews_per_student=30, n_cards=10, seed=42
        )
        assert len(all_tasks) == 15

        # Step 2: Split into train/test
        train_tasks = all_tasks[:12]
        test_tasks = all_tasks[12:]
        for t in test_tasks:
            t.split(support_ratio=0.70)

        # Step 3: Create model
        model = create_model(config.model)
        assert model.count_parameters() > 0

        # Step 4: Create task sampler
        sampler = TaskSampler(train_tasks, min_reviews=10)
        assert len(sampler) > 0

        # Step 5: Run Reptile training (minimal iterations)
        trainer = ReptileTrainer(model, config)
        phi = trainer.train(sampler, n_iters=5)
        assert isinstance(phi, dict)

        # Step 6: Evaluate
        evaluator = MetaSRSEvaluator(model, config)
        results = evaluator.evaluate_on_tasks(phi, test_tasks, k_steps=2)
        assert results.n_samples > 0
        assert 0.0 <= results.auc_roc <= 1.0

    @pytest.mark.timeout(60)
    def test_model_forward_with_synthetic_batch(self):
        """Test that model can do a forward pass on synthetic data."""
        config = MetaSRSConfig()
        model = create_model(config.model)
        loss_fn = MetaSRSLoss(w20=config.fsrs.w[20])

        tasks = ReviewDataset.generate_synthetic(
            n_students=5, reviews_per_student=20, n_cards=10, seed=42
        )
        task = tasks[0]
        task.split(support_ratio=0.70)

        from data.task_sampler import reviews_to_batch
        batch = reviews_to_batch(task.support_set, task.card_embeddings)

        # Forward pass
        state = model(
            D_prev=batch["D_prev"],
            S_prev=batch["S_prev"],
            R_at_review=batch["R_at_review"],
            delta_t=batch["delta_t"],
            grade=batch["grade"],
            review_count=batch["review_count"],
            card_embedding_raw=batch["card_embedding_raw"],
            user_stats=batch["user_stats"],
            history_grades=batch["history_grades"],
            history_delta_ts=batch["history_delta_ts"],
            history_lengths=batch["history_lengths"],
        )

        assert not torch.isnan(state.p_recall).any()
        assert not torch.isnan(state.S_next).any()
        assert not torch.isnan(state.D_next).any()

        # Loss computation
        losses = compute_loss(model, batch, loss_fn)
        assert torch.isfinite(losses["total"])

    @pytest.mark.timeout(60)
    def test_checkpoint_save_and_load(self):
        """Test that checkpoints can be saved and loaded."""
        config = MetaSRSConfig()
        model = create_model(config.model)
        phi = deepcopy(model.state_dict())

        path = "/tmp/meta-srs-test-phi.pt"
        torch.save({"phi": phi, "iteration": 100, "config": config}, path)

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert "phi" in checkpoint
        assert checkpoint["iteration"] == 100

        # Load phi into model
        model.load_state_dict(checkpoint["phi"])
        # Verify model works after loading
        features = torch.randn(2, 113)
        S_prev = torch.tensor([5.0, 10.0])
        state = model.forward_from_features(features, S_prev)
        assert state.p_recall.shape == (2,)

        os.unlink(path)
