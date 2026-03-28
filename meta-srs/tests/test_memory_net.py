"""Tests for MemoryNet model."""

import sys
import os
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import ModelConfig
from models.memory_net import MemoryNet, MemoryState


class TestMemoryNetArchitecture:
    def test_creation(self, model):
        assert isinstance(model, MemoryNet)

    def test_parameter_count(self, model):
        n_params = model.count_parameters()
        # Should be around 50K–70K as per architecture
        assert 30_000 < n_params < 150_000, f"Unexpected param count: {n_params}"

    def test_input_dim(self, model):
        assert model.input_dim == 113

    def test_submodules_exist(self, model):
        assert hasattr(model, "card_projector")
        assert hasattr(model, "gru_encoder")
        assert hasattr(model, "stability_head")
        assert hasattr(model, "difficulty_head")
        assert hasattr(model, "recall_head")


class TestMemoryNetForward:
    def test_forward_output_shapes(self, model, sample_batch_tensors, device):
        batch = sample_batch_tensors
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
        assert isinstance(state, MemoryState)
        assert state.S_next.shape == (4,)
        assert state.D_next.shape == (4,)
        assert state.p_recall.shape == (4,)

    def test_output_ranges(self, model, sample_batch_tensors, device):
        batch = sample_batch_tensors
        model.eval()
        with torch.no_grad():
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

        # S_next clamped to [0.001, 36500]
        assert (state.S_next >= 1e-3).all()
        assert (state.S_next <= 36500.0).all()
        # D_next in [1, 10]
        assert (state.D_next >= 1.0).all()
        assert (state.D_next <= 10.0).all()
        # p_recall in [0, 1]
        assert (state.p_recall >= 0.0).all()
        assert (state.p_recall <= 1.0).all()

    def test_no_nan_in_output(self, model, sample_batch_tensors, device):
        batch = sample_batch_tensors
        model.eval()
        with torch.no_grad():
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
        assert not torch.isnan(state.S_next).any()
        assert not torch.isnan(state.D_next).any()
        assert not torch.isnan(state.p_recall).any()


class TestBuildFeatures:
    def test_feature_dim(self, model, sample_batch_tensors, device):
        batch = sample_batch_tensors
        features = model.build_features(
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
        assert features.shape == (4, 113)

    def test_no_history_uses_zero_context(self, model, sample_batch_tensors, device):
        """When history is None, GRU should produce zero context."""
        batch = sample_batch_tensors
        features = model.build_features(
            D_prev=batch["D_prev"],
            S_prev=batch["S_prev"],
            R_at_review=batch["R_at_review"],
            delta_t=batch["delta_t"],
            grade=batch["grade"],
            review_count=batch["review_count"],
            card_embedding_raw=batch["card_embedding_raw"],
            user_stats=batch["user_stats"],
            history_grades=None,
            history_delta_ts=None,
            history_lengths=None,
        )
        assert features.shape == (4, 113)
        # Last 32 dims (GRU context) should be zero
        assert (features[:, -32:] == 0).all()


class TestForwardFromFeatures:
    def test_forward_from_features(self, model, device):
        features = torch.randn(4, 113, device=device)
        S_prev = torch.tensor([5.0, 10.0, 1.0, 20.0], device=device)
        state = model.forward_from_features(features, S_prev)
        assert state.S_next.shape == (4,)
        assert state.D_next.shape == (4,)
        assert state.p_recall.shape == (4,)

    def test_predict_recall(self, model, device):
        features = torch.randn(4, 113, device=device)
        S_prev = torch.tensor([5.0, 10.0, 1.0, 20.0], device=device)
        p = model.predict_recall(features, S_prev)
        assert p.shape == (4,)
        assert (p >= 0).all() and (p <= 1).all()

    def test_predict_stability(self, model, device):
        features = torch.randn(4, 113, device=device)
        S_prev = torch.tensor([5.0, 10.0, 1.0, 20.0], device=device)
        S = model.predict_stability(features, S_prev)
        assert S.shape == (4,)
        assert (S >= 1e-3).all()


class TestGradientFlow:
    def test_backward_pass(self, model, sample_batch_tensors, device):
        """Ensure gradients flow through the full forward pass."""
        batch = sample_batch_tensors
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
        loss = state.p_recall.sum()
        loss.backward()

        # Check at least some parameters have gradients
        grads_found = 0
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                grads_found += 1
        assert grads_found > 0, "No gradients found after backward pass"
