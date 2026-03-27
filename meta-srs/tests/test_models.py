"""
Tests for neural model components:
  - CardEmbeddingProjector
  - CardEmbeddingStore
  - GRUHistoryEncoder
  - MemoryNet
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import torch
import torch.nn.functional as F

from models.card_embeddings import CardEmbeddingProjector, CardEmbeddingStore
from models.gru_encoder import GRUHistoryEncoder
from models.memory_net import MemoryNet, MemoryState


# ---------------------------------------------------------------------------
# CardEmbeddingProjector
# ---------------------------------------------------------------------------

class TestCardEmbeddingProjector:
    @pytest.fixture
    def projector(self):
        torch.manual_seed(0)
        return CardEmbeddingProjector(raw_dim=384, embed_dim=64)

    def test_output_shape(self, projector):
        batch = torch.randn(8, 384)
        out = projector(batch)
        assert out.shape == (8, 64)

    def test_output_is_l2_normalized(self, projector):
        batch = torch.randn(8, 384)
        out = projector(batch)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(8), atol=1e-5)

    def test_single_sample(self, projector):
        x = torch.randn(1, 384)
        out = projector(x)
        assert out.shape == (1, 64)

    def test_different_inputs_different_outputs(self, projector):
        x1 = torch.randn(4, 384)
        x2 = torch.randn(4, 384)
        o1 = projector(x1)
        o2 = projector(x2)
        assert not torch.allclose(o1, o2)


# ---------------------------------------------------------------------------
# CardEmbeddingStore
# ---------------------------------------------------------------------------

class TestCardEmbeddingStore:
    @pytest.fixture
    def store(self):
        np.random.seed(1)
        data = {
            "card_a": np.random.randn(384).astype(np.float32),
            "card_b": np.random.randn(384).astype(np.float32),
        }
        return CardEmbeddingStore(data)

    def test_len(self, store):
        assert len(store) == 2

    def test_lookup_known_card(self, store):
        result = store.lookup(["card_a"], device=torch.device("cpu"))
        assert result.shape == (1, 384)

    def test_lookup_batch(self, store):
        result = store.lookup(["card_a", "card_b"], device=torch.device("cpu"))
        assert result.shape == (2, 384)

    def test_lookup_unknown_card_returns_zeros(self, store):
        result = store.lookup(["unknown_card"], device=torch.device("cpu"))
        assert result.shape == (1, 384)
        assert torch.allclose(result, torch.zeros(1, 384))

    def test_lookup_mixed_known_unknown(self, store):
        result = store.lookup(["card_a", "missing"], device=torch.device("cpu"))
        assert result.shape == (2, 384)
        # Second row is zeros
        assert torch.allclose(result[1], torch.zeros(384))

    def test_save_and_load(self, store, tmp_path):
        path = str(tmp_path / "embeddings.npz")
        store.save(path)
        loaded = CardEmbeddingStore.from_file(path)
        assert len(loaded) == len(store)
        for key in ["card_a", "card_b"]:
            original = store.lookup([key], torch.device("cpu"))
            restored = loaded.lookup([key], torch.device("cpu"))
            assert torch.allclose(original, restored, atol=1e-6)

    def test_empty_store_default_dim(self):
        empty = CardEmbeddingStore({})
        assert empty._dim == 384


# ---------------------------------------------------------------------------
# GRUHistoryEncoder
# ---------------------------------------------------------------------------

class TestGRUHistoryEncoder:
    @pytest.fixture
    def encoder(self):
        torch.manual_seed(2)
        return GRUHistoryEncoder(hidden_dim=32, max_len=16)

    def test_output_shape_without_lengths(self, encoder):
        grades = torch.rand(4, 8)
        delta_ts = torch.rand(4, 8)
        out = encoder(grades, delta_ts)
        assert out.shape == (4, 32)

    def test_output_shape_with_lengths(self, encoder):
        grades = torch.rand(4, 8)
        delta_ts = torch.rand(4, 8)
        lengths = torch.tensor([8, 6, 4, 2])
        out = encoder(grades, delta_ts, lengths)
        assert out.shape == (4, 32)

    def test_zero_state_shape(self, encoder):
        z = encoder.zero_state(batch_size=5, device=torch.device("cpu"))
        assert z.shape == (5, 32)
        assert torch.all(z == 0)

    def test_sequence_truncated_to_max_len(self, encoder):
        # Provide more history than max_len=16; should truncate silently
        grades = torch.rand(2, 30)
        delta_ts = torch.rand(2, 30)
        lengths = torch.tensor([30, 25])
        out = encoder(grades, delta_ts, lengths)
        assert out.shape == (2, 32)

    def test_different_histories_different_outputs(self, encoder):
        torch.manual_seed(99)
        g1 = torch.rand(2, 5)
        g2 = torch.rand(2, 5) + 2.0
        dt = torch.ones(2, 5)
        o1 = encoder(g1, dt)
        o2 = encoder(g2, dt)
        assert not torch.allclose(o1, o2)

    def test_single_step_sequence(self, encoder):
        grades = torch.rand(1, 1)
        delta_ts = torch.rand(1, 1)
        lengths = torch.tensor([1])
        out = encoder(grades, delta_ts, lengths)
        assert out.shape == (1, 32)


# ---------------------------------------------------------------------------
# MemoryNet
# ---------------------------------------------------------------------------

BATCH = 8
RAW_DIM = 384
USER_STATS_DIM = 8


def _make_inputs(batch_size=BATCH, with_history=False):
    torch.manual_seed(7)
    D_prev = torch.rand(batch_size) * 9.0 + 1.0
    S_prev = torch.rand(batch_size) * 29.0 + 1.0
    R_at_review = torch.rand(batch_size)
    delta_t = torch.rand(batch_size) * 14.0
    grade = torch.randint(1, 5, (batch_size,))
    review_count = torch.randint(1, 50, (batch_size,)).float()
    card_embed_raw = torch.randn(batch_size, RAW_DIM)
    user_stats = torch.zeros(batch_size, USER_STATS_DIM)

    if with_history:
        history_grades = torch.rand(batch_size, 10)
        history_delta_ts = torch.rand(batch_size, 10) * 10.0
        history_lengths = torch.randint(1, 11, (batch_size,))
        return (D_prev, S_prev, R_at_review, delta_t, grade, review_count,
                card_embed_raw, user_stats, history_grades, history_delta_ts, history_lengths)

    return (D_prev, S_prev, R_at_review, delta_t, grade, review_count,
            card_embed_raw, user_stats, None, None, None)


class TestMemoryNet:
    @pytest.fixture
    def net(self):
        torch.manual_seed(3)
        return MemoryNet()  # all defaults → input_dim=113

    def test_forward_output_types(self, net):
        inputs = _make_inputs()
        state = net.forward(*inputs)
        assert isinstance(state, MemoryState)

    def test_forward_shape(self, net):
        inputs = _make_inputs()
        state = net.forward(*inputs)
        assert state.S_next.shape == (BATCH,)
        assert state.D_next.shape == (BATCH,)
        assert state.p_recall.shape == (BATCH,)

    def test_forward_with_history(self, net):
        inputs = _make_inputs(with_history=True)
        state = net.forward(*inputs)
        assert state.p_recall.shape == (BATCH,)

    def test_S_next_clamped(self, net):
        inputs = _make_inputs()
        state = net.forward(*inputs)
        assert state.S_next.min().item() >= 1e-3
        assert state.S_next.max().item() <= 36500.0

    def test_D_next_in_range(self, net):
        inputs = _make_inputs()
        state = net.forward(*inputs)
        assert state.D_next.min().item() >= 1.0
        assert state.D_next.max().item() <= 10.0

    def test_p_recall_in_range(self, net):
        inputs = _make_inputs()
        state = net.forward(*inputs)
        assert state.p_recall.min().item() >= 0.0
        assert state.p_recall.max().item() <= 1.0

    def test_parameter_count_approx_50k(self, net):
        n = net.count_parameters()
        # Target: ~50K; allow a generous range
        assert 20_000 < n < 200_000, f"Unexpected param count: {n}"

    def test_build_features_shape(self, net):
        (D_prev, S_prev, R_at_review, delta_t, grade, review_count,
         card_embed_raw, user_stats, _, _, _) = _make_inputs()
        x = net.build_features(
            D_prev, S_prev, R_at_review, delta_t, grade,
            review_count, card_embed_raw, user_stats
        )
        assert x.shape == (BATCH, 113)

    def test_build_features_no_nan(self, net):
        inputs = _make_inputs()
        x = net.build_features(*inputs[:8])
        assert not torch.isnan(x).any()

    def test_forward_from_features(self, net):
        (D_prev, S_prev, R_at_review, delta_t, grade, review_count,
         card_embed_raw, user_stats, _, _, _) = _make_inputs()
        x = net.build_features(D_prev, S_prev, R_at_review, delta_t,
                               grade, review_count, card_embed_raw, user_stats)
        state = net.forward_from_features(x, S_prev)
        assert state.p_recall.shape == (BATCH,)

    def test_predict_recall_convenience(self, net):
        (D_prev, S_prev, R_at_review, delta_t, grade, review_count,
         card_embed_raw, user_stats, _, _, _) = _make_inputs()
        x = net.build_features(D_prev, S_prev, R_at_review, delta_t,
                               grade, review_count, card_embed_raw, user_stats)
        p = net.predict_recall(x, S_prev)
        assert p.shape == (BATCH,)
        assert p.min().item() >= 0.0

    def test_predict_stability_convenience(self, net):
        (D_prev, S_prev, R_at_review, delta_t, grade, review_count,
         card_embed_raw, user_stats, _, _, _) = _make_inputs()
        x = net.build_features(D_prev, S_prev, R_at_review, delta_t,
                               grade, review_count, card_embed_raw, user_stats)
        S = net.predict_stability(x, S_prev)
        assert S.shape == (BATCH,)
        assert S.min().item() > 0

    def test_padding_safety_net_short_features(self, net):
        # Build a feature shorter than input_dim and check padding
        batch_size = 3
        short_feat = torch.randn(batch_size, 100)  # shorter than 113
        S_prev = torch.ones(batch_size) * 5.0
        # Manually trigger the padding path
        if short_feat.size(-1) < net.input_dim:
            pad = torch.zeros(batch_size, net.input_dim - short_feat.size(-1))
            x = torch.cat([short_feat, pad], dim=-1)
        else:
            x = short_feat
        state = net.forward_from_features(x, S_prev)
        assert state.p_recall.shape == (batch_size,)

    def test_no_nan_in_outputs(self, net):
        inputs = _make_inputs()
        state = net.forward(*inputs)
        assert not torch.isnan(state.S_next).any()
        assert not torch.isnan(state.D_next).any()
        assert not torch.isnan(state.p_recall).any()

    def test_grad_flows_through_forward(self, net):
        (D_prev, S_prev, R_at_review, delta_t, grade, review_count,
         card_embed_raw, user_stats, _, _, _) = _make_inputs()
        net.train()
        state = net.forward(D_prev, S_prev, R_at_review, delta_t, grade,
                            review_count, card_embed_raw, user_stats)
        loss = state.p_recall.sum()
        loss.backward()
        # At least one parameter should have a gradient
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in net.parameters())
        assert has_grad
