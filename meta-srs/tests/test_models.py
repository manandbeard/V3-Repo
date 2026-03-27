"""Tests for GRU history encoder and card embedding projector."""

import sys
import os
import tempfile
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.gru_encoder import GRUHistoryEncoder
from models.card_embeddings import CardEmbeddingProjector, CardEmbeddingStore


class TestGRUHistoryEncoder:
    @pytest.fixture
    def encoder(self):
        return GRUHistoryEncoder(hidden_dim=32, max_len=32)

    def test_output_shape(self, encoder):
        grades = torch.tensor([[3, 2, 4, 0], [3, 0, 0, 0]], dtype=torch.float32)
        delta_ts = torch.tensor([[1.0, 3.0, 7.0, 0.0], [2.0, 0.0, 0.0, 0.0]])
        lengths = torch.tensor([3, 1], dtype=torch.long)
        out = encoder(grades, delta_ts, lengths)
        assert out.shape == (2, 32)

    def test_zero_state(self, encoder):
        z = encoder.zero_state(4, torch.device("cpu"))
        assert z.shape == (4, 32)
        assert (z == 0).all()

    def test_no_nan(self, encoder):
        grades = torch.rand(3, 10) * 4
        delta_ts = torch.rand(3, 10) * 30
        lengths = torch.tensor([10, 5, 1], dtype=torch.long)
        out = encoder(grades, delta_ts, lengths)
        assert not torch.isnan(out).any()

    def test_without_lengths(self, encoder):
        """Should work without explicit lengths."""
        grades = torch.rand(2, 5) * 4
        delta_ts = torch.rand(2, 5) * 10
        out = encoder(grades, delta_ts, lengths=None)
        assert out.shape == (2, 32)

    def test_truncation(self, encoder):
        """Sequences longer than max_len should be truncated."""
        grades = torch.rand(2, 50) * 4
        delta_ts = torch.rand(2, 50) * 10
        lengths = torch.tensor([50, 40], dtype=torch.long)
        out = encoder(grades, delta_ts, lengths)
        assert out.shape == (2, 32)


class TestCardEmbeddingProjector:
    @pytest.fixture
    def projector(self):
        return CardEmbeddingProjector(raw_dim=384, embed_dim=64)

    def test_output_shape(self, projector):
        raw = torch.randn(3, 384)
        out = projector(raw)
        assert out.shape == (3, 64)

    def test_l2_normalized(self, projector):
        """Output should be L2-normalized."""
        raw = torch.randn(3, 384)
        out = projector(raw)
        norms = out.norm(p=2, dim=-1)
        assert torch.allclose(norms, torch.ones(3), atol=1e-5)


class TestCardEmbeddingStore:
    def test_lookup_known_cards(self):
        np.random.seed(42)
        embeddings = {
            "c1": np.random.randn(384).astype(np.float32),
            "c2": np.random.randn(384).astype(np.float32),
        }
        store = CardEmbeddingStore(embeddings)
        result = store.lookup(["c1", "c2"], torch.device("cpu"))
        assert result.shape == (2, 384)

    def test_lookup_unknown_card_returns_zero(self):
        embeddings = {"c1": np.random.randn(384).astype(np.float32)}
        store = CardEmbeddingStore(embeddings)
        result = store.lookup(["c1", "unknown"], torch.device("cpu"))
        assert (result[1] == 0).all()

    def test_len(self):
        embeddings = {
            "c1": np.zeros(384, dtype=np.float32),
            "c2": np.zeros(384, dtype=np.float32),
        }
        store = CardEmbeddingStore(embeddings)
        assert len(store) == 2

    def test_save_and_load(self):
        np.random.seed(42)
        embeddings = {
            "c1": np.random.randn(384).astype(np.float32),
            "c2": np.random.randn(384).astype(np.float32),
        }
        store = CardEmbeddingStore(embeddings)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name

        try:
            store.save(path)
            loaded = CardEmbeddingStore.from_file(path)
            assert len(loaded) == 2
            assert np.allclose(loaded.embeddings["c1"], embeddings["c1"])
        finally:
            os.unlink(path)
