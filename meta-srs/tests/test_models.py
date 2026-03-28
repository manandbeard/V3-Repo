"""Tests for GRU history encoder."""

import sys
import os
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.gru_encoder import GRUHistoryEncoder


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
