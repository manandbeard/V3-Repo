"""
Shared pytest fixtures for MetaSRS test suite.
"""

import sys
import os

# Ensure the meta-srs package root is on sys.path so tests can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import torch

from config import MetaSRSConfig, ModelConfig, FSRSConfig, TrainingConfig
from data.task_sampler import Review, Task, reviews_to_batch
from models.memory_net import MemoryNet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_review(
    card_id="card_0001",
    grade=3,
    elapsed_days=1.0,
    S_prev=1.0,
    D_prev=5.0,
    timestamp=0,
):
    """Build a minimal Review fixture."""
    return Review(
        card_id=card_id,
        timestamp=timestamp,
        elapsed_days=elapsed_days,
        grade=grade,
        recalled=grade >= 2,
        S_prev=S_prev,
        D_prev=D_prev,
        R_at_review=0.9,
        S_target=S_prev * 2.0,
        D_target=D_prev,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def default_config():
    return MetaSRSConfig()


@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")


@pytest.fixture(scope="session")
def model(default_config):
    """A freshly initialised MemoryNet (CPU, deterministic)."""
    torch.manual_seed(42)
    cfg = default_config.model
    return MemoryNet(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        card_embed_dim=cfg.card_embed_dim,
        card_raw_dim=cfg.card_raw_dim,
        gru_hidden_dim=cfg.gru_hidden_dim,
        user_stats_dim=cfg.user_stats_dim,
        dropout=cfg.dropout,
        history_len=cfg.history_len,
    )


@pytest.fixture(scope="session")
def phi_star(model):
    """A copy of the model's initial state dict (serves as phi*)."""
    from copy import deepcopy
    return deepcopy(model.state_dict())


@pytest.fixture
def synthetic_reviews():
    """10 synthetic reviews across 3 cards."""
    reviews = [
        make_review("card_A", grade=3, elapsed_days=1.0, S_prev=1.0, timestamp=i * 86400)
        for i in range(4)
    ] + [
        make_review("card_B", grade=2, elapsed_days=2.0, S_prev=2.0, timestamp=i * 86400)
        for i in range(3)
    ] + [
        make_review("card_C", grade=1, elapsed_days=0.5, S_prev=0.5, timestamp=i * 86400)
        for i in range(3)
    ]
    return reviews


@pytest.fixture
def card_embeddings():
    """Random 384-dim embeddings for a few card IDs."""
    np.random.seed(0)
    return {
        "card_A": np.random.randn(384).astype(np.float32),
        "card_B": np.random.randn(384).astype(np.float32),
        "card_C": np.random.randn(384).astype(np.float32),
    }


@pytest.fixture
def batch(synthetic_reviews, card_embeddings, device):
    """A pre-built tensor batch from the synthetic reviews."""
    return reviews_to_batch(synthetic_reviews, card_embeddings, device)


@pytest.fixture
def synthetic_task(synthetic_reviews, card_embeddings):
    """A Task built from synthetic reviews, with support/query split."""
    task = Task(
        student_id="student_test",
        reviews=synthetic_reviews,
        card_embeddings=card_embeddings,
    )
    task.split(support_ratio=0.7)
    return task
