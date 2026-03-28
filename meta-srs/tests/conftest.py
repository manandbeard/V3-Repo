"""Shared fixtures for MetaSRS tests."""

import sys
import os
import torch
import numpy as np
import pytest

# Add meta-srs root to sys.path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import MetaSRSConfig, ModelConfig, TrainingConfig, FSRSConfig
from models.memory_net import MemoryNet
from data.task_sampler import Review, Task, ReviewDataset
from training.fsrs_warmstart import FSRS6


def create_model(config=None):
    """Create a MemoryNet from ModelConfig, filtering out config-only fields."""
    if config is None:
        config = ModelConfig()
    init_args = {k: v for k, v in config.__dict__.items()
                 if k in MemoryNet.__init__.__code__.co_varnames}
    return MemoryNet(**init_args)


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def config():
    return MetaSRSConfig()


@pytest.fixture
def model_config():
    return ModelConfig()


@pytest.fixture
def training_config():
    return TrainingConfig()


@pytest.fixture
def fsrs_config():
    return FSRSConfig()


@pytest.fixture
def model(model_config):
    return create_model(model_config)


@pytest.fixture
def fsrs():
    return FSRS6()


@pytest.fixture
def sample_batch_tensors(device):
    """Create a small batch of tensors suitable for MemoryNet forward."""
    batch_size = 4
    return {
        "D_prev": torch.tensor([5.0, 3.0, 7.0, 2.0], device=device),
        "S_prev": torch.tensor([10.0, 5.0, 20.0, 1.0], device=device),
        "R_at_review": torch.tensor([0.9, 0.7, 0.5, 1.0], device=device),
        "delta_t": torch.tensor([3.0, 7.0, 14.0, 0.0], device=device),
        "grade": torch.tensor([3, 2, 4, 1], dtype=torch.long, device=device),
        "review_count": torch.tensor([5.0, 3.0, 10.0, 1.0], device=device),
        "user_stats": torch.zeros(batch_size, 8, device=device),
        "history_grades": torch.tensor(
            [[3, 2, 0, 0], [4, 0, 0, 0], [3, 3, 2, 4], [0, 0, 0, 0]],
            dtype=torch.float32, device=device,
        ),
        "history_delta_ts": torch.tensor(
            [[1.0, 3.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0],
             [1.0, 3.0, 7.0, 14.0], [0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32, device=device,
        ),
        "history_lengths": torch.tensor([2, 1, 4, 1], dtype=torch.long, device=device),
        "recalled": torch.tensor([1.0, 1.0, 1.0, 0.0], device=device),
        "S_target": torch.tensor([15.0, 8.0, 30.0, 0.5], device=device),
        "D_target": torch.tensor([4.5, 3.2, 6.8, 2.5], device=device),
    }


@pytest.fixture
def sample_reviews():
    """Create a list of sample Review objects."""
    return [
        Review(card_id="c1", timestamp=0, elapsed_days=0.0, grade=3,
               recalled=True, S_prev=3.0, D_prev=5.0, R_at_review=1.0,
               S_target=5.0, D_target=4.8),
        Review(card_id="c2", timestamp=86400, elapsed_days=1.0, grade=2,
               recalled=True, S_prev=1.0, D_prev=7.0, R_at_review=0.9,
               S_target=2.0, D_target=7.2),
        Review(card_id="c1", timestamp=172800, elapsed_days=2.0, grade=4,
               recalled=True, S_prev=5.0, D_prev=4.8, R_at_review=0.85,
               S_target=15.0, D_target=4.0),
        Review(card_id="c3", timestamp=259200, elapsed_days=0.0, grade=1,
               recalled=False, S_prev=1.0, D_prev=5.0, R_at_review=1.0,
               S_target=0.5, D_target=6.0),
    ]


@pytest.fixture
def synthetic_tasks():
    """Generate a small set of synthetic tasks for testing."""
    return ReviewDataset.generate_synthetic(
        n_students=20, reviews_per_student=50, n_cards=30, seed=42
    )
