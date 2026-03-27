"""
Tests for Reptile meta-training utilities (training/reptile.py):
  - sample_batch
  - inner_loop
  - reptile_update
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import torch
from copy import deepcopy
from collections import OrderedDict

from training.reptile import sample_batch, inner_loop, reptile_update
from training.loss import MetaSRSLoss
from models.memory_net import MemoryNet
from data.task_sampler import Review, Task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_review(card_id="A", grade=3, elapsed=1.0, i=0):
    return Review(
        card_id=card_id,
        timestamp=i * 1000,
        elapsed_days=elapsed,
        grade=grade,
        recalled=grade >= 2,
        S_prev=1.0 + i * 0.1,
        D_prev=5.0,
        R_at_review=0.9,
        S_target=2.0,
        D_target=5.0,
    )


def make_task_with_split(n_reviews=30, n_cards=5):
    reviews = [
        make_review(f"card_{i % n_cards}", grade=(i % 4) + 1, elapsed=float(i % 10 + 1), i=i)
        for i in range(n_reviews)
    ]
    card_embs = {
        f"card_{i}": np.random.randn(384).astype(np.float32)
        for i in range(n_cards)
    }
    task = Task(student_id="s0", reviews=reviews, card_embeddings=card_embs)
    task.split(support_ratio=0.7)
    return task


# ---------------------------------------------------------------------------
# sample_batch
# ---------------------------------------------------------------------------

class TestSampleBatch:
    @pytest.fixture
    def reviews(self):
        return [make_review(f"c{i}", i=i) for i in range(20)]

    def test_returns_tensor_dict(self, reviews):
        batch = sample_batch(reviews, size=8, card_embeddings={},
                             device=torch.device("cpu"))
        assert isinstance(batch, dict)
        assert "grade" in batch

    def test_exact_size_when_larger_pool(self, reviews):
        batch = sample_batch(reviews, size=8, card_embeddings={},
                             device=torch.device("cpu"))
        assert batch["grade"].shape[0] == 8

    def test_full_pool_when_smaller_than_size(self, reviews):
        # Request more than available
        batch = sample_batch(reviews[:5], size=20, card_embeddings={},
                             device=torch.device("cpu"))
        assert batch["grade"].shape[0] == 5

    def test_batch_on_correct_device(self, reviews):
        batch = sample_batch(reviews, size=5, card_embeddings={},
                             device=torch.device("cpu"))
        assert batch["D_prev"].device.type == "cpu"


# ---------------------------------------------------------------------------
# inner_loop
# ---------------------------------------------------------------------------

class TestInnerLoop:
    @pytest.fixture
    def model(self):
        torch.manual_seed(0)
        return MemoryNet()

    @pytest.fixture
    def loss_fn(self):
        return MetaSRSLoss()

    @pytest.fixture
    def task(self):
        np.random.seed(0)
        return make_task_with_split(n_reviews=30)

    def test_returns_state_dict(self, model, loss_fn, task):
        phi = deepcopy(model.state_dict())
        result = inner_loop(phi, model, task, loss_fn, k_steps=2)
        assert isinstance(result, OrderedDict)

    def test_state_dict_has_same_keys(self, model, loss_fn, task):
        phi = deepcopy(model.state_dict())
        original_keys = set(phi.keys())
        result = inner_loop(phi, model, task, loss_fn, k_steps=2)
        assert set(result.keys()) == original_keys

    def test_weights_change_after_adaptation(self, model, loss_fn, task):
        phi = deepcopy(model.state_dict())
        result = inner_loop(phi, model, task, loss_fn, k_steps=3,
                            inner_lr=0.05, batch_size=8)
        # At least one parameter should differ from phi
        any_changed = False
        for key in phi:
            if not torch.allclose(phi[key], result[key]):
                any_changed = True
                break
        assert any_changed, "Parameters should change after inner loop adaptation"

    def test_phi_not_modified_in_place(self, model, loss_fn, task):
        phi = deepcopy(model.state_dict())
        phi_copy = deepcopy(phi)
        inner_loop(phi, model, task, loss_fn, k_steps=3)
        for key in phi:
            assert torch.allclose(phi[key], phi_copy[key]), \
                f"phi was modified in-place for key {key}"

    def test_no_nan_in_result(self, model, loss_fn, task):
        phi = deepcopy(model.state_dict())
        result = inner_loop(phi, model, task, loss_fn, k_steps=3)
        for key, val in result.items():
            if val.is_floating_point():
                assert not torch.isnan(val).any(), f"NaN in {key}"


# ---------------------------------------------------------------------------
# reptile_update
# ---------------------------------------------------------------------------

class TestReptileUpdate:
    @pytest.fixture
    def phi(self):
        torch.manual_seed(10)
        return OrderedDict({
            "w1": torch.randn(4, 4),
            "b1": torch.randn(4),
        })

    def test_returns_new_ordered_dict(self, phi):
        W_i = deepcopy(phi)
        result = reptile_update(phi, [W_i], epsilon=0.1)
        assert isinstance(result, OrderedDict)

    def test_same_keys(self, phi):
        W_i = deepcopy(phi)
        result = reptile_update(phi, [W_i], epsilon=0.1)
        assert set(result.keys()) == set(phi.keys())

    def test_no_update_when_W_equals_phi(self, phi):
        # If W_i == phi, reptile_grad = 0 → result == phi
        W_i = deepcopy(phi)
        result = reptile_update(phi, [W_i], epsilon=0.5)
        for key in phi:
            assert torch.allclose(result[key], phi[key])

    def test_update_moves_toward_W_i(self, phi):
        W_i = deepcopy(phi)
        # Set W_i to all-ones, phi to all-zeros → result should move toward ones
        phi_zeros = OrderedDict({"w": torch.zeros(3, 3)})
        W_i_ones = deepcopy(phi_zeros)
        W_i_ones["w"] = torch.ones(3, 3)
        result = reptile_update(phi_zeros, [W_i_ones], epsilon=0.5)
        expected = torch.ones(3, 3) * 0.5
        assert torch.allclose(result["w"], expected)

    def test_update_averaged_over_tasks(self, phi):
        # Two tasks: one nudges +1, one nudges -1 → net effect is zero
        phi_zero = OrderedDict({"w": torch.zeros(2)})
        W_plus = OrderedDict({"w": torch.ones(2)})
        W_minus = OrderedDict({"w": -torch.ones(2)})
        result = reptile_update(phi_zero, [W_plus, W_minus], epsilon=0.5)
        assert torch.allclose(result["w"], torch.zeros(2), atol=1e-6)

    def test_epsilon_scales_update(self, phi):
        phi_zero = OrderedDict({"w": torch.zeros(3)})
        W_i = OrderedDict({"w": torch.ones(3)})
        r1 = reptile_update(phi_zero, [W_i], epsilon=0.1)
        r2 = reptile_update(phi_zero, [W_i], epsilon=0.5)
        # Larger epsilon should move phi further
        assert r2["w"].mean() > r1["w"].mean()

    def test_phi_not_modified_in_place(self, phi):
        phi_orig_vals = {k: v.clone() for k, v in phi.items()}
        W_i = deepcopy(phi)
        W_i["w1"] += 1.0
        reptile_update(phi, [W_i], epsilon=0.3)
        for key in phi:
            assert torch.allclose(phi[key], phi_orig_vals[key]), \
                "Original phi should not be modified in-place"
