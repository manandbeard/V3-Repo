"""
Tests for data layer: Review, Task, reviews_to_batch, TaskSampler,
and ReviewDataset.generate_synthetic (data/task_sampler.py).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import io
import csv
import tempfile
import pytest
import numpy as np
import torch

from data.task_sampler import (
    Review, Task, reviews_to_batch, TaskSampler, ReviewDataset
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_review(card_id="A", grade=3, elapsed=1.0, timestamp=0):
    return Review(
        card_id=card_id,
        timestamp=timestamp,
        elapsed_days=elapsed,
        grade=grade,
        recalled=grade >= 2,
        S_prev=1.0,
        D_prev=5.0,
        R_at_review=0.9,
        S_target=2.0,
        D_target=5.0,
    )


def make_task(n_reviews=20, n_cards=5, support_ratio=0.7):
    reviews = [
        make_review(f"card_{i % n_cards}", grade=(i % 4) + 1, timestamp=i * 1000)
        for i in range(n_reviews)
    ]
    task = Task(
        student_id="student_0",
        reviews=reviews,
        card_embeddings={
            f"card_{i}": np.random.randn(384).astype(np.float32)
            for i in range(n_cards)
        },
    )
    task.split(support_ratio)
    return task


# ---------------------------------------------------------------------------
# Review dataclass
# ---------------------------------------------------------------------------

class TestReview:
    def test_recalled_true_when_grade_ge_2(self):
        for grade in [2, 3, 4]:
            r = make_review(grade=grade)
            assert r.recalled is True

    def test_recalled_false_when_grade_1(self):
        r = make_review(grade=1)
        assert r.recalled is False

    def test_default_values(self):
        r = Review("x", 0, 0.0, 3, True)
        assert r.S_prev == 1.0
        assert r.D_prev == 5.0


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

class TestTask:
    def test_split_support_ratio(self):
        task = make_task(n_reviews=10, support_ratio=0.7)
        assert len(task.support_set) == 7
        assert len(task.query_set) == 3

    def test_split_total_preserved(self):
        task = make_task(n_reviews=15)
        assert len(task.support_set) + len(task.query_set) == 15

    def test_split_minimum_one_in_support(self):
        task = make_task(n_reviews=2, support_ratio=0.1)
        # max(1, int(2*0.1)) = max(1, 0) = 1
        assert len(task.support_set) >= 1

    def test_unique_cards_no_duplicates(self):
        task = make_task()
        seen = set()
        for cid in task.unique_cards:
            assert cid not in seen
            seen.add(cid)

    def test_get_review_history_empty_for_first(self):
        task = make_task(n_reviews=10)
        history = task.get_review_history("card_0", up_to_idx=0)
        assert history == []

    def test_get_review_history_returns_prior_reviews(self):
        reviews = [
            make_review("card_0", timestamp=i * 1000)
            for i in range(5)
        ]
        task = Task("s0", reviews)
        task.split(0.7)
        history = task.get_review_history("card_0", up_to_idx=5)
        assert len(history) == 5

    def test_get_review_history_only_specific_card(self):
        reviews = [
            make_review("card_A", timestamp=0),
            make_review("card_B", timestamp=1000),
            make_review("card_A", timestamp=2000),
        ]
        task = Task("s1", reviews)
        history = task.get_review_history("card_A", up_to_idx=3)
        assert all(r.card_id == "card_A" for r in history)
        assert len(history) == 2


# ---------------------------------------------------------------------------
# reviews_to_batch
# ---------------------------------------------------------------------------

class TestReviewsToBatch:
    @pytest.fixture
    def reviews(self):
        return [make_review(f"card_{i}", grade=(i % 4) + 1, timestamp=i)
                for i in range(12)]

    @pytest.fixture
    def embeddings(self):
        return {f"card_{i}": np.random.randn(384).astype(np.float32)
                for i in range(12)}

    def test_batch_keys(self, reviews, embeddings):
        batch = reviews_to_batch(reviews, embeddings, torch.device("cpu"))
        required = {"D_prev", "S_prev", "R_at_review", "delta_t", "grade",
                    "review_count", "card_embedding_raw", "user_stats",
                    "recalled", "S_target", "D_target"}
        assert required.issubset(set(batch.keys()))

    def test_batch_size(self, reviews, embeddings):
        batch = reviews_to_batch(reviews, embeddings, torch.device("cpu"))
        n = len(reviews)
        assert batch["D_prev"].shape == (n,)
        assert batch["grade"].shape == (n,)
        assert batch["card_embedding_raw"].shape == (n, 384)
        assert batch["user_stats"].shape == (n, 8)

    def test_grade_values_correct(self, reviews, embeddings):
        batch = reviews_to_batch(reviews, embeddings, torch.device("cpu"))
        for i, r in enumerate(reviews):
            assert batch["grade"][i].item() == r.grade

    def test_recalled_binary(self, reviews, embeddings):
        batch = reviews_to_batch(reviews, embeddings, torch.device("cpu"))
        vals = batch["recalled"].unique()
        assert all(v.item() in (0.0, 1.0) for v in vals)

    def test_review_count_increases_per_card(self, reviews, embeddings):
        # Multiple reviews of same card should have ascending count
        same_card_reviews = [
            make_review("card_X", timestamp=i * 1000) for i in range(5)
        ]
        batch = reviews_to_batch(same_card_reviews, {}, torch.device("cpu"))
        counts = batch["review_count"].tolist()
        assert counts == sorted(counts)

    def test_unknown_card_embedding_is_zeros(self, reviews):
        batch = reviews_to_batch(reviews, {}, torch.device("cpu"))
        # All embeddings should be zero since no embeddings provided
        assert torch.allclose(batch["card_embedding_raw"], torch.zeros(len(reviews), 384))

    def test_history_grades_shape(self, reviews, embeddings):
        batch = reviews_to_batch(reviews, embeddings, torch.device("cpu"),
                                 history_len=8)
        assert batch["history_grades"].shape == (len(reviews), 8)
        assert batch["history_delta_ts"].shape == (len(reviews), 8)
        assert batch["history_lengths"].shape == (len(reviews),)

    def test_history_lengths_at_least_1(self, reviews, embeddings):
        batch = reviews_to_batch(reviews, embeddings, torch.device("cpu"))
        assert batch["history_lengths"].min().item() >= 1

    def test_no_nan_in_batch(self, reviews, embeddings):
        batch = reviews_to_batch(reviews, embeddings, torch.device("cpu"))
        for key, tensor in batch.items():
            assert not torch.isnan(tensor).any(), f"NaN found in {key}"


# ---------------------------------------------------------------------------
# TaskSampler
# ---------------------------------------------------------------------------

class TestTaskSampler:
    @pytest.fixture
    def tasks(self):
        np.random.seed(10)
        return [make_task(n_reviews=20) for _ in range(10)]

    def test_filters_short_tasks(self):
        short_tasks = [make_task(n_reviews=5) for _ in range(5)]
        long_tasks = [make_task(n_reviews=20) for _ in range(5)]
        sampler = TaskSampler(short_tasks + long_tasks, min_reviews=10)
        assert len(sampler) == 5  # only the 5 long tasks

    def test_sample_returns_correct_size(self, tasks):
        sampler = TaskSampler(tasks, min_reviews=10)
        batch = sampler.sample(batch_size=4)
        assert len(batch) == 4

    def test_sample_returns_tasks(self, tasks):
        sampler = TaskSampler(tasks, min_reviews=10)
        batch = sampler.sample(batch_size=3)
        for t in batch:
            assert isinstance(t, Task)

    def test_tasks_are_split_after_init(self, tasks):
        sampler = TaskSampler(tasks, min_reviews=10)
        for t in sampler.tasks:
            assert len(t.support_set) > 0

    def test_deterministic_with_same_seed(self):
        tasks = [make_task(n_reviews=20) for _ in range(10)]
        s1 = TaskSampler(tasks, seed=99)
        s2 = TaskSampler(tasks, seed=99)
        ids1 = [t.student_id for t in s1.sample(5)]
        ids2 = [t.student_id for t in s2.sample(5)]
        assert ids1 == ids2


# ---------------------------------------------------------------------------
# ReviewDataset.generate_synthetic
# ---------------------------------------------------------------------------

class TestGenerateSynthetic:
    def test_returns_correct_number_of_tasks(self):
        tasks = ReviewDataset.generate_synthetic(n_students=5,
                                                 reviews_per_student=20,
                                                 n_cards=10,
                                                 seed=0)
        assert len(tasks) == 5

    def test_each_task_has_reviews(self):
        tasks = ReviewDataset.generate_synthetic(n_students=3,
                                                 reviews_per_student=15,
                                                 n_cards=8,
                                                 seed=1)
        for t in tasks:
            assert len(t.reviews) == 15

    def test_grades_in_valid_range(self):
        tasks = ReviewDataset.generate_synthetic(n_students=5,
                                                 reviews_per_student=20,
                                                 n_cards=10,
                                                 seed=2)
        for t in tasks:
            for r in t.reviews:
                assert r.grade in [1, 2, 3, 4]

    def test_embeddings_attached(self):
        tasks = ReviewDataset.generate_synthetic(n_students=2,
                                                 reviews_per_student=10,
                                                 n_cards=5,
                                                 card_raw_dim=384,
                                                 seed=3)
        for t in tasks:
            assert len(t.card_embeddings) > 0

    def test_timestamps_non_negative(self):
        tasks = ReviewDataset.generate_synthetic(n_students=3,
                                                 reviews_per_student=10,
                                                 n_cards=5,
                                                 seed=4)
        for t in tasks:
            for r in t.reviews:
                assert r.timestamp >= 0

    def test_deterministic_with_same_seed(self):
        tasks1 = ReviewDataset.generate_synthetic(n_students=3, seed=77)
        tasks2 = ReviewDataset.generate_synthetic(n_students=3, seed=77)
        for t1, t2 in zip(tasks1, tasks2):
            grades1 = [r.grade for r in t1.reviews]
            grades2 = [r.grade for r in t2.reviews]
            assert grades1 == grades2


# ---------------------------------------------------------------------------
# ReviewDataset.from_csv
# ---------------------------------------------------------------------------

class TestFromCSV:
    def test_loads_from_csv(self, tmp_path):
        csv_path = str(tmp_path / "reviews.csv")
        rows = [
            {"student_id": "s1", "card_id": "c1", "timestamp": "0",
             "elapsed_days": "0.0", "grade": "3"},
            {"student_id": "s1", "card_id": "c2", "timestamp": "86400",
             "elapsed_days": "1.0", "grade": "2"},
            {"student_id": "s2", "card_id": "c1", "timestamp": "0",
             "elapsed_days": "0.0", "grade": "4"},
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        tasks = ReviewDataset.from_csv(csv_path)
        assert len(tasks) == 2  # 2 students
        total_reviews = sum(len(t.reviews) for t in tasks)
        assert total_reviews == 3

    def test_reviews_sorted_by_timestamp(self, tmp_path):
        csv_path = str(tmp_path / "sorted.csv")
        rows = [
            {"student_id": "s1", "card_id": "c1", "timestamp": "200",
             "elapsed_days": "2.0", "grade": "3"},
            {"student_id": "s1", "card_id": "c2", "timestamp": "100",
             "elapsed_days": "1.0", "grade": "2"},
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        tasks = ReviewDataset.from_csv(csv_path)
        ts = [r.timestamp for r in tasks[0].reviews]
        assert ts == sorted(ts)
