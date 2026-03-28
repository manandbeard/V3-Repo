"""Tests for task sampling, review data, and batch construction."""

import sys
import os
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.task_sampler import Review, Task, TaskSampler, ReviewDataset, reviews_to_batch


class TestReview:
    def test_creation(self):
        r = Review(card_id="c1", timestamp=0, elapsed_days=0.0,
                   grade=3, recalled=True)
        assert r.card_id == "c1"
        assert r.grade == 3
        assert r.recalled is True
        assert r.S_prev == 1.0  # default
        assert r.D_prev == 5.0  # default

    def test_recalled_matches_grade(self):
        r_success = Review(card_id="c1", timestamp=0, elapsed_days=0.0,
                           grade=3, recalled=True)
        r_fail = Review(card_id="c1", timestamp=0, elapsed_days=0.0,
                        grade=1, recalled=False)
        assert r_success.recalled is True
        assert r_fail.recalled is False


class TestTask:
    def test_split(self, sample_reviews):
        task = Task(student_id="s1", reviews=sample_reviews)
        task.split(support_ratio=0.70)
        assert len(task.support_set) + len(task.query_set) == len(sample_reviews)
        assert len(task.support_set) > 0
        assert len(task.query_set) > 0

    def test_split_chronological(self, sample_reviews):
        """Support set should come before query set chronologically."""
        task = Task(student_id="s1", reviews=sample_reviews)
        task.split(support_ratio=0.70)
        if task.support_set and task.query_set:
            last_support_ts = task.support_set[-1].timestamp
            first_query_ts = task.query_set[0].timestamp
            assert first_query_ts >= last_support_ts

    def test_get_review_history(self, sample_reviews):
        task = Task(student_id="s1", reviews=sample_reviews)
        # c1 appears at index 0 and 2
        history = task.get_review_history("c1", up_to_idx=3)
        assert len(history) == 2
        assert all(r.card_id == "c1" for r in history)

    def test_unique_cards(self, sample_reviews):
        task = Task(student_id="s1", reviews=sample_reviews)
        cards = task.unique_cards
        assert set(cards) == {"c1", "c2", "c3"}


class TestReviewsToBatch:
    def test_batch_shapes(self, sample_reviews, card_embeddings):
        batch = reviews_to_batch(sample_reviews, card_embeddings)
        n = len(sample_reviews)
        assert batch["D_prev"].shape == (n,)
        assert batch["S_prev"].shape == (n,)
        assert batch["R_at_review"].shape == (n,)
        assert batch["delta_t"].shape == (n,)
        assert batch["grade"].shape == (n,)
        assert batch["review_count"].shape == (n,)
        assert batch["card_embedding_raw"].shape == (n, 384)
        assert batch["user_stats"].shape == (n, 8)
        assert batch["recalled"].shape == (n,)
        assert batch["history_grades"].shape[0] == n
        assert batch["history_delta_ts"].shape[0] == n
        assert batch["history_lengths"].shape == (n,)

    def test_batch_on_device(self, sample_reviews, card_embeddings, device):
        batch = reviews_to_batch(sample_reviews, card_embeddings, device)
        for key, tensor in batch.items():
            assert tensor.device == device

    def test_review_count_accumulates(self, card_embeddings):
        """Review count should increase for repeated cards."""
        reviews = [
            Review(card_id="c1", timestamp=0, elapsed_days=0.0,
                   grade=3, recalled=True),
            Review(card_id="c1", timestamp=1, elapsed_days=1.0,
                   grade=3, recalled=True),
            Review(card_id="c1", timestamp=2, elapsed_days=2.0,
                   grade=3, recalled=True),
        ]
        batch = reviews_to_batch(reviews, card_embeddings)
        counts = batch["review_count"].tolist()
        assert counts == [1.0, 2.0, 3.0]

    def test_unknown_card_gets_zero_embedding(self):
        """Cards not in embeddings dict should get zero vectors."""
        reviews = [
            Review(card_id="unknown", timestamp=0, elapsed_days=0.0,
                   grade=3, recalled=True),
        ]
        batch = reviews_to_batch(reviews, card_embeddings={})
        assert (batch["card_embedding_raw"] == 0).all()


class TestTaskSampler:
    def test_filters_short_histories(self):
        tasks = [
            Task(student_id="s1", reviews=[
                Review(card_id=f"c{i}", timestamp=i, elapsed_days=float(i),
                       grade=3, recalled=True)
                for i in range(5)
            ]),  # Only 5 reviews - should be filtered
            Task(student_id="s2", reviews=[
                Review(card_id=f"c{i}", timestamp=i, elapsed_days=float(i),
                       grade=3, recalled=True)
                for i in range(20)
            ]),  # 20 reviews - should pass
        ]
        sampler = TaskSampler(tasks, min_reviews=10)
        assert len(sampler) == 1

    def test_sample_returns_correct_count(self):
        tasks = [
            Task(student_id=f"s{j}", reviews=[
                Review(card_id=f"c{i}", timestamp=i, elapsed_days=float(i),
                       grade=3, recalled=True)
                for i in range(20)
            ])
            for j in range(5)
        ]
        sampler = TaskSampler(tasks, min_reviews=10)
        batch = sampler.sample(batch_size=3)
        assert len(batch) == 3

    def test_sampled_tasks_have_splits(self):
        tasks = [
            Task(student_id=f"s{j}", reviews=[
                Review(card_id=f"c{i}", timestamp=i, elapsed_days=float(i),
                       grade=3, recalled=True)
                for i in range(20)
            ])
            for j in range(3)
        ]
        sampler = TaskSampler(tasks, min_reviews=10)
        batch = sampler.sample(2)
        for task in batch:
            assert len(task.support_set) > 0
            assert len(task.query_set) > 0


class TestSyntheticGeneration:
    def test_generate_correct_count(self):
        tasks = ReviewDataset.generate_synthetic(
            n_students=10, reviews_per_student=20, n_cards=15, seed=42
        )
        assert len(tasks) == 10
        for task in tasks:
            assert len(task.reviews) == 20

    def test_synthetic_reviews_have_state(self):
        tasks = ReviewDataset.generate_synthetic(
            n_students=5, reviews_per_student=10, n_cards=10, seed=42
        )
        for task in tasks:
            for review in task.reviews:
                assert review.S_prev > 0
                assert review.D_prev >= 1.0
                assert review.D_prev <= 10.0
                assert 0.0 <= review.R_at_review <= 1.0

    def test_synthetic_has_card_embeddings(self):
        tasks = ReviewDataset.generate_synthetic(
            n_students=3, reviews_per_student=10, n_cards=5, seed=42
        )
        for task in tasks:
            assert len(task.card_embeddings) == 5
            for cid, emb in task.card_embeddings.items():
                assert emb.shape == (384,)

    def test_synthetic_deterministic(self):
        """Same seed should produce identical data."""
        tasks1 = ReviewDataset.generate_synthetic(
            n_students=5, reviews_per_student=10, seed=123
        )
        tasks2 = ReviewDataset.generate_synthetic(
            n_students=5, reviews_per_student=10, seed=123
        )
        for t1, t2 in zip(tasks1, tasks2):
            assert t1.student_id == t2.student_id
            assert len(t1.reviews) == len(t2.reviews)
            for r1, r2 in zip(t1.reviews, t2.reviews):
                assert r1.card_id == r2.card_id
                assert r1.grade == r2.grade
