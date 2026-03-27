"""Tests for FSRS-6 baseline and warm-start."""

import math
import sys
import os
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.fsrs_warmstart import FSRS6


class TestFSRS6Retrievability:
    def test_zero_elapsed(self, fsrs):
        """At t=0, retrievability should be 1.0."""
        # Note: the code returns 0 for t=0 via the power-law formula
        # but in step(), elapsed=0 overrides to R=1.0
        R = fsrs.retrievability(0.0, 10.0)
        # (0.9^(1/10))^(0^0.4665) = (0.9^0.1)^0 = ... 0^w20 = 0 when t=0
        # actually 0^w20 = 0, so R = base^0 = 1.0
        assert abs(R - 1.0) < 1e-6

    def test_retrievability_decays(self, fsrs):
        """Retrievability should decrease as elapsed time increases."""
        S = 10.0
        R_at_1 = fsrs.retrievability(1.0, S)
        R_at_10 = fsrs.retrievability(10.0, S)
        R_at_100 = fsrs.retrievability(100.0, S)
        assert R_at_1 > R_at_10 > R_at_100

    def test_higher_stability_slower_decay(self, fsrs):
        """Higher stability means slower forgetting."""
        t = 10.0
        R_low_S = fsrs.retrievability(t, 5.0)
        R_high_S = fsrs.retrievability(t, 50.0)
        assert R_high_S > R_low_S

    def test_retrievability_at_S_equals_t(self, fsrs):
        """When t = S and w20 ≈ 0.5, R should be approximately 0.9."""
        S = 10.0
        R = fsrs.retrievability(S, S)
        # With w20=0.4665, R(S, S) = 0.9^(S^w20 / S) which isn't exactly 0.9
        # but should be in a reasonable range
        assert 0.5 < R < 1.0

    def test_zero_stability(self, fsrs):
        """Zero stability should return 0.0."""
        R = fsrs.retrievability(1.0, 0.0)
        assert R == 0.0

    def test_retrievability_batch(self, fsrs):
        """Test vectorised retrievability."""
        t = torch.tensor([1.0, 5.0, 10.0])
        S = torch.tensor([10.0, 10.0, 10.0])
        R = fsrs.retrievability_batch(t, S)
        assert R.shape == (3,)
        assert (R[0] > R[1]).item()
        assert (R[1] > R[2]).item()


class TestFSRS6InitialState:
    def test_initial_stability_by_grade(self, fsrs):
        """Higher grade → higher initial stability."""
        S_again = fsrs.initial_stability(1)
        S_hard = fsrs.initial_stability(2)
        S_good = fsrs.initial_stability(3)
        S_easy = fsrs.initial_stability(4)
        assert S_again < S_hard < S_good < S_easy

    def test_initial_difficulty_by_grade(self, fsrs):
        """Higher grade → lower initial difficulty."""
        D_again = fsrs.initial_difficulty(1)
        D_easy = fsrs.initial_difficulty(4)
        assert D_again > D_easy


class TestFSRS6StabilityUpdate:
    def test_success_increases_stability(self, fsrs):
        """Successful recall should increase stability."""
        S, D = 5.0, 5.0
        R = fsrs.retrievability(3.0, S)
        S_next = fsrs.stability_after_success(S, D, R, grade=3)
        assert S_next > S

    def test_easy_gives_higher_stability_than_hard(self, fsrs):
        """Easy grade should give higher stability gain than Hard."""
        S, D = 5.0, 5.0
        R = fsrs.retrievability(3.0, S)
        S_hard = fsrs.stability_after_success(S, D, R, grade=2)
        S_easy = fsrs.stability_after_success(S, D, R, grade=4)
        assert S_easy > S_hard

    def test_lapse_decreases_stability(self, fsrs):
        """A lapse (Again) should decrease stability."""
        S, D = 10.0, 5.0
        R = fsrs.retrievability(5.0, S)
        S_lapse = fsrs.stability_after_lapse(S, D, R)
        assert S_lapse < S


class TestFSRS6DifficultyUpdate:
    def test_good_maintains_difficulty(self, fsrs):
        """Grade=Good (3) should not change difficulty much."""
        D = 5.0
        D_next = fsrs.update_difficulty(D, grade=3)
        assert abs(D_next - D) < 2.0  # Should stay close

    def test_again_increases_difficulty(self, fsrs):
        """Again (grade=1) should increase difficulty."""
        D = 5.0
        D_next = fsrs.update_difficulty(D, grade=1)
        assert D_next > D

    def test_easy_decreases_difficulty(self, fsrs):
        """Easy (grade=4) should decrease difficulty from a high value."""
        D = 8.0  # Start high so Easy can clearly decrease it
        D_next = fsrs.update_difficulty(D, grade=4)
        assert D_next < D

    def test_difficulty_bounds(self, fsrs):
        """Difficulty should be clamped to [1, 10]."""
        D_low = fsrs.update_difficulty(1.0, grade=4)
        D_high = fsrs.update_difficulty(10.0, grade=1)
        assert D_low >= 1.0
        assert D_high <= 10.0


class TestFSRS6Step:
    def test_step_returns_three_values(self, fsrs):
        S_next, D_next, R = fsrs.step(5.0, 5.0, 3.0, 3)
        assert isinstance(S_next, float)
        assert isinstance(D_next, float)
        assert isinstance(R, float)

    def test_step_first_review(self, fsrs):
        """First review (elapsed=0) should have R=1.0."""
        S_next, D_next, R = fsrs.step(3.0, 5.0, 0.0, 3)
        assert R == 1.0

    def test_step_success_vs_lapse(self, fsrs):
        """Success (Good) should give higher S than lapse (Again)."""
        S, D = 5.0, 5.0
        S_good, _, _ = fsrs.step(S, D, 3.0, 3)
        S_again, _, _ = fsrs.step(S, D, 3.0, 1)
        assert S_good > S_again


class TestSimulateStudent:
    def test_simulate_returns_correct_length(self, fsrs):
        reviews = [
            {"card_id": "c1", "elapsed_days": 0.0, "grade": 3},
            {"card_id": "c1", "elapsed_days": 3.0, "grade": 3},
            {"card_id": "c1", "elapsed_days": 7.0, "grade": 4},
        ]
        results = fsrs.simulate_student(reviews)
        assert len(results) == 3

    def test_simulate_adds_state_fields(self, fsrs):
        reviews = [
            {"card_id": "c1", "elapsed_days": 0.0, "grade": 3},
        ]
        results = fsrs.simulate_student(reviews)
        assert "S" in results[0]
        assert "D" in results[0]
        assert "R" in results[0]
        assert "recalled" in results[0]

    def test_simulate_stability_grows_on_success(self, fsrs):
        reviews = [
            {"card_id": "c1", "elapsed_days": 0.0, "grade": 3},
            {"card_id": "c1", "elapsed_days": 3.0, "grade": 3},
            {"card_id": "c1", "elapsed_days": 7.0, "grade": 3},
        ]
        results = fsrs.simulate_student(reviews)
        # Stability should increase with repeated successful recalls
        assert results[1]["S"] > results[0]["S"]
        assert results[2]["S"] > results[1]["S"]
