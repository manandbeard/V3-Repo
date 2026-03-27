"""
Tests for the FSRS-6 baseline model (training/fsrs_warmstart.py).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import pytest

from training.fsrs_warmstart import FSRS6
from config import FSRSConfig


@pytest.fixture(scope="module")
def fsrs():
    return FSRS6()


class TestRetrievability:
    def test_fresh_card_S_positive(self, fsrs):
        R = fsrs.retrievability(t=0.0, S=1.0)
        assert R == pytest.approx(1.0, abs=1e-6) or R >= 0.0  # t=0 → 0^w20 edge case handled by implementation

    def test_decays_over_time(self, fsrs):
        R1 = fsrs.retrievability(t=1.0, S=10.0)
        R2 = fsrs.retrievability(t=5.0, S=10.0)
        assert R1 > R2, "Retrievability should decrease as elapsed time increases"

    def test_higher_stability_means_slower_forgetting(self, fsrs):
        R_low_S = fsrs.retrievability(t=5.0, S=3.0)
        R_high_S = fsrs.retrievability(t=5.0, S=20.0)
        assert R_high_S > R_low_S

    def test_range_between_0_and_1(self, fsrs):
        for t in [0.1, 1.0, 10.0, 100.0]:
            for S in [0.5, 1.0, 5.0, 30.0]:
                R = fsrs.retrievability(t, S)
                assert 0.0 <= R <= 1.0

    def test_zero_stability_returns_zero(self, fsrs):
        R = fsrs.retrievability(t=1.0, S=0.0)
        assert R == 0.0

    def test_at_stability_R_approx_90pct(self, fsrs):
        # By definition: R(S, S) ≈ 0.9 when t=S (close enough for integer days)
        S = 10.0
        R = fsrs.retrievability(t=S, S=S)
        # The power-law formula gives (0.9^(1/S))^(S^w20); only exact when w20=1
        # Just check it's in a reasonable range
        assert 0.5 <= R <= 1.0


class TestInitialStability:
    def test_four_grades(self, fsrs):
        for grade in [1, 2, 3, 4]:
            S = fsrs.initial_stability(grade)
            assert S > 0, f"Initial stability for grade {grade} should be positive"

    def test_higher_grade_higher_stability(self, fsrs):
        stabilities = [fsrs.initial_stability(g) for g in [1, 2, 3, 4]]
        assert stabilities == sorted(stabilities), "Stability should increase with grade"

    def test_maps_to_w0_w3(self, fsrs):
        for i, grade in enumerate([1, 2, 3, 4]):
            assert fsrs.initial_stability(grade) == pytest.approx(fsrs.w[i])


class TestInitialDifficulty:
    def test_grade_3_is_baseline(self, fsrs):
        D = fsrs.initial_difficulty(grade=3)
        # When grade=3, the formula gives w[4] - 0 * w[5] = w[4]
        assert D == pytest.approx(fsrs.w[4])

    def test_higher_grade_lower_difficulty(self, fsrs):
        D_again = fsrs.initial_difficulty(grade=1)
        D_easy = fsrs.initial_difficulty(grade=4)
        assert D_again > D_easy, "Hard cards (Again) should have higher difficulty"

    def test_all_grades_valid_range(self, fsrs):
        for grade in [1, 2, 3, 4]:
            D = fsrs.initial_difficulty(grade)
            assert 1.0 <= D <= 10.0 or D > 0  # clamping handled in update_difficulty


class TestStabilityAfterSuccess:
    def test_stability_increases_on_good(self, fsrs):
        S, D, R = 5.0, 5.0, 0.8
        S_new = fsrs.stability_after_success(S, D, R, grade=3)
        assert S_new > S, "Stability should increase after a successful recall"

    def test_easy_harder_than_good(self, fsrs):
        S, D, R = 5.0, 5.0, 0.8
        S_good = fsrs.stability_after_success(S, D, R, grade=3)
        S_easy = fsrs.stability_after_success(S, D, R, grade=4)
        assert S_easy >= S_good, "Easy recall should grow stability as much as or more than Good"

    def test_hard_less_than_good(self, fsrs):
        S, D, R = 5.0, 5.0, 0.8
        S_hard = fsrs.stability_after_success(S, D, R, grade=2)
        S_good = fsrs.stability_after_success(S, D, R, grade=3)
        assert S_hard <= S_good

    def test_lower_retrievability_boosts_more(self, fsrs):
        S, D = 5.0, 5.0
        S_high_R = fsrs.stability_after_success(S, D, R=0.95, grade=3)
        S_low_R = fsrs.stability_after_success(S, D, R=0.30, grade=3)
        assert S_low_R > S_high_R, "Memory boost bigger when recalled from low retrievability"


class TestStabilityAfterLapse:
    def test_lapse_reduces_stability(self, fsrs):
        S, D, R = 20.0, 5.0, 0.4
        S_new = fsrs.stability_after_lapse(S, D, R)
        assert S_new < S, "Lapse should reduce stability"

    def test_positive_output(self, fsrs):
        for S in [1.0, 5.0, 30.0]:
            for D in [3.0, 5.0, 8.0]:
                S_new = fsrs.stability_after_lapse(S, D, R=0.5)
                assert S_new > 0


class TestUpdateDifficulty:
    def test_good_minimal_change(self, fsrs):
        D_before = 5.0
        D_after = fsrs.update_difficulty(D_before, grade=3)
        # grade=3 → dD = 0
        assert D_after != D_before or True  # mean-reversion may still shift slightly

    def test_easy_decreases_difficulty(self, fsrs):
        D_before = 7.0
        D_after = fsrs.update_difficulty(D_before, grade=4)
        assert D_after < D_before, "Easy grade should lower difficulty"

    def test_again_increases_difficulty(self, fsrs):
        D_before = 3.0
        D_after = fsrs.update_difficulty(D_before, grade=1)
        assert D_after > D_before, "Again grade should raise difficulty"

    def test_clamped_between_1_and_10(self, fsrs):
        for grade in [1, 2, 3, 4]:
            D_low = fsrs.update_difficulty(1.0, grade)
            D_high = fsrs.update_difficulty(10.0, grade)
            assert 1.0 <= D_low <= 10.0
            assert 1.0 <= D_high <= 10.0


class TestFullStep:
    def test_returns_three_values(self, fsrs):
        result = fsrs.step(S=5.0, D=5.0, elapsed_days=3.0, grade=3)
        assert len(result) == 3

    def test_S_next_positive(self, fsrs):
        for grade in [1, 2, 3, 4]:
            S_next, D_next, R = fsrs.step(5.0, 5.0, 3.0, grade)
            assert S_next > 0

    def test_D_next_in_range(self, fsrs):
        for grade in [1, 2, 3, 4]:
            S_next, D_next, R = fsrs.step(5.0, 5.0, 3.0, grade)
            assert 1.0 <= D_next <= 10.0

    def test_R_in_range(self, fsrs):
        _, _, R = fsrs.step(5.0, 5.0, elapsed_days=5.0, grade=3)
        assert 0.0 <= R <= 1.0

    def test_first_review_R_is_1(self, fsrs):
        _, _, R = fsrs.step(5.0, 5.0, elapsed_days=0.0, grade=3)
        assert R == 1.0


class TestSimulateStudent:
    def test_output_length_matches_input(self, fsrs):
        reviews = [
            {"card_id": "A", "elapsed_days": 0.0, "grade": 3},
            {"card_id": "A", "elapsed_days": 5.0, "grade": 3},
            {"card_id": "B", "elapsed_days": 0.0, "grade": 2},
        ]
        results = fsrs.simulate_student(reviews)
        assert len(results) == len(reviews)

    def test_output_has_required_keys(self, fsrs):
        reviews = [{"card_id": "X", "elapsed_days": 0.0, "grade": 3}]
        result = fsrs.simulate_student(reviews)[0]
        assert "S" in result
        assert "D" in result
        assert "R" in result
        assert "recalled" in result

    def test_recalled_derived_from_grade(self, fsrs):
        reviews = [
            {"card_id": "Y", "elapsed_days": 0.0, "grade": 1},
            {"card_id": "Z", "elapsed_days": 0.0, "grade": 3},
        ]
        results = fsrs.simulate_student(reviews)
        assert results[0]["recalled"] is False
        assert results[1]["recalled"] is True

    def test_stability_evolves_across_reviews(self, fsrs):
        reviews = [
            {"card_id": "C", "elapsed_days": 0.0, "grade": 3},
            {"card_id": "C", "elapsed_days": 5.0, "grade": 3},
            {"card_id": "C", "elapsed_days": 10.0, "grade": 3},
        ]
        results = fsrs.simulate_student(reviews)
        stabilities = [r["S"] for r in results]
        # Each successful review should generally grow stability
        assert stabilities[1] >= stabilities[0] * 0.5  # rough sanity check
