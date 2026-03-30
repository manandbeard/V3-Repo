"""Tests for the MetaSRS Flask REST API."""

import sys
import os
import json
import pytest
import torch

# Ensure meta-srs root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api import create_app
from config import MetaSRSConfig


@pytest.fixture
def app():
    """Create a test Flask app with no checkpoint (fresh model)."""
    app = create_app(checkpoint_path=None)
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app):
    """Flask test client."""
    return app.test_client()


# ── Health ────────────────────────────────────────────────────────────────


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert "model_params" in data
        assert "active_students" in data

    def test_health_model_params_reasonable(self, client):
        resp = client.get("/api/health")
        data = resp.get_json()
        assert data["model_params"] > 10_000  # ~50K expected


# ── Submit Review ─────────────────────────────────────────────────────────


class TestSubmitReview:
    def test_submit_first_review(self, client):
        resp = client.post("/api/review", json={
            "student_id": "s1",
            "card_id": "c1",
            "grade": 3,
            "elapsed_days": 0.0,
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["student_id"] == "s1"
        assert data["phase"] == "ZERO_SHOT"
        assert data["n_reviews"] == 1
        assert "schedule" in data
        assert data["schedule"]["card_id"] == "c1"
        assert data["schedule"]["interval_days"] >= 1

    def test_submit_multiple_reviews_transitions_phase(self, client):
        # Submit 6 reviews to cross the phase1 threshold (default: 5)
        for i in range(6):
            resp = client.post("/api/review", json={
                "student_id": "s2",
                "card_id": f"c{i}",
                "grade": 3,
                "elapsed_days": float(i),
            })
            assert resp.status_code == 200

        data = resp.get_json()
        assert data["n_reviews"] == 6
        assert data["phase"] == "RAPID_ADAPT"

    def test_submit_review_subsequent_same_card(self, client):
        # First review
        client.post("/api/review", json={
            "student_id": "s3",
            "card_id": "c1",
            "grade": 3,
            "elapsed_days": 0.0,
        })
        # Second review of same card
        resp = client.post("/api/review", json={
            "student_id": "s3",
            "card_id": "c1",
            "grade": 4,
            "elapsed_days": 3.0,
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["n_reviews"] == 2

    def test_submit_review_missing_fields(self, client):
        resp = client.post("/api/review", json={
            "student_id": "s1",
        })
        assert resp.status_code == 400
        data = resp.get_json()
        assert "details" in data

    def test_submit_review_invalid_grade(self, client):
        resp = client.post("/api/review", json={
            "student_id": "s1",
            "card_id": "c1",
            "grade": 5,
            "elapsed_days": 1.0,
        })
        assert resp.status_code == 400

    def test_submit_review_negative_elapsed(self, client):
        resp = client.post("/api/review", json={
            "student_id": "s1",
            "card_id": "c1",
            "grade": 3,
            "elapsed_days": -1.0,
        })
        assert resp.status_code == 400

    def test_submit_review_not_json(self, client):
        resp = client.post("/api/review", data="not json")
        assert resp.status_code == 400

    def test_schedule_result_has_expected_fields(self, client):
        resp = client.post("/api/review", json={
            "student_id": "s1",
            "card_id": "c1",
            "grade": 3,
            "elapsed_days": 0.0,
        })
        sched = resp.get_json()["schedule"]
        expected_keys = {
            "card_id", "interval_days", "p_recall_mean", "p_recall_sigma",
            "S_pred", "D_pred", "confidence_factor", "difficulty_factor",
        }
        assert set(sched.keys()) == expected_keys


# ── Schedule Deck ─────────────────────────────────────────────────────────


class TestScheduleDeck:
    def test_schedule_deck(self, client):
        resp = client.post("/api/schedule", json={
            "student_id": "s1",
            "cards": [
                {"card_id": "c1", "elapsed_days": 1.0},
                {"card_id": "c2", "elapsed_days": 0.0},
                {"card_id": "c3", "elapsed_days": 5.0},
            ],
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["cards"]) == 3
        assert data["cards"][0]["card_id"] == "c1"

    def test_schedule_empty_cards(self, client):
        resp = client.post("/api/schedule", json={
            "student_id": "s1",
            "cards": [],
        })
        assert resp.status_code == 400

    def test_schedule_missing_student(self, client):
        resp = client.post("/api/schedule", json={
            "cards": [{"card_id": "c1", "elapsed_days": 1.0}],
        })
        assert resp.status_code == 400


# ── Student Status ────────────────────────────────────────────────────────


class TestStudentStatus:
    def test_new_student_status(self, client):
        resp = client.get("/api/student/new_student/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["student_id"] == "new_student"
        assert data["phase"] == "ZERO_SHOT"
        assert data["n_reviews"] == 0

    def test_status_after_reviews(self, client):
        client.post("/api/review", json={
            "student_id": "s5",
            "card_id": "c1",
            "grade": 3,
            "elapsed_days": 0.0,
        })
        resp = client.get("/api/student/s5/status")
        data = resp.get_json()
        assert data["n_reviews"] == 1
        assert data["unique_cards"] == 1


# ── Reset Student ─────────────────────────────────────────────────────────


class TestResetStudent:
    def test_reset_clears_state(self, client):
        # Add a review
        client.post("/api/review", json={
            "student_id": "s6",
            "card_id": "c1",
            "grade": 3,
            "elapsed_days": 0.0,
        })
        # Verify review was recorded
        resp = client.get("/api/student/s6/status")
        assert resp.get_json()["n_reviews"] == 1

        # Reset
        resp = client.post("/api/student/s6/reset")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "reset"

        # Verify cleared
        resp = client.get("/api/student/s6/status")
        assert resp.get_json()["n_reviews"] == 0

    def test_reset_nonexistent_student(self, client):
        resp = client.post("/api/student/doesnt_exist/reset")
        assert resp.status_code == 200


# ── Next Card ─────────────────────────────────────────────────────────────


class TestNextCard:
    def test_next_card_selection(self, client):
        resp = client.post("/api/next-card", json={
            "student_id": "s7",
            "cards": [
                {"card_id": "c1", "elapsed_days": 1.0},
                {"card_id": "c2", "elapsed_days": 10.0},
            ],
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["selected"]["card_id"] in ("c1", "c2")
        assert data["strategy"] == "uncertainty_weighted"

    def test_next_card_custom_strategy(self, client):
        resp = client.post("/api/next-card", json={
            "student_id": "s8",
            "cards": [
                {"card_id": "c1", "elapsed_days": 1.0},
                {"card_id": "c2", "elapsed_days": 10.0},
            ],
            "strategy": "most_due",
        })
        assert resp.status_code == 200
        assert resp.get_json()["strategy"] == "most_due"

    def test_next_card_empty_cards(self, client):
        resp = client.post("/api/next-card", json={
            "student_id": "s9",
            "cards": [],
        })
        assert resp.status_code == 400


# ── 404 ───────────────────────────────────────────────────────────────────


class TestErrorHandling:
    def test_404_returns_json(self, client):
        resp = client.get("/api/nonexistent")
        assert resp.status_code == 404
        assert resp.get_json()["error"] == "Endpoint not found"
