"""
MetaSRS Flask API — REST backend for Node.js web app.

Provides HTTP endpoints for the MetaSRS spaced-repetition engine
so a Node.js frontend can submit reviews, get scheduling predictions,
and manage per-student adaptation state.

Usage (local):
    python api.py

Usage (Replit / production):
    gunicorn api:app --bind 0.0.0.0:5000
"""

import os
import sys
import io
import json
import time
import threading
import logging
from dataclasses import asdict
from collections import OrderedDict

import torch

# Ensure meta-srs root is on the path for bare imports used by submodules
_meta_srs_root = os.path.dirname(os.path.abspath(__file__))
if _meta_srs_root not in sys.path:
    sys.path.insert(0, _meta_srs_root)

from flask import Flask, request, jsonify

from config import MetaSRSConfig, ModelConfig
from models.memory_net import MemoryNet
from inference.adaptation import FastAdapter
from inference.scheduling import Scheduler, ScheduleResult
from data.task_sampler import Review, reviews_to_batch
from training.fsrs_warmstart import FSRS6

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("metasrs-api")

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    checkpoint_path: str | None = None,
    config: MetaSRSConfig | None = None,
) -> Flask:
    """
    Create and configure the Flask application.

    Args:
        checkpoint_path: Path to phi_star.pt checkpoint. If None, uses
            a freshly-initialised model (good for testing/demo).
        config: MetaSRS config override.

    Returns:
        Configured Flask app with all routes registered.
    """
    app = Flask(__name__)
    cfg = config or MetaSRSConfig()
    device = torch.device("cpu")

    # ── Model initialisation ──────────────────────────────────────────
    model_kwargs = {
        k: v for k, v in cfg.model.__dict__.items()
        if k in MemoryNet.__init__.__code__.co_varnames
    }
    model = MemoryNet(**model_kwargs).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        phi_star: OrderedDict = ckpt if isinstance(ckpt, OrderedDict) else ckpt.get("phi", ckpt)
        model.load_state_dict(phi_star)
        logger.info("Loaded checkpoint from %s", checkpoint_path)
    else:
        phi_star = OrderedDict(model.state_dict())
        if checkpoint_path:
            logger.warning("Checkpoint %s not found — using fresh model", checkpoint_path)
        else:
            logger.info("No checkpoint specified — using fresh model")

    # FSRS-6 baseline (for first-review state initialisation)
    fsrs = FSRS6(cfg.fsrs)

    # ── Per-student state ─────────────────────────────────────────────
    # In production, replace with a database-backed store.
    _adapters: dict[str, FastAdapter] = {}
    _lock = threading.Lock()

    def _get_adapter(student_id: str) -> FastAdapter:
        """Get or create a FastAdapter for a student (thread-safe)."""
        with _lock:
            if student_id not in _adapters:
                _adapters[student_id] = FastAdapter(
                    model=MemoryNet(**model_kwargs).to(device),
                    phi_star=phi_star,
                    config=cfg,
                    device=device,
                )
            return _adapters[student_id]

    # ── Helpers ───────────────────────────────────────────────────────

    def _schedule_result_to_dict(r: ScheduleResult) -> dict:
        return {
            "card_id": r.card_id,
            "interval_days": r.interval_days,
            "p_recall_mean": round(r.p_recall_mean, 4),
            "p_recall_sigma": round(r.p_recall_sigma, 4),
            "S_pred": round(r.S_pred, 4),
            "D_pred": round(r.D_pred, 4),
            "confidence_factor": round(r.confidence_factor, 4),
            "difficulty_factor": round(r.difficulty_factor, 4),
        }

    def _validate_grade(grade) -> int | None:
        """Validate grade is 1-4 integer."""
        try:
            g = int(grade)
            if 1 <= g <= 4:
                return g
        except (TypeError, ValueError):
            pass
        return None

    # ── Routes ────────────────────────────────────────────────────────

    @app.route("/api/health", methods=["GET"])
    def health():
        """Health check for Replit / load balancers."""
        return jsonify({
            "status": "ok",
            "model_params": model.count_parameters(),
            "active_students": len(_adapters),
            "timestamp": int(time.time()),
        })

    @app.route("/api/review", methods=["POST"])
    def submit_review():
        """
        Submit a review event for a student.

        Triggers adaptation if appropriate, then returns the updated
        scheduling prediction for the reviewed card.

        Expected JSON body:
        {
            "student_id": "student_001",
            "card_id": "card_042",
            "grade": 3,              // 1=Again, 2=Hard, 3=Good, 4=Easy
            "elapsed_days": 2.5      // days since last review (0 for new card)
        }
        """
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400

        # Validate required fields
        student_id = data.get("student_id")
        card_id = data.get("card_id")
        grade = _validate_grade(data.get("grade"))
        elapsed_days = data.get("elapsed_days")

        errors = []
        if not student_id:
            errors.append("student_id is required")
        if not card_id:
            errors.append("card_id is required")
        if grade is None:
            errors.append("grade must be 1 (Again), 2 (Hard), 3 (Good), or 4 (Easy)")
        if elapsed_days is None:
            errors.append("elapsed_days is required")
        else:
            try:
                elapsed_days = float(elapsed_days)
                if elapsed_days < 0:
                    errors.append("elapsed_days must be >= 0")
            except (TypeError, ValueError):
                errors.append("elapsed_days must be a number")

        if errors:
            return jsonify({"error": "Validation failed", "details": errors}), 400

        adapter = _get_adapter(student_id)

        # Compute memory state for this card from student's history
        card_history = [r for r in adapter.reviews if r.card_id == card_id]
        if card_history:
            last = card_history[-1]
            S_prev = last.S_target
            D_prev = last.D_target
            R_at = fsrs.retrievability(elapsed_days, S_prev) if elapsed_days > 0 else 1.0
        else:
            # First review of this card for this student
            S_prev = fsrs.initial_stability(grade)
            D_prev = fsrs.initial_difficulty(grade)
            R_at = 1.0
            elapsed_days = 0.0

        # Compute FSRS-6 targets for warm-start consistency
        S_target, D_target, _ = fsrs.step(S_prev, D_prev, elapsed_days, grade)

        review = Review(
            card_id=card_id,
            timestamp=int(time.time()),
            elapsed_days=elapsed_days,
            grade=grade,
            recalled=grade >= 2,
            S_prev=S_prev,
            D_prev=D_prev,
            R_at_review=R_at,
            S_target=S_target,
            D_target=D_target,
        )

        # Process review (triggers adaptation if thresholds met)
        adapter.add_review(review)

        # Schedule this card with the current (possibly adapted) model
        adapted_model = adapter.get_model()
        scheduler = Scheduler(adapted_model, cfg.scheduling, mc_samples=cfg.model.mc_samples)

        batch = reviews_to_batch([review], device)
        features = adapted_model.build_features(
            batch["D_prev"], batch["S_prev"], batch["R_at_review"],
            batch["delta_t"], batch["grade"], batch["review_count"],
            batch["user_stats"],
            batch.get("history_grades"), batch.get("history_delta_ts"),
            batch.get("history_lengths"),
        )

        result = scheduler.schedule_card(card_id, features, batch["S_prev"])

        return jsonify({
            "student_id": student_id,
            "phase": adapter.phase.name,
            "n_reviews": adapter.n_reviews,
            "schedule": _schedule_result_to_dict(result),
        })

    @app.route("/api/schedule", methods=["POST"])
    def schedule_deck():
        """
        Get scheduling predictions for a list of cards.

        Expected JSON body:
        {
            "student_id": "student_001",
            "cards": [
                {"card_id": "c1", "elapsed_days": 3.0},
                {"card_id": "c2", "elapsed_days": 0.0}
            ]
        }
        """
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400

        student_id = data.get("student_id")
        cards = data.get("cards", [])

        if not student_id:
            return jsonify({"error": "student_id is required"}), 400
        if not cards:
            return jsonify({"error": "cards list is required and must not be empty"}), 400

        adapter = _get_adapter(student_id)
        adapted_model = adapter.get_model()
        scheduler = Scheduler(adapted_model, cfg.scheduling, mc_samples=cfg.model.mc_samples)

        # Build Review objects for each card to get features
        reviews_for_batch = []
        card_ids = []
        for card_data in cards:
            cid = card_data.get("card_id")
            elapsed = float(card_data.get("elapsed_days", 0.0))
            if not cid:
                continue

            card_history = [r for r in adapter.reviews if r.card_id == cid]
            if card_history:
                last = card_history[-1]
                S_prev = last.S_target
                D_prev = last.D_target
            else:
                S_prev = fsrs.initial_stability(3)  # Default to "Good"
                D_prev = fsrs.initial_difficulty(3)

            R_at = fsrs.retrievability(elapsed, S_prev) if elapsed > 0 else 1.0

            reviews_for_batch.append(Review(
                card_id=cid,
                timestamp=int(time.time()),
                elapsed_days=elapsed,
                grade=3,  # Placeholder — scheduling doesn't depend on grade
                recalled=True,
                S_prev=S_prev,
                D_prev=D_prev,
                R_at_review=R_at,
            ))
            card_ids.append(cid)

        if not reviews_for_batch:
            return jsonify({"error": "No valid cards in request"}), 400

        batch = reviews_to_batch(reviews_for_batch, device)
        features = adapted_model.build_features(
            batch["D_prev"], batch["S_prev"], batch["R_at_review"],
            batch["delta_t"], batch["grade"], batch["review_count"],
            batch["user_stats"],
            batch.get("history_grades"), batch.get("history_delta_ts"),
            batch.get("history_lengths"),
        )

        results = scheduler.schedule_deck(card_ids, features, batch["S_prev"])

        return jsonify({
            "student_id": student_id,
            "phase": adapter.phase.name,
            "n_reviews": adapter.n_reviews,
            "cards": [_schedule_result_to_dict(r) for r in results],
        })

    @app.route("/api/student/<student_id>/status", methods=["GET"])
    def student_status(student_id: str):
        """Get a student's current adaptation phase and stats."""
        adapter = _get_adapter(student_id)

        return jsonify({
            "student_id": student_id,
            "phase": adapter.phase.name,
            "n_reviews": adapter.n_reviews,
            "unique_cards": len({r.card_id for r in adapter.reviews}),
        })

    @app.route("/api/student/<student_id>/reset", methods=["POST"])
    def reset_student(student_id: str):
        """Reset a student's personalisation back to phi*."""
        with _lock:
            if student_id in _adapters:
                del _adapters[student_id]

        return jsonify({
            "student_id": student_id,
            "status": "reset",
            "phase": "ZERO_SHOT",
            "n_reviews": 0,
        })

    @app.route("/api/next-card", methods=["POST"])
    def next_card():
        """
        Select the next card to review using uncertainty-weighted exploration.

        Expected JSON body:
        {
            "student_id": "student_001",
            "cards": [
                {"card_id": "c1", "elapsed_days": 3.0},
                {"card_id": "c2", "elapsed_days": 0.0}
            ],
            "strategy": "uncertainty_weighted"  // optional
        }
        """
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400

        student_id = data.get("student_id")
        cards = data.get("cards", [])
        strategy = data.get("strategy", "uncertainty_weighted")

        if not student_id:
            return jsonify({"error": "student_id is required"}), 400
        if not cards:
            return jsonify({"error": "cards list is required"}), 400

        adapter = _get_adapter(student_id)
        adapted_model = adapter.get_model()
        scheduler = Scheduler(adapted_model, cfg.scheduling, mc_samples=cfg.model.mc_samples)

        reviews_for_batch = []
        card_ids = []
        for card_data in cards:
            cid = card_data.get("card_id")
            elapsed = float(card_data.get("elapsed_days", 0.0))
            if not cid:
                continue

            card_history = [r for r in adapter.reviews if r.card_id == cid]
            if card_history:
                last = card_history[-1]
                S_prev = last.S_target
                D_prev = last.D_target
            else:
                S_prev = fsrs.initial_stability(3)
                D_prev = fsrs.initial_difficulty(3)

            R_at = fsrs.retrievability(elapsed, S_prev) if elapsed > 0 else 1.0

            reviews_for_batch.append(Review(
                card_id=cid,
                timestamp=int(time.time()),
                elapsed_days=elapsed,
                grade=3,
                recalled=True,
                S_prev=S_prev,
                D_prev=D_prev,
                R_at_review=R_at,
            ))
            card_ids.append(cid)

        if not reviews_for_batch:
            return jsonify({"error": "No valid cards in request"}), 400

        batch = reviews_to_batch(reviews_for_batch, device)
        features = adapted_model.build_features(
            batch["D_prev"], batch["S_prev"], batch["R_at_review"],
            batch["delta_t"], batch["grade"], batch["review_count"],
            batch["user_stats"],
            batch.get("history_grades"), batch.get("history_delta_ts"),
            batch.get("history_lengths"),
        )

        deck_results = scheduler.schedule_deck(card_ids, features, batch["S_prev"])
        selected = scheduler.select_next_card(deck_results, strategy=strategy)

        return jsonify({
            "student_id": student_id,
            "selected": _schedule_result_to_dict(selected),
            "strategy": strategy,
        })

    # ── Error handlers ────────────────────────────────────────────────

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Endpoint not found"}), 404

    @app.errorhandler(500)
    def internal_error(e):
        logger.exception("Internal server error")
        return jsonify({"error": "Internal server error"}), 500

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# Module-level app instance for gunicorn: `gunicorn api:app`
app = create_app(
    checkpoint_path=os.environ.get("METASRS_CHECKPOINT", "checkpoints/phi_star.pt"),
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    logger.info("Starting MetaSRS API on port %d (debug=%s)", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug)
