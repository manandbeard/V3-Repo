"""
Evaluation Framework (Section 7).

Key Metrics:
    Metric              Description                         Target
    ──────────────────────────────────────────────────────────────────
    AUC-ROC             Recall prediction accuracy          > 0.80 (FSRS-6: ~0.78)
    Calibration Error   |predicted R - actual R|            < 0.05
    Cold-Start AUC      AUC with only 5 reviews             > 0.73
    Adaptation Speed    Reviews to match FSRS-6 AUC         < 30 reviews
    30-day Retention    Recall rate at 30 days              > 85%
    RMSE (stability)    Error in S estimation               < 2.0 days

Ablation Study Design (3 variants):
    A: Baseline         — FSRS-6 population params, no adaptation (cold-start)
    B: Reptile only     — Meta-params, no GRU history encoder
    C: Full model       — Reptile + GRU history
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
from copy import deepcopy

from config import MetaSRSConfig
from models.memory_net import MemoryNet
from training.loss import MetaSRSLoss, compute_loss
from training.reptile import inner_loop, sample_batch
from data.task_sampler import Task, reviews_to_batch


@dataclass
class EvalResults:
    """Container for all evaluation metrics."""
    auc_roc: float = 0.0
    calibration_error: float = 0.0
    cold_start_auc: float = 0.0
    adaptation_speed: int = 0         # Reviews to match FSRS-6 AUC
    retention_30d: float = 0.0
    rmse_stability: float = 0.0
    recall_loss: float = 0.0
    n_samples: int = 0

    # Per-phase breakdowns
    phase1_auc: float = 0.0           # 0 reviews
    phase2_auc: float = 0.0           # 5-50 reviews
    phase3_auc: float = 0.0           # 50+ reviews

    def summary(self) -> str:
        target_met = lambda val, target, higher_better=True: (
            "✓" if (val > target if higher_better else val < target) else "✗"
        )
        return (
            f"=== MetaSRS Evaluation Results ===\n"
            f"  AUC-ROC:           {self.auc_roc:.4f}  {target_met(self.auc_roc, 0.80)} (target >0.80)\n"
            f"  Calibration Error: {self.calibration_error:.4f}  {target_met(self.calibration_error, 0.05, False)} (target <0.05)\n"
            f"  Cold-Start AUC:    {self.cold_start_auc:.4f}  {target_met(self.cold_start_auc, 0.73)} (target >0.73)\n"
            f"  Adaptation Speed:  {self.adaptation_speed} reviews  {target_met(self.adaptation_speed, 30, False)} (target <30)\n"
            f"  30-day Retention:  {self.retention_30d:.2%}  {target_met(self.retention_30d, 0.85)} (target >85%)\n"
            f"  RMSE (stability):  {self.rmse_stability:.4f} days  {target_met(self.rmse_stability, 2.0, False)} (target <2.0)\n"
            f"  Samples evaluated: {self.n_samples}\n"
        )


class MetaSRSEvaluator:
    """
    Comprehensive evaluator for the MetaSRS system.
    Computes all metrics from Section 7.1 and supports the ablation study.
    """

    def __init__(
        self,
        model: MemoryNet,
        config: Optional[MetaSRSConfig] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model.to(device)
        self.config = config or MetaSRSConfig()
        self.device = device
        self.loss_fn = MetaSRSLoss(w20=self.config.fsrs.w[20])

    def compute_auc_roc(
        self, predictions: np.ndarray, labels: np.ndarray
    ) -> float:
        """
        Compute AUC-ROC for recall predictions.

        Uses a simple implementation to avoid sklearn dependency,
        but falls back to sklearn if available.
        """
        try:
            from sklearn.metrics import roc_auc_score
            return float(roc_auc_score(labels, predictions))
        except ImportError:
            # Manual AUC via trapezoidal rule on sorted predictions
            return self._manual_auc(predictions, labels)

    def _manual_auc(self, scores: np.ndarray, labels: np.ndarray) -> float:
        """Simple AUC computation without sklearn."""
        sorted_idx = np.argsort(-scores)
        sorted_labels = labels[sorted_idx]

        n_pos = labels.sum()
        n_neg = len(labels) - n_pos

        if n_pos == 0 or n_neg == 0:
            return 0.5

        tp = 0
        fp = 0
        auc = 0.0
        prev_fpr = 0.0

        for label in sorted_labels:
            if label == 1:
                tp += 1
            else:
                fp += 1
                tpr = tp / n_pos
                fpr = fp / n_neg
                auc += tpr * (fpr - prev_fpr)
                prev_fpr = fpr

        return auc

    def evaluate_on_tasks(
        self,
        phi: OrderedDict,
        tasks: List[Task],
        k_steps: int = 5,
    ) -> EvalResults:
        """
        Full evaluation on a set of test tasks (students).

        For each student:
        1. Adapt from phi using support set (inner loop).
        2. Evaluate on query set.
        3. Also evaluate cold-start (no adaptation).

        Args:
            phi: Meta-parameters to start from.
            tasks: List of test tasks with support/query splits.
            k_steps: Inner-loop adaptation steps.

        Returns:
            EvalResults with all metrics populated.
        """
        all_preds = []
        all_labels = []
        all_preds_cold = []
        all_labels_cold = []
        all_S_preds = []
        all_S_targets = []
        calibration_errors = []

        for task in tasks:
            if not task.query_set:
                continue

            # Cold-start evaluation (no adaptation, use phi directly)
            self.model.load_state_dict(phi)
            self.model.eval()

            query_batch = reviews_to_batch(
                task.query_set, self.device
            )

            with torch.no_grad():
                state_cold = self.model(
                    D_prev=query_batch["D_prev"],
                    S_prev=query_batch["S_prev"],
                    R_at_review=query_batch["R_at_review"],
                    delta_t=query_batch["delta_t"],
                    grade=query_batch["grade"],
                    review_count=query_batch["review_count"],
                    user_stats=query_batch["user_stats"],
                    history_grades=query_batch.get("history_grades"),
                    history_delta_ts=query_batch.get("history_delta_ts"),
                    history_lengths=query_batch.get("history_lengths"),
                )

            preds_cold = state_cold.p_recall.cpu().numpy()
            labels = query_batch["recalled"].cpu().numpy()
            all_preds_cold.extend(preds_cold)
            all_labels_cold.extend(labels)

            # Adapted evaluation (inner loop on support set)
            adapted = inner_loop(
                phi_state_dict=phi,
                model=self.model,
                task=task,
                loss_fn=self.loss_fn,
                k_steps=k_steps,
                inner_lr=self.config.training.inner_lr,
                batch_size=self.config.training.task_batch_size,
                device=self.device,
            )

            self.model.load_state_dict(adapted)
            self.model.eval()

            with torch.no_grad():
                state = self.model(
                    D_prev=query_batch["D_prev"],
                    S_prev=query_batch["S_prev"],
                    R_at_review=query_batch["R_at_review"],
                    delta_t=query_batch["delta_t"],
                    grade=query_batch["grade"],
                    review_count=query_batch["review_count"],
                    user_stats=query_batch["user_stats"],
                    history_grades=query_batch.get("history_grades"),
                    history_delta_ts=query_batch.get("history_delta_ts"),
                    history_lengths=query_batch.get("history_lengths"),
                )

            preds = state.p_recall.cpu().numpy()
            S_preds = state.S_next.cpu().numpy()
            S_targets = query_batch["S_target"].cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)
            all_S_preds.extend(S_preds)
            all_S_targets.extend(S_targets)

            # Calibration: mean |predicted R - actual recalled|
            cal_err = np.abs(preds - labels).mean()
            calibration_errors.append(cal_err)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_preds_cold = np.array(all_preds_cold)
        all_labels_cold = np.array(all_labels_cold)
        all_S_preds = np.array(all_S_preds)
        all_S_targets = np.array(all_S_targets)

        # Compute metrics (guard against NaN from diverged models)
        n_nan_preds = np.isnan(all_preds).sum()
        n_nan_cold = np.isnan(all_preds_cold).sum()
        n_nan_S = np.isnan(all_S_preds).sum()
        if n_nan_preds + n_nan_cold + n_nan_S > 0:
            import warnings
            warnings.warn(
                f"NaN detected in predictions: {n_nan_preds} preds, "
                f"{n_nan_cold} cold-start preds, {n_nan_S} S predictions. "
                f"Model may have diverged during adaptation."
            )
        all_preds = np.nan_to_num(all_preds, nan=0.5)
        all_preds_cold = np.nan_to_num(all_preds_cold, nan=0.5)
        all_S_preds = np.nan_to_num(all_S_preds, nan=0.0)

        results = EvalResults(
            auc_roc=self.compute_auc_roc(all_preds, all_labels),
            calibration_error=np.mean(calibration_errors) if calibration_errors else 0.0,
            cold_start_auc=self.compute_auc_roc(all_preds_cold, all_labels_cold),
            rmse_stability=float(np.sqrt(np.mean((all_S_preds - all_S_targets) ** 2)))
                if len(all_S_targets) > 0 else 0.0,
            n_samples=len(all_preds),
        )

        return results

    def cold_start_curve(
        self,
        phi: OrderedDict,
        tasks: List[Task],
        review_counts: List[int] = [0, 5, 10, 20, 30, 50],
    ) -> Dict[int, float]:
        """
        Compute AUC at different numbers of adaptation reviews.
        Used to measure adaptation speed.

        Returns dict: {n_reviews → AUC}
        """
        results = {}

        for n_reviews in review_counts:
            all_preds = []
            all_labels = []

            for task in tasks:
                if not task.query_set:
                    continue

                # Create a truncated task with only n_reviews in support set
                truncated_task = Task(
                    student_id=task.student_id,
                    reviews=task.reviews,
                )
                truncated_task.support_set = task.support_set[:n_reviews]
                truncated_task.query_set = task.query_set

                if n_reviews == 0:
                    # Zero-shot: use phi directly
                    self.model.load_state_dict(phi)
                else:
                    # Adapt with n_reviews
                    k = min(5, n_reviews)
                    adapted = inner_loop(
                        phi_state_dict=phi,
                        model=self.model,
                        task=truncated_task,
                        loss_fn=self.loss_fn,
                        k_steps=k,
                        inner_lr=self.config.training.inner_lr,
                        batch_size=min(32, n_reviews),
                        device=self.device,
                    )
                    self.model.load_state_dict(adapted)

                self.model.eval()
                query_batch = reviews_to_batch(
                    task.query_set, self.device
                )

                with torch.no_grad():
                    state = self.model(
                        D_prev=query_batch["D_prev"],
                        S_prev=query_batch["S_prev"],
                        R_at_review=query_batch["R_at_review"],
                        delta_t=query_batch["delta_t"],
                        grade=query_batch["grade"],
                        review_count=query_batch["review_count"],
                        user_stats=query_batch["user_stats"],
                        history_grades=query_batch.get("history_grades"),
                        history_delta_ts=query_batch.get("history_delta_ts"),
                        history_lengths=query_batch.get("history_lengths"),
                    )

                all_preds.extend(state.p_recall.cpu().numpy())
                all_labels.extend(query_batch["recalled"].cpu().numpy())

            if all_preds:
                results[n_reviews] = self.compute_auc_roc(
                    np.array(all_preds), np.array(all_labels)
                )

        return results

    def ablation_study(
        self,
        phi_variants: Dict[str, OrderedDict],
        tasks: List[Task],
    ) -> Dict[str, EvalResults]:
        """
        Run the ablation study from Section 7.2.

        Args:
            phi_variants: Dict mapping variant name → meta-parameters.
                Expected keys: 'A_baseline', 'B_reptile', 'C_content',
                               'D_full', 'E_transformer'
            tasks: Test tasks.

        Returns:
            Dict mapping variant name → EvalResults.
        """
        results = {}
        for name, phi in phi_variants.items():
            print(f"  Evaluating variant: {name}")
            results[name] = self.evaluate_on_tasks(phi, tasks)
            print(f"    AUC: {results[name].auc_roc:.4f}")
        return results
