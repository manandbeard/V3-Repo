#!/usr/bin/env python3
"""
MetaSRS — Main Training Script.

Orchestrates the full training pipeline:
    1. Generate or load review data
    2. Warm-start MemoryNet from FSRS-6 predictions
    3. Run Reptile meta-training
    4. Evaluate and save phi*

Usage:
    # Quick test with synthetic data:
    python train.py --synthetic --n-students 100 --n-iters 500

    # Full training with real data:
    python train.py --data reviews.csv --n-iters 50000

    # Resume from checkpoint:
    python train.py --data reviews.csv --resume checkpoints/phi_iter_10000.pt
"""

import os
import sys
import argparse
import torch
import random
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MetaSRSConfig
from models.memory_net import MemoryNet
from data.task_sampler import TaskSampler, ReviewDataset
from training.reptile import ReptileTrainer
from training.loss import MetaSRSLoss
from evaluation.metrics import MetaSRSEvaluator


def set_seed(seed: int):
    """Reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="MetaSRS — Meta-Initialized Spaced Repetition Scheduler")

    # Data
    parser.add_argument("--data", type=str, default=None, help="Path to review CSV file")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for testing")
    parser.add_argument("--n-students", type=int, default=500, help="Number of synthetic students")

    # Training
    parser.add_argument("--n-iters", type=int, default=None, help="Override number of meta-iterations")
    parser.add_argument("--inner-steps", type=int, default=None, help="Override inner-loop steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Override meta batch size")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--skip-warmstart", action="store_true", help="Skip FSRS-6 warm-start")

    # System
    parser.add_argument("--device", type=str, default="auto", help="Device: cpu, cuda, auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs")

    # Evaluation
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    parser.add_argument("--eval-checkpoint", type=str, default=None, help="Checkpoint for eval-only mode")

    return parser.parse_args()


def main():
    args = parse_args()
    config = MetaSRSConfig()

    # Apply CLI overrides
    if args.n_iters:
        config.training.n_iters = args.n_iters
    if args.inner_steps:
        config.training.inner_steps_phase1 = args.inner_steps
    if args.batch_size:
        config.training.meta_batch_size = args.batch_size
    config.training.checkpoint_dir = args.checkpoint_dir
    config.training.log_dir = args.log_dir
    config.training.seed = args.seed

    set_seed(args.seed)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ──────────────────────────────────────────────────
    # 1. Load or generate data
    # ──────────────────────────────────────────────────
    print("\n=== Step 1: Loading Data ===")

    if args.synthetic:
        print(f"Generating synthetic data: {args.n_students} students...")
        all_tasks = ReviewDataset.generate_synthetic(
            n_students=args.n_students,
            reviews_per_student=100,
            seed=args.seed,
        )
    elif args.data:
        print(f"Loading data from {args.data}...")
        all_tasks = ReviewDataset.from_csv(args.data)
    else:
        print("No data specified. Use --synthetic or --data. Defaulting to synthetic.")
        all_tasks = ReviewDataset.generate_synthetic(
            n_students=500, reviews_per_student=100, seed=args.seed
        )

    # Train/test split (80/20 by students)
    random.shuffle(all_tasks)
    split_idx = int(len(all_tasks) * 0.8)
    train_tasks = all_tasks[:split_idx]
    test_tasks = all_tasks[split_idx:]

    print(f"  Train: {len(train_tasks)} students")
    print(f"  Test:  {len(test_tasks)} students")

    # Build task sampler
    sampler = TaskSampler(
        train_tasks,
        support_ratio=config.training.support_ratio,
        seed=args.seed,
    )

    # Split test tasks
    for task in test_tasks:
        task.split(config.training.support_ratio)

    # ──────────────────────────────────────────────────
    # 2. Create model
    # ──────────────────────────────────────────────────
    print("\n=== Step 2: Creating MemoryNet ===")

    model = MemoryNet(
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        gru_hidden_dim=config.model.gru_hidden_dim,
        user_stats_dim=config.model.user_stats_dim,
        dropout=config.model.dropout,
        history_len=config.model.history_len,
    ).to(device)

    n_params = model.count_parameters()
    print(f"  Parameters: {n_params:,} (target: ~50K)")

    # ──────────────────────────────────────────────────
    # 3. Evaluation only mode
    # ──────────────────────────────────────────────────
    if args.eval_only:
        ckpt_path = args.eval_checkpoint or os.path.join(args.checkpoint_dir, "phi_star.pt")
        print(f"\n=== Evaluation Only: loading {ckpt_path} ===")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        phi = checkpoint["phi"]

        evaluator = MetaSRSEvaluator(model, config, device)
        results = evaluator.evaluate_on_tasks(phi, test_tasks)
        print(results.summary())

        print("\n--- Cold-Start Curve ---")
        curve = evaluator.cold_start_curve(phi, test_tasks)
        for n, auc in sorted(curve.items()):
            print(f"  {n:3d} reviews → AUC = {auc:.4f}")
        return

    # ──────────────────────────────────────────────────
    # 4. (Optional) Warm-start from FSRS-6
    # ──────────────────────────────────────────────────
    if not args.skip_warmstart and not args.resume:
        print("\n=== Step 3: FSRS-6 Warm-Start ===")
        print("  Pre-training MemoryNet to reproduce FSRS-6 predictions...")
        print("  (This grounds meta-init in a cognitively-valid baseline)")

        from training.fsrs_warmstart import warm_start_from_fsrs6, FSRS6
        from data.task_sampler import reviews_to_batch

        # Generate FSRS-6 targets from training data
        fsrs = FSRS6(config.fsrs)
        warmstart_batches = []

        for task in train_tasks[:200]:  # Use subset for speed
            batch = reviews_to_batch(
                task.reviews, device
            )
            features = model.build_features(
                batch["D_prev"], batch["S_prev"], batch["R_at_review"],
                batch["delta_t"], batch["grade"], batch["review_count"],
                batch["user_stats"],
            )
            targets = {
                "S_target": batch["S_target"],
                "D_target": batch["D_target"],
                "R_target": batch["recalled"],
            }
            warmstart_batches.append((features.detach(), batch["S_prev"], targets))

        warm_start_from_fsrs6(model, warmstart_batches, config.training, device=str(device))
        print("  Warm-start complete!")
    else:
        if args.resume:
            print("\n=== Skipping warm-start (resuming from checkpoint) ===")
        else:
            print("\n=== Skipping warm-start (--skip-warmstart) ===")

    # ──────────────────────────────────────────────────
    # 5. Reptile meta-training
    # ──────────────────────────────────────────────────
    print("\n=== Step 4: Reptile Meta-Training ===")

    trainer = ReptileTrainer(model, config, device)

    # Evaluation callback
    evaluator = MetaSRSEvaluator(model, config, device)

    def eval_callback(m, iteration):
        results = evaluator.evaluate_on_tasks(
            trainer.phi, test_tasks[:50],  # Subset for speed
            k_steps=config.training.inner_steps_phase1,
        )
        print(f"  [Eval @ iter {iteration}] AUC={results.auc_roc:.4f}  "
              f"CalErr={results.calibration_error:.4f}  "
              f"RMSE_S={results.rmse_stability:.4f}")
        if trainer.writer:
            trainer.writer.add_scalar("eval/auc_roc", results.auc_roc, iteration)
            trainer.writer.add_scalar("eval/calibration_error", results.calibration_error, iteration)
            trainer.writer.add_scalar("eval/rmse_stability", results.rmse_stability, iteration)

    phi_star = trainer.train(
        task_sampler=sampler,
        eval_fn=eval_callback,
        resume_from=args.resume,
    )

    # ──────────────────────────────────────────────────
    # 6. Final evaluation
    # ──────────────────────────────────────────────────
    print("\n=== Step 5: Final Evaluation ===")

    results = evaluator.evaluate_on_tasks(phi_star, test_tasks)
    print(results.summary())

    print("--- Cold-Start Adaptation Curve ---")
    curve = evaluator.cold_start_curve(phi_star, test_tasks)
    for n, auc in sorted(curve.items()):
        marker = " ← FSRS-6 baseline" if n == 0 else ""
        print(f"  {n:3d} reviews → AUC = {auc:.4f}{marker}")

    print("\nDone! 🎓")
    print(f"  Meta-parameters saved to: {config.training.checkpoint_dir}/phi_star.pt")
    print(f"  TensorBoard logs: {config.training.log_dir}/")


if __name__ == "__main__":
    main()
