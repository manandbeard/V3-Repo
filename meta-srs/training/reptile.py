"""
Reptile Meta-Training Loop (Sections 3.2 and 3.4).

Inner Loop (Per-Student Adaptation):
    def inner_loop(phi, task, k_steps=5, inner_lr=0.01):
        theta = copy(phi)
        optimizer = Adam(theta, lr=inner_lr)
        for step in range(k_steps):
            batch = sample_batch(task.support_set, size=32)
            loss = compute_loss(theta, batch)
            loss.backward(); optimizer.step(); optimizer.zero_grad()
        return theta  # adapted parameters W_i

Outer Loop (Meta-Update):
    def meta_train(dataset, n_iters=50_000, meta_lr=1e-3, k=10, epsilon=0.1):
        phi = warm_start_from_fsrs6()
        for iteration in range(n_iters):
            tasks = sample_tasks(dataset, batch_size=16)
            reptile_grad = zeros_like(phi)
            for task in tasks:
                W_i = inner_loop(phi, task, k_steps=k)
                reptile_grad += (W_i - phi)
            phi = phi + epsilon * reptile_grad / len(tasks)
        return phi
"""

import os
import math
import random
import torch
import torch.nn as nn
from copy import deepcopy
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

from config import MetaSRSConfig, TrainingConfig
from models.memory_net import MemoryNet
from data.task_sampler import Task, reviews_to_batch
from training.loss import MetaSRSLoss, compute_loss


def sample_batch(
    reviews: list,
    size: int,
    card_embeddings: dict,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Sample a mini-batch of reviews from a support/query set."""
    if len(reviews) <= size:
        sampled = reviews
    else:
        sampled = random.sample(reviews, size)
    return reviews_to_batch(sampled, card_embeddings, device)


def inner_loop(
    phi_state_dict: OrderedDict,
    model: MemoryNet,
    task: Task,
    loss_fn: MetaSRSLoss,
    k_steps: int = 5,
    inner_lr: float = 0.01,
    batch_size: int = 32,
    device: torch.device = torch.device("cpu"),
) -> OrderedDict:
    """
    Per-student inner-loop adaptation (Section 3.2).

    Args:
        phi_state_dict: Current meta-parameters (not modified).
        model: MemoryNet architecture (weights will be overwritten).
        task: One student's review history with support/query sets.
        loss_fn: Multi-component loss function.
        k_steps: Number of gradient steps.
        inner_lr: Adam learning rate.
        batch_size: Reviews per mini-batch.
        device: Computation device.

    Returns:
        Adapted parameters W_i (state_dict).
    """
    # Copy meta-parameters — do not modify phi in-place
    model.load_state_dict(deepcopy(phi_state_dict))
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=inner_lr)

    for step in range(k_steps):
        batch = sample_batch(
            task.support_set, batch_size, task.card_embeddings, device
        )
        losses = compute_loss(model, batch, loss_fn)
        loss = losses["total"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return deepcopy(model.state_dict())


def reptile_update(
    phi: OrderedDict,
    adapted_weights: List[OrderedDict],
    epsilon: float,
) -> OrderedDict:
    """
    Reptile outer-loop update: nudge phi toward adapted weights.

    phi = phi + epsilon * mean(W_i - phi)

    Mathematical insight: this update maximises the inner product between
    gradients from different minibatches of the same task, improving
    within-task generalisation — equivalent to a first-order MAML update.
    """
    n_tasks = len(adapted_weights)
    new_phi = OrderedDict()

    for key in phi:
        # Accumulate (W_i - phi) across all tasks
        reptile_grad = torch.zeros_like(phi[key])
        for W_i in adapted_weights:
            reptile_grad += (W_i[key] - phi[key])
        reptile_grad /= n_tasks

        # Nudge phi toward adapted weights
        new_phi[key] = phi[key] + epsilon * reptile_grad

    return new_phi


class ReptileTrainer:
    """
    Full Reptile meta-training orchestrator.

    Implements the outer loop from Section 3.4 with:
    - Cosine LR decay for the outer optimizer
    - Linear epsilon decay
    - Periodic evaluation and checkpointing
    - TensorBoard logging
    """

    def __init__(
        self,
        model: MemoryNet,
        config: MetaSRSConfig,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model.to(device)
        self.config = config
        self.cfg = config.training
        self.device = device

        self.loss_fn = MetaSRSLoss(
            stability_weight=self.cfg.stability_loss_weight,
            monotonicity_weight=self.cfg.monotonicity_loss_weight,
            w20=config.fsrs.w[20],
        )

        # The meta-parameters phi
        self.phi = deepcopy(model.state_dict())

        # Optional: outer Adam optimizer on phi (alternative to pure Reptile)
        # For now we use pure Reptile as in the research
        self.writer = None  # TensorBoard writer, set up in train()

    def train(
        self,
        task_sampler,
        n_iters: Optional[int] = None,
        eval_fn=None,
        resume_from: Optional[str] = None,
    ) -> OrderedDict:
        """
        Run the full Reptile meta-training loop.

        Args:
            task_sampler: TaskSampler instance.
            n_iters: Override number of iterations.
            eval_fn: Optional callback(model, iteration) for evaluation.
            resume_from: Path to checkpoint to resume from.

        Returns:
            Final meta-parameters phi*.
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(self.cfg.log_dir, exist_ok=True)
            self.writer = SummaryWriter(self.cfg.log_dir)
        except ImportError:
            self.writer = None

        n_iters = n_iters or self.cfg.n_iters
        start_iter = 0

        if resume_from and os.path.exists(resume_from):
            checkpoint = torch.load(resume_from, map_location=self.device, weights_only=False)
            self.phi = checkpoint["phi"]
            start_iter = checkpoint["iteration"] + 1
            print(f"Resumed from iteration {start_iter}")

        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)

        print(f"Starting Reptile meta-training for {n_iters} iterations...")
        print(f"  Meta batch size: {self.cfg.meta_batch_size}")
        print(f"  Inner steps: {self.cfg.inner_steps_phase1}")
        print(f"  Inner LR: {self.cfg.inner_lr}")
        print(f"  Model params: {self.model.count_parameters():,}")

        for iteration in range(start_iter, n_iters):
            # Schedule epsilon (linear decay)
            epsilon = self.cfg.epsilon_schedule(iteration)

            # Sample tasks (students) for this meta-iteration
            tasks = task_sampler.sample(self.cfg.meta_batch_size)

            # Inner loop: adapt to each student
            adapted_weights = []
            inner_losses = []

            for task in tasks:
                # Run inner loop
                W_i = inner_loop(
                    phi_state_dict=self.phi,
                    model=self.model,
                    task=task,
                    loss_fn=self.loss_fn,
                    k_steps=self.cfg.inner_steps_phase1,
                    inner_lr=self.cfg.inner_lr,
                    batch_size=self.cfg.task_batch_size,
                    device=self.device,
                )
                adapted_weights.append(W_i)

                # Evaluate adapted model on query set for monitoring
                if task.query_set:
                    self.model.load_state_dict(W_i)
                    self.model.eval()
                    with torch.no_grad():
                        query_batch = sample_batch(
                            task.query_set, 64, task.card_embeddings, self.device
                        )
                        qloss = compute_loss(self.model, query_batch, self.loss_fn)
                        inner_losses.append(qloss["total"].item())

            # Reptile update: phi ← phi + epsilon * mean(W_i - phi)
            self.phi = reptile_update(self.phi, adapted_weights, epsilon)

            # Logging
            if iteration % self.cfg.log_every == 0:
                avg_qloss = sum(inner_losses) / max(len(inner_losses), 1)
                print(
                    f"  iter {iteration:6d}/{n_iters}  "
                    f"eps={epsilon:.4f}  "
                    f"query_loss={avg_qloss:.4f}"
                )
                if self.writer:
                    self.writer.add_scalar("meta/query_loss", avg_qloss, iteration)
                    self.writer.add_scalar("meta/epsilon", epsilon, iteration)

            # Evaluation
            if eval_fn and iteration % self.cfg.eval_every == 0:
                self.model.load_state_dict(self.phi)
                eval_fn(self.model, iteration)

            # Checkpointing
            if iteration % self.cfg.save_every == 0 and iteration > 0:
                ckpt_path = os.path.join(
                    self.cfg.checkpoint_dir, f"phi_iter_{iteration}.pt"
                )
                torch.save({
                    "phi": self.phi,
                    "iteration": iteration,
                    "config": self.config,
                }, ckpt_path)
                print(f"  Checkpoint saved: {ckpt_path}")

        # Save final phi*
        final_path = os.path.join(self.cfg.checkpoint_dir, "phi_star.pt")
        torch.save({
            "phi": self.phi,
            "iteration": n_iters,
            "config": self.config,
        }, final_path)
        print(f"\nMeta-training complete. Final phi* saved to {final_path}")

        if self.writer:
            self.writer.close()

        return self.phi
