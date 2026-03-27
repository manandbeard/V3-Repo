"""
MetaSRS Configuration — All hyperparameters from Section 6.3.
"""

from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class ModelConfig:
    """MemoryNet architecture (Section 2.2)."""
    input_dim: int = 112          # Feature vector dimension
    hidden_dim: int = 128         # Hidden layer width
    card_embed_dim: int = 64      # Projected card embedding dimension
    card_raw_dim: int = 384       # Raw BERT embedding dimension (all-MiniLM-L6-v2)
    gru_hidden_dim: int = 32      # GRU history encoder hidden state
    history_len: int = 32         # Max review history length for GRU
    user_stats_dim: int = 8       # User-level statistics features
    dropout: float = 0.1          # Dropout rate (used for MC Dropout uncertainty)
    mc_samples: int = 20          # Monte Carlo forward passes for uncertainty


@dataclass
class FSRSConfig:
    """FSRS-6 default population parameters (Section 1.1)."""
    # 21 optimizable weights (w0..w20), population defaults from FSRS-6
    w: list = field(default_factory=lambda: [
        0.40255,   # w0  - initial stability (Again)
        1.18385,   # w1  - initial stability (Hard)
        3.173,     # w2  - initial stability (Good)
        15.69105,  # w3  - initial stability (Easy)
        7.1949,    # w4  - difficulty baseline
        0.5345,    # w5  - difficulty mean-reversion weight
        1.4604,    # w6  - difficulty update scaling
        0.0046,    # w7  - (unused placeholder)
        1.54575,   # w8  - stability growth factor (success)
        0.1740,    # w9  - stability decay factor (success)
        1.01925,   # w10 - retrievability bonus factor (success)
        1.9395,    # w11 - lapse stability factor
        0.11,      # w12 - lapse difficulty exponent
        0.29605,   # w13 - lapse stability exponent
        2.2698,    # w14 - lapse retrievability factor
        0.2315,    # w15 - Hard grade modifier
        2.9898,    # w16 - Easy grade modifier
        0.51655,   # w17 - (short-term scheduling)
        0.6621,    # w18 - (short-term scheduling)
        0.0600,    # w19 - (short-term scheduling)
        0.4665,    # w20 - power-law exponent for forgetting curve
    ])


@dataclass
class TrainingConfig:
    """Reptile meta-training hyperparameters (Section 6.3)."""
    # Outer loop
    n_iters: int = 50_000                 # Total meta-iterations
    meta_batch_size: int = 16             # Tasks (students) per outer update
    outer_lr_start: float = 1e-3          # Outer Adam LR (cosine decay)
    outer_lr_end: float = 1e-4
    epsilon_start: float = 0.10           # Reptile step size
    epsilon_end: float = 0.01

    # Inner loop
    inner_lr: float = 0.01               # Per-student Adam LR
    inner_steps_phase1: int = 5           # k steps for rapid adaptation
    inner_steps_phase2: int = 20          # k steps for full personalization
    task_batch_size: int = 32             # Reviews per inner-loop mini-batch
    support_ratio: float = 0.70           # Fraction for inner-loop training

    # Loss weights (Section 3.3)
    stability_loss_weight: float = 0.10
    monotonicity_loss_weight: float = 0.01

    # Warm-start
    warmstart_epochs: int = 10            # Pre-train on FSRS-6 predictions
    warmstart_lr: float = 1e-3

    # General
    seed: int = 42
    log_every: int = 100
    eval_every: int = 1_000
    save_every: int = 5_000
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    def epsilon_schedule(self, iteration: int) -> float:
        """Linear decay from epsilon_start to epsilon_end."""
        progress = min(iteration / self.n_iters, 1.0)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress

    def outer_lr_schedule(self, iteration: int) -> float:
        """Cosine decay from outer_lr_start to outer_lr_end."""
        progress = min(iteration / self.n_iters, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.outer_lr_end + (self.outer_lr_start - self.outer_lr_end) * cosine


@dataclass
class AdaptationConfig:
    """Online fast-adaptation settings (Section 4.1)."""
    phase1_threshold: int = 5             # Reviews before first adaptation
    phase2_threshold: int = 50            # Reviews before full personalization
    phase1_k_steps: int = 5
    phase2_k_steps: int = 10
    phase3_k_steps: int = 20
    phase3_inner_lr: float = 0.02         # Higher LR for full personalization
    adapt_every_n_reviews: int = 5        # Re-adapt frequency in phase 2
    streaming_after: int = 50             # Enable per-review gradient steps


@dataclass
class SchedulingConfig:
    """Interval computation settings (Section 4.2)."""
    desired_retention: float = 0.90
    min_interval: int = 1                 # Days
    max_interval: int = 365               # Days
    fuzz_range: tuple = (0.95, 1.05)      # Prevent card clustering
    difficulty_centre: float = 5.0        # D reference point for discounting
    difficulty_slope: float = 0.05        # Interval penalty per unit of D
    uncertainty_discount: float = 0.5     # Max reduction from high uncertainty


@dataclass
class MetaSRSConfig:
    """Top-level config aggregating all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    fsrs: FSRSConfig = field(default_factory=FSRSConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    scheduling: SchedulingConfig = field(default_factory=SchedulingConfig)
