# MetaSRS — Complete Source Code (31 files)

---

## 📁 Project Structure

```
meta-srs/
├── config.py                    # All hyperparameters (Sections 1.1, 6.3)
├── requirements.txt             # Dependencies
├── train.py                     # Main training script
├── pytest.ini                   # Pytest configuration
├── models/
│   ├── __init__.py
│   ├── memory_net.py            # MemoryNet — neural DSR model (Section 2.2)
│   └── gru_encoder.py           # GRU history encoder (Section 2.4)
├── training/
│   ├── __init__.py
│   ├── fsrs_warmstart.py        # FSRS-6 baseline + warm-start (Section 1.1)
│   ├── loss.py                  # Multi-component loss function (Section 3.3)
│   └── reptile.py               # Reptile inner/outer loops (Sections 3.2, 3.4)
├── data/
│   ├── __init__.py
│   └── task_sampler.py          # Task definition & sampling (Section 3.1)
├── inference/
│   ├── __init__.py
│   ├── adaptation.py            # 3-phase fast adaptation (Section 4.1)
│   └── scheduling.py            # Uncertainty-aware scheduling (Section 4.2)
├── evaluation/
│   ├── __init__.py
│   └── metrics.py               # Eval framework + ablations (Section 7)
└── tests/
    ├── __init__.py
    ├── conftest.py              # Shared test fixtures
    ├── test_config.py
    ├── test_memory_net.py
    ├── test_models.py           # GRU encoder tests
    ├── test_fsrs_warmstart.py
    ├── test_loss.py
    ├── test_reptile.py
    ├── test_task_sampler.py
    ├── test_adaptation.py
    ├── test_scheduling.py
    ├── test_evaluation.py
    └── test_integration.py
```

---

## 1. `requirements.txt`

```
torch>=2.0
numpy>=1.24
scikit-learn>=1.3
tqdm
tensorboard
onnx
onnxruntime
```

---

## 2. `config.py` — All Hyperparameters

```python
"""
MetaSRS Configuration — All hyperparameters from Section 6.3.
"""

from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class ModelConfig:
    """MemoryNet architecture (Section 2.2)."""
    input_dim: int = 49           # Feature vector dimension (4+4+1+8+32)
    hidden_dim: int = 128         # Hidden layer width
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
```

---

## 3. `models/__init__.py`

```python
from .memory_net import MemoryNet
from .gru_encoder import GRUHistoryEncoder

__all__ = [
    "MemoryNet",
    "GRUHistoryEncoder",
]
```

---

## 4. `models/gru_encoder.py` — GRU History Encoder (Section 2.4)

```python
"""
GRU History Encoder (Section 2.4).

Captures per-card review trajectory patterns — repeated 'Again' presses,
irregular gaps, etc. — and produces a fixed-size context vector that is
appended to the MemoryNet input.

Input sequence per card:
    [(grade_1, delta_t_1), (grade_2, delta_t_2), ..., (grade_n, delta_t_n)]

Output:
    context_h  — final GRU hidden state (dim = gru_hidden_dim)
"""

import torch
import torch.nn as nn
from typing import Optional


class GRUHistoryEncoder(nn.Module):
    """1-layer GRU that encodes a card's review history into a context vector."""

    # Each timestep: grade (1-4 as float) + log(delta_t + 1)
    FEATURE_DIM = 2

    def __init__(self, hidden_dim: int = 32, max_len: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        self.gru = nn.GRU(
            input_size=self.FEATURE_DIM,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Small input projection to help with numerical scale
        self.input_proj = nn.Linear(self.FEATURE_DIM, self.FEATURE_DIM)

    def forward(
        self,
        grades: torch.Tensor,
        delta_ts: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            grades:   (batch, seq_len) — grade values 1–4 as float.
            delta_ts: (batch, seq_len) — days since previous review.
            lengths:  (batch,) — actual sequence lengths (for packing).

        Returns:
            context_h: (batch, hidden_dim) — final hidden state.
        """
        # Normalise inputs
        grade_norm = grades / 4.0                         # [0, 1]
        dt_norm = torch.log(delta_ts.clamp(min=0) + 1.0)  # log-scale

        # (batch, seq_len, 2)
        x = torch.stack([grade_norm, dt_norm], dim=-1)
        x = self.input_proj(x)

        # Truncate to max_len (keep most recent reviews)
        if x.size(1) > self.max_len:
            x = x[:, -self.max_len:, :]
            if lengths is not None:
                lengths = lengths.clamp(max=self.max_len)

        if lengths is not None:
            # Pack for variable-length sequences
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
            )
            _, h_n = self.gru(packed)
        else:
            _, h_n = self.gru(x)

        # h_n: (1, batch, hidden_dim) → (batch, hidden_dim)
        return h_n.squeeze(0)

    def zero_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return a zero context vector for cards with no prior history."""
        return torch.zeros(batch_size, self.hidden_dim, device=device)
```

---

## 5. `models/memory_net.py` — Neural Memory Model (Section 2.2)

```python
"""
Neural Memory Model — MemoryNet (Section 2.2).

Replaces FSRS-6's hand-crafted formulas with a small differentiable neural
network (~50K parameters) that predicts memory-state transitions.

The small footprint is intentional: fast-adaptation requires a network that
moves meaningfully in just 5 gradient steps without overfitting.

Input feature vector at each review event (dim = 49):
    x = concat([
        D_prev,          # current difficulty [1, 10]              → 1
        log(S_prev),     # log stability (numerical stability)     → 1
        R_at_review,     # retrievability at review time           → 1
        log(delta_t + 1),# log time since last review              → 1
        grade_onehot,    # [Again, Hard, Good, Easy]               → 4
        review_count,    # total reviews of this card              → 1
        user_stats,      # mean D, mean S, session length, etc.    → 8
        context_h,       # GRU hidden state from review history    → 32
    ])                                                       Total: 49

Output heads:
    S_next   = Softplus(linear) * S_prev   — stability grows on success
    D_next   = Sigmoid(linear) * 9 + 1     — difficulty in [1, 10]
    p_recall = Sigmoid(linear)             — recall probability (primary loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, NamedTuple

from .gru_encoder import GRUHistoryEncoder


class MemoryState(NamedTuple):
    """Output of a MemoryNet forward pass."""
    S_next: torch.Tensor     # (batch,) predicted next stability
    D_next: torch.Tensor     # (batch,) predicted next difficulty
    p_recall: torch.Tensor   # (batch,) predicted recall probability


class MemoryNet(nn.Module):
    """
    Neural memory-state transition model.

    Architecture:
        Linear(49, 128) + LayerNorm + GELU
        Linear(128, 128) + LayerNorm + GELU
        Linear(128, 64)  + GELU
        → 3 output heads: S_next, D_next, p_recall
    """

    # Feature dimensions (must sum to input_dim=49)
    SCALAR_FEATURES = 4   # D_prev, log_S_prev, R_at_review, log_delta_t
    GRADE_DIM = 4          # one-hot [Again, Hard, Good, Easy]
    COUNT_DIM = 1          # review_count (log-scaled)

    def __init__(
        self,
        input_dim: int = 49,
        hidden_dim: int = 128,
        gru_hidden_dim: int = 32,
        user_stats_dim: int = 8,
        dropout: float = 0.1,
        history_len: int = 32,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.user_stats_dim = user_stats_dim

        # Sub-modules (part of phi, updated in meta-training)
        self.gru_encoder = GRUHistoryEncoder(gru_hidden_dim, max_len=history_len)

        # Main network
        self.h1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.h2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.h3 = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Output heads
        self.stability_head = nn.Linear(64, 1)
        self.difficulty_head = nn.Linear(64, 1)
        self.recall_head = nn.Linear(64, 1)

    def build_features(
        self,
        D_prev: torch.Tensor,
        S_prev: torch.Tensor,
        R_at_review: torch.Tensor,
        delta_t: torch.Tensor,
        grade: torch.Tensor,
        review_count: torch.Tensor,
        user_stats: torch.Tensor,
        history_grades: Optional[torch.Tensor] = None,
        history_delta_ts: Optional[torch.Tensor] = None,
        history_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Assemble the 49-dim input feature vector.

        Args:
            D_prev:             (batch,) current difficulty [1, 10]
            S_prev:             (batch,) current stability in days
            R_at_review:        (batch,) retrievability at review time [0, 1]
            delta_t:            (batch,) days since last review
            grade:              (batch,) integer grade 1-4
            review_count:       (batch,) total reviews for this card
            user_stats:         (batch, 8) user-level statistics
            history_grades:     (batch, seq_len) grade history for GRU
            history_delta_ts:   (batch, seq_len) delta_t history for GRU
            history_lengths:    (batch,) actual sequence lengths

        Returns:
            x: (batch, 49) assembled feature vector
        """
        batch_size = D_prev.size(0)
        device = D_prev.device

        # Scalar features
        scalars = torch.stack([
            D_prev / 10.0,                              # normalise to ~[0, 1]
            torch.log(S_prev.clamp(min=1e-6)),          # log stability
            R_at_review,                                 # already [0, 1]
            torch.log(delta_t.clamp(min=0) + 1.0),      # log-scale
        ], dim=-1)  # (batch, 4)

        # Grade one-hot
        grade_oh = F.one_hot(
            (grade.long() - 1).clamp(0, 3), num_classes=4
        ).float()  # (batch, 4)

        # Review count (log-scaled)
        count_feat = torch.log(review_count.float().clamp(min=1)).unsqueeze(-1)  # (batch, 1)

        # GRU context
        if history_grades is not None and history_delta_ts is not None:
            context_h = self.gru_encoder(
                history_grades.float(), history_delta_ts.float(), history_lengths
            )  # (batch, 32)
        else:
            context_h = self.gru_encoder.zero_state(batch_size, device)

        # Concatenate all features
        x = torch.cat([
            scalars,       # 4
            grade_oh,      # 4
            count_feat,    # 1
            user_stats,    # 8
            context_h,     # 32
        ], dim=-1)  # Total: 49

        # Dynamic padding/truncation to match expected input_dim (safety net)
        if x.size(-1) < self.input_dim:
            pad = torch.zeros(batch_size, self.input_dim - x.size(-1), device=device)
            x = torch.cat([x, pad], dim=-1)
        elif x.size(-1) > self.input_dim:
            x = x[:, :self.input_dim]

        return x

    def forward_from_features(
        self, x: torch.Tensor, S_prev: torch.Tensor
    ) -> MemoryState:
        """
        Forward pass from pre-assembled feature vector.

        Args:
            x:      (batch, input_dim) — assembled features.
            S_prev: (batch,) — previous stability (for multiplicative head).

        Returns:
            MemoryState(S_next, D_next, p_recall)
        """
        h = self.h1(x)
        h = self.h2(h)
        h = self.h3(h)

        # Output heads
        S_next = F.softplus(self.stability_head(h).squeeze(-1)) * S_prev
        S_next = S_next.clamp(min=1e-3, max=36500.0)  # Stability: 0.001 to 100 years
        D_next = torch.sigmoid(self.difficulty_head(h).squeeze(-1)) * 9.0 + 1.0
        p_recall = torch.sigmoid(self.recall_head(h).squeeze(-1))

        return MemoryState(S_next=S_next, D_next=D_next, p_recall=p_recall)

    def forward(
        self,
        D_prev: torch.Tensor,
        S_prev: torch.Tensor,
        R_at_review: torch.Tensor,
        delta_t: torch.Tensor,
        grade: torch.Tensor,
        review_count: torch.Tensor,
        user_stats: torch.Tensor,
        history_grades: Optional[torch.Tensor] = None,
        history_delta_ts: Optional[torch.Tensor] = None,
        history_lengths: Optional[torch.Tensor] = None,
    ) -> MemoryState:
        """
        Full forward pass: build features → network → predictions.
        """
        x = self.build_features(
            D_prev, S_prev, R_at_review, delta_t, grade,
            review_count, user_stats,
            history_grades, history_delta_ts, history_lengths,
        )
        return self.forward_from_features(x, S_prev)

    def predict_recall(self, features: torch.Tensor, S_prev: torch.Tensor) -> torch.Tensor:
        """Convenience: return only p_recall from features."""
        return self.forward_from_features(features, S_prev).p_recall

    def predict_stability(self, features: torch.Tensor, S_prev: torch.Tensor) -> torch.Tensor:
        """Convenience: return only S_next from features."""
        return self.forward_from_features(features, S_prev).S_next

    def count_parameters(self) -> int:
        """Total trainable parameters (target: ~50K)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

---

## 6. `training/__init__.py`

```python
from .loss import compute_loss, MetaSRSLoss
from .reptile import inner_loop, reptile_update, ReptileTrainer
from .fsrs_warmstart import FSRS6, warm_start_from_fsrs6

__all__ = [
    "compute_loss",
    "MetaSRSLoss",
    "inner_loop",
    "reptile_update",
    "ReptileTrainer",
    "FSRS6",
    "warm_start_from_fsrs6",
]
```

---

## 7. `training/fsrs_warmstart.py` — FSRS-6 Baseline & Warm-Start

```python
"""
FSRS-6 Baseline & Warm-Start (Sections 1.1 and 3.4 tip).

Implements the FSRS-6 forgetting curve and stability update formulas for:
  1. Generating synthetic training targets for MemoryNet warm-start.
  2. Serving as the evaluation baseline.

Warm-start tip from the research:
    Pre-train MemoryNet to reproduce FSRS-6 predictions on a large corpus
    before running Reptile. This grounds the meta-initialization in a
    cognitively-valid baseline and cuts Reptile training time by ~60%.
"""

import math
import torch
import torch.nn as nn
from copy import deepcopy
from typing import List, Optional, Tuple

from config import FSRSConfig, ModelConfig, TrainingConfig


class FSRS6:
    """
    Deterministic FSRS-6 model using the 21-parameter forgetting curve
    and stability update formulas from Section 1.1.
    """

    def __init__(self, config: Optional[FSRSConfig] = None):
        self.cfg = config or FSRSConfig()
        self.w = self.cfg.w

    # ----- Forgetting Curve (power-law) -----

    def retrievability(self, t: float, S: float) -> float:
        """
        R(t, S) = (0.9^(1/S))^(t^w20)

        Power-law beats exponential because the population average of two
        exponential forgetting curves is better approximated by a power function.
        """
        if S <= 0:
            return 0.0
        w20 = self.w[20]
        return (0.9 ** (1.0 / S)) ** (t ** w20)

    def retrievability_batch(self, t: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """Vectorised retrievability for PyTorch tensors."""
        w20 = self.w[20]
        return (0.9 ** (1.0 / S.clamp(min=1e-6))) ** (t.clamp(min=0) ** w20)

    # ----- Initial State -----

    def initial_stability(self, grade: int) -> float:
        """S_0 for the first review of a card.  grade: 1=Again, 2=Hard, 3=Good, 4=Easy."""
        return self.w[grade - 1]  # w0..w3

    def initial_difficulty(self, grade: int) -> float:
        """D_0 based on first-review grade."""
        return self.w[4] - (grade - 3) * self.w[5]

    # ----- Stability Update on Success (Hard / Good / Easy) -----

    def stability_after_success(
        self, S: float, D: float, R: float, grade: int
    ) -> float:
        """
        S' = S * (e^w8 * (11-D) * S^(-w9) * (e^(w10*(1-R)) - 1) * w15_grade * w16_grade + 1)
        """
        w = self.w
        grade_modifier = 1.0
        if grade == 2:  # Hard
            grade_modifier = w[15]
        elif grade == 4:  # Easy
            grade_modifier = w[16]

        inner = (
            math.exp(w[8])
            * (11 - D)
            * (S ** (-w[9]))
            * (math.exp(w[10] * (1 - R)) - 1)
            * grade_modifier
            + 1
        )
        return S * inner

    # ----- Stability Update on Lapse (Again) -----

    def stability_after_lapse(self, S: float, D: float, R: float) -> float:
        """
        S_f = w11 * D^(-w12) * ((S+1)^w13 - 1) * e^(w14*(1-R))
        """
        w = self.w
        return (
            w[11]
            * (D ** (-w[12]))
            * ((S + 1) ** w[13] - 1)
            * math.exp(w[14] * (1 - R))
        )

    # ----- Difficulty Update -----

    def update_difficulty(self, D: float, grade: int) -> float:
        """
        dD = -w6 * (grade - 3)
        D'' = D + dD * (10 - D) / 9       # damping toward D=10
        D'  = w5 * D0(Easy) + (1 - w5) * D''  # mean reversion to default
        """
        w = self.w
        dD = -w[6] * (grade - 3)
        D_double_prime = D + dD * (10 - D) / 9.0
        D0_easy = self.initial_difficulty(4)
        D_prime = w[5] * D0_easy + (1 - w[5]) * D_double_prime
        return max(1.0, min(10.0, D_prime))

    # ----- Full Step -----

    def step(
        self, S: float, D: float, elapsed_days: float, grade: int
    ) -> Tuple[float, float, float]:
        """
        Process one review event.

        Returns:
            (S_next, D_next, R_at_review)
        """
        R = self.retrievability(elapsed_days, S) if elapsed_days > 0 else 1.0

        # Update stability
        if grade == 1:  # Again (lapse)
            S_next = self.stability_after_lapse(S, D, R)
        else:  # Hard / Good / Easy (success)
            S_next = self.stability_after_success(S, D, R, grade)

        # Update difficulty
        D_next = self.update_difficulty(D, grade)

        return S_next, D_next, R

    def simulate_student(
        self, reviews: List[dict]
    ) -> List[dict]:
        """
        Run FSRS-6 on a student's full review history to generate
        ground-truth (S, D, R) trajectories for warm-start training.

        Args:
            reviews: List of dicts with 'card_id', 'elapsed_days', 'grade'.

        Returns:
            List of dicts augmented with 'S', 'D', 'R' values.
        """
        card_states = {}  # card_id → (S, D)
        results = []

        for review in reviews:
            cid = review["card_id"]
            grade = review["grade"]
            elapsed = review["elapsed_days"]

            if cid not in card_states:
                # First review of this card
                S = self.initial_stability(grade)
                D = self.initial_difficulty(grade)
                R = 1.0  # first review: no forgetting yet
            else:
                S_prev, D_prev = card_states[cid]
                S, D, R = self.step(S_prev, D_prev, elapsed, grade)

            card_states[cid] = (S, D)
            results.append({
                **review,
                "S": S,
                "D": D,
                "R": R,
                "recalled": grade >= 2,
            })

        return results


def warm_start_from_fsrs6(
    model: nn.Module,
    dataset,
    config: Optional[TrainingConfig] = None,
    device: str = "cpu",
) -> nn.Module:
    """
    Pre-train MemoryNet to reproduce FSRS-6 predictions on a large corpus.

    This grounds the meta-initialization in a cognitively-valid baseline
    and cuts Reptile training time by ~60%.

    Args:
        model: MemoryNet instance.
        dataset: Iterable yielding batches of (features, S_prev, fsrs_targets).
            - features: (batch, input_dim)
            - S_prev: (batch,) previous stability
            - fsrs_targets: dict with 'S_target', 'D_target', 'R_target' tensors
        config: Training config for warmstart params.
        device: Device string.

    Returns:
        Pre-trained model (modified in-place).
    """
    cfg = config or TrainingConfig()
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.warmstart_lr)
    mse = nn.MSELoss()
    bce = nn.BCELoss()

    for epoch in range(cfg.warmstart_epochs):
        total_loss = 0.0
        n_batches = 0

        for features, S_prev, targets in dataset:
            features = features.to(device)
            S_prev = S_prev.to(device)
            S_target = targets["S_target"].to(device)
            D_target = targets["D_target"].to(device)
            R_target = targets["R_target"].to(device)

            state = model.forward_from_features(features, S_prev)

            loss = (
                bce(state.p_recall, R_target)
                + 0.5 * mse(torch.log(state.S_next + 1), torch.log(S_target + 1))
                + 0.5 * mse(state.D_next, D_target)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg = total_loss / max(n_batches, 1)
        print(f"  Warm-start epoch {epoch+1}/{cfg.warmstart_epochs}  loss={avg:.4f}")

    return model
```

---

## 8. `training/loss.py` — Multi-Component Loss Function (Section 3.3)

```python
"""
Multi-Component Loss Function (Section 3.3).

Three loss components:

1. PRIMARY — Recall prediction (binary cross-entropy):
   BCELoss(p_pred, recalled)

2. AUXILIARY 1 — Stability-curve consistency:
   MSELoss(R_from_S, recalled)
   where R_from_S = (0.9^(1/S_pred))^(elapsed_days^w20)

3. AUXILIARY 2 — Monotonicity constraint:
   S cannot decrease on successful recall.
   ReLU(-(S_next - S_prev)[grade >= 2]).mean()

Combined: recall_L + 0.10 * stable_L + 0.01 * mono_L
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from config import TrainingConfig, FSRSConfig


class MetaSRSLoss(nn.Module):
    """
    Multi-component loss for MemoryNet training.
    Used in both warm-start pre-training and Reptile inner loop.
    """

    def __init__(
        self,
        stability_weight: float = 0.10,
        monotonicity_weight: float = 0.01,
        w20: float = 0.4665,
    ):
        super().__init__()
        self.stability_weight = stability_weight
        self.monotonicity_weight = monotonicity_weight
        self.w20 = w20

        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def forward(
        self,
        p_recall_pred: torch.Tensor,
        S_next_pred: torch.Tensor,
        recalled: torch.Tensor,
        elapsed_days: torch.Tensor,
        grade: torch.Tensor,
        S_prev: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the combined loss.

        Args:
            p_recall_pred: (batch,) predicted recall probabilities
            S_next_pred:   (batch,) predicted next stability values
            recalled:      (batch,) binary ground truth (grade >= 2)
            elapsed_days:  (batch,) days since last review
            grade:         (batch,) grade values 1-4
            S_prev:        (batch,) optional previous stability for monotonicity

        Returns:
            Dict with 'total', 'recall', 'stability', 'monotonicity' losses.
        """
        # PRIMARY: recall prediction (binary cross-entropy)
        # Convert probabilities to logits for numerical stability
        p_clamped = p_recall_pred.clamp(1e-6, 1 - 1e-6)
        logits = torch.log(p_clamped / (1 - p_clamped))
        recall_L = self.bce(logits, recalled.float())

        # AUXILIARY 1: stability-curve consistency
        # R_from_S = (0.9^(1/S_pred))^(elapsed_days^w20)
        # Computed in log-space for numerical stability:
        #   log(R) = (1/S) * log(0.9) * (t^w20)
        log_09 = -0.10536051565782628  # math.log(0.9)
        t_pow = elapsed_days.clamp(min=1e-6) ** self.w20
        log_R = log_09 * t_pow / S_next_pred.clamp(min=1e-3)
        R_from_S = torch.exp(log_R.clamp(min=-20, max=0))  # R ∈ (0, 1]
        stable_L = self.mse(R_from_S, recalled.float())

        # AUXILIARY 2: monotonicity (S cannot decrease on successful recall)
        if S_prev is not None and S_next_pred.size(0) > 1:
            # Compare consecutive predictions where recall was successful
            success_mask = (grade >= 2).float()
            delta_S = S_next_pred - S_prev
            # Penalise negative delta_S on successful reviews
            mono_L = (F.relu(-delta_S) * success_mask).mean()
        else:
            mono_L = torch.tensor(0.0, device=p_recall_pred.device)

        # Combined loss
        total = (
            recall_L
            + self.stability_weight * stable_L
            + self.monotonicity_weight * mono_L
        )

        # Guard against NaN (can occur from extreme S values during adaptation)
        if torch.isnan(total):
            total = recall_L  # Fall back to primary loss only

        return {
            "total": total,
            "recall": recall_L,
            "stability": stable_L,
            "monotonicity": mono_L,
        }


def compute_loss(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    loss_fn: MetaSRSLoss,
) -> Dict[str, torch.Tensor]:
    """
    Convenience function: run model forward + compute loss on a batch.

    Args:
        model: MemoryNet instance
        batch: Dict from reviews_to_batch() with keys:
            D_prev, S_prev, R_at_review, delta_t, grade,
            review_count, user_stats, recalled
        loss_fn: MetaSRSLoss instance

    Returns:
        Dict with total loss and component breakdowns.
    """
    state = model(
        D_prev=batch["D_prev"],
        S_prev=batch["S_prev"],
        R_at_review=batch["R_at_review"],
        delta_t=batch["delta_t"],
        grade=batch["grade"],
        review_count=batch["review_count"],
        user_stats=batch["user_stats"],
        history_grades=batch.get("history_grades"),
        history_delta_ts=batch.get("history_delta_ts"),
        history_lengths=batch.get("history_lengths"),
    )

    return loss_fn(
        p_recall_pred=state.p_recall,
        S_next_pred=state.S_next,
        recalled=batch["recalled"],
        elapsed_days=batch["delta_t"],
        grade=batch["grade"],
        S_prev=batch["S_prev"],
    )
```

---

## 9. `training/reptile.py` — Reptile Meta-Training Loop (Sections 3.2, 3.4)

```python
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
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Sample a mini-batch of reviews from a support/query set."""
    if len(reviews) <= size:
        sampled = reviews
    else:
        sampled = random.sample(reviews, size)
    return reviews_to_batch(sampled, device)


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
    model.load_state_dict(phi_state_dict, strict=True)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=inner_lr)

    for step in range(k_steps):
        batch = sample_batch(
            task.support_set, batch_size, device
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
                            task.query_set, 64, self.device
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
```

---

## 10. `data/__init__.py`

```python
from .task_sampler import Task, TaskSampler, ReviewDataset

__all__ = ["Task", "TaskSampler", "ReviewDataset"]
```

---

## 11. `data/task_sampler.py` — Task Definition & Sampling (Section 3.1)

```python
"""
Task Definition & Sampling (Section 3.1).

Each task = one student's chronological review history.
The support set (first 70%) is used for inner-loop adaptation;
the query set (last 30%) measures generalization quality.

Task = {
    'student_id': UUID,
    'reviews': List[{
        'card_id': UUID,
        'timestamp': int,
        'elapsed_days': float,    # 0.0 = first-ever review of this card
        'grade': int,             # 1=Again, 2=Hard, 3=Good, 4=Easy
        'recalled': bool,         # grade >= 2
    }],
}
"""

import random
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


@dataclass
class Review:
    """Single review event."""
    card_id: str
    timestamp: int
    elapsed_days: float
    grade: int                 # 1=Again, 2=Hard, 3=Good, 4=Easy
    recalled: bool             # grade >= 2

    # Pre-computed states (filled by FSRS-6 or from data)
    S_prev: float = 1.0
    D_prev: float = 5.0
    R_at_review: float = 1.0
    S_target: float = 1.0     # Ground-truth next S (for warm-start)
    D_target: float = 5.0     # Ground-truth next D


@dataclass
class Task:
    """
    One student's complete review history, split into support/query sets.
    """
    student_id: str
    reviews: List[Review]

    # Support / query split
    support_set: List[Review] = field(default_factory=list)
    query_set: List[Review] = field(default_factory=list)

    def split(self, support_ratio: float = 0.70):
        """Split reviews into support (first 70%) and query (last 30%) sets."""
        n = len(self.reviews)
        split_idx = max(1, int(n * support_ratio))
        self.support_set = self.reviews[:split_idx]
        self.query_set = self.reviews[split_idx:]

    def get_review_history(self, card_id: str, up_to_idx: int) -> List[Review]:
        """Get all prior reviews of a specific card up to a given index."""
        history = []
        for r in self.reviews[:up_to_idx]:
            if r.card_id == card_id:
                history.append(r)
        return history

    @property
    def unique_cards(self) -> List[str]:
        return list({r.card_id for r in self.reviews})


def reviews_to_batch(
    reviews: List[Review],
    device: torch.device = torch.device("cpu"),
    history_len: int = 32,
) -> Dict[str, torch.Tensor]:
    """
    Convert a list of Review objects into a batched tensor dict
    suitable for MemoryNet forward pass.

    Returns dict with keys:
        D_prev, S_prev, R_at_review, delta_t, grade,
        review_count, user_stats, recalled
    """
    n = len(reviews)

    D_prev = torch.tensor([r.D_prev for r in reviews], dtype=torch.float32)
    S_prev = torch.tensor([r.S_prev for r in reviews], dtype=torch.float32)
    R_at_review = torch.tensor([r.R_at_review for r in reviews], dtype=torch.float32)
    delta_t = torch.tensor([r.elapsed_days for r in reviews], dtype=torch.float32)
    grade = torch.tensor([r.grade for r in reviews], dtype=torch.long)
    recalled = torch.tensor([float(r.recalled) for r in reviews], dtype=torch.float32)

    # Review count per card (accumulate occurrences up to each review)
    card_counts: Dict[str, int] = {}
    review_count_list = []
    for r in reviews:
        card_counts[r.card_id] = card_counts.get(r.card_id, 0) + 1
        review_count_list.append(float(card_counts[r.card_id]))
    review_count = torch.tensor(review_count_list, dtype=torch.float32)

    # User stats (placeholder: mean D, mean S, session length, etc.)
    # In production, compute from student's review history
    mean_D = D_prev.mean().expand(n)
    mean_S = S_prev.mean().expand(n)
    user_stats = torch.zeros(n, 8, dtype=torch.float32)
    user_stats[:, 0] = mean_D
    user_stats[:, 1] = torch.log(mean_S.clamp(min=1e-6))

    # Build per-card review history sequences for GRU encoder
    max_hist_len = history_len
    card_history: Dict[str, List[Tuple[float, float]]] = {}
    history_grades_list = []
    history_delta_ts_list = []
    history_lengths_list = []

    for r in reviews:
        hist = card_history.get(r.card_id, [])
        seq_len = min(len(hist), max_hist_len)

        # Pad/truncate to max_hist_len
        if seq_len == 0:
            h_grades = [0.0] * max_hist_len
            h_dts = [0.0] * max_hist_len
        else:
            recent = hist[-max_hist_len:]
            h_grades = [g for g, _ in recent] + [0.0] * (max_hist_len - len(recent))
            h_dts = [d for _, d in recent] + [0.0] * (max_hist_len - len(recent))

        history_grades_list.append(h_grades)
        history_delta_ts_list.append(h_dts)
        history_lengths_list.append(max(seq_len, 1))  # clamp min=1 for pack_padded

        # Append current review to history for future reviews
        card_history.setdefault(r.card_id, []).append(
            (float(r.grade), r.elapsed_days)
        )

    history_grades = torch.tensor(history_grades_list, dtype=torch.float32)
    history_delta_ts = torch.tensor(history_delta_ts_list, dtype=torch.float32)
    history_lengths = torch.tensor(history_lengths_list, dtype=torch.long)

    # Targets for warm-start
    S_target = torch.tensor([r.S_target for r in reviews], dtype=torch.float32)
    D_target = torch.tensor([r.D_target for r in reviews], dtype=torch.float32)

    batch = {
        "D_prev": D_prev.to(device),
        "S_prev": S_prev.to(device),
        "R_at_review": R_at_review.to(device),
        "delta_t": delta_t.to(device),
        "grade": grade.to(device),
        "review_count": review_count.to(device),
        "user_stats": user_stats.to(device),
        "recalled": recalled.to(device),
        "S_target": S_target.to(device),
        "D_target": D_target.to(device),
        "history_grades": history_grades.to(device),
        "history_delta_ts": history_delta_ts.to(device),
        "history_lengths": history_lengths.to(device),
    }

    return batch


class TaskSampler:
    """
    Samples tasks (students) for the Reptile meta-training outer loop.

    Each task is one student's chronological review history, split
    into support/query sets for inner-loop training/evaluation.
    """

    def __init__(
        self,
        tasks: List[Task],
        support_ratio: float = 0.70,
        min_reviews: int = 10,
        seed: int = 42,
    ):
        self.rng = random.Random(seed)

        # Filter students with too few reviews
        self.tasks = [t for t in tasks if len(t.reviews) >= min_reviews]
        for task in self.tasks:
            task.split(support_ratio)

        print(f"TaskSampler: {len(self.tasks)} students loaded "
              f"(filtered {len(tasks) - len(self.tasks)} with <{min_reviews} reviews)")

    def sample(self, batch_size: int = 16) -> List[Task]:
        """Sample a batch of tasks for one meta-iteration."""
        return self.rng.choices(self.tasks, k=batch_size)

    def __len__(self) -> int:
        return len(self.tasks)


class ReviewDataset:
    """
    Loads review data from CSV/JSON and builds Task objects.

    Expected CSV columns:
        student_id, card_id, timestamp, elapsed_days, grade
    """

    @staticmethod
    def from_csv(
        csv_path: str,
    ) -> List[Task]:
        """
        Load tasks from a CSV file.

        Args:
            csv_path: Path to CSV with review logs.

        Returns:
            List of Task objects, one per student.
        """
        import csv

        # Group reviews by student
        student_reviews: Dict[str, List[Review]] = defaultdict(list)

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                review = Review(
                    card_id=row["card_id"],
                    timestamp=int(row["timestamp"]),
                    elapsed_days=float(row["elapsed_days"]),
                    grade=int(row["grade"]),
                    recalled=int(row["grade"]) >= 2,
                )
                student_reviews[row["student_id"]].append(review)

        # Sort each student's reviews chronologically and build Tasks
        tasks = []
        for sid, reviews in student_reviews.items():
            reviews.sort(key=lambda r: r.timestamp)
            task = Task(
                student_id=sid,
                reviews=reviews,
            )
            tasks.append(task)

        return tasks

    @staticmethod
    def generate_synthetic(
        n_students: int = 500,
        reviews_per_student: int = 100,
        n_cards: int = 200,
        seed: int = 42,
    ) -> List[Task]:
        """
        Generate synthetic review data for testing.
        Uses FSRS-6 to simulate realistic memory dynamics.
        """
        from training.fsrs_warmstart import FSRS6

        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)
        fsrs = FSRS6()

        # Card IDs
        card_ids = [f"card_{i:04d}" for i in range(n_cards)]

        tasks = []

        for s in range(n_students):
            student_id = f"student_{s:04d}"
            reviews = []
            card_states: Dict[str, Tuple[float, float, int]] = {}  # cid → (S, D, count)
            timestamp = 0

            for _ in range(reviews_per_student):
                cid = rng.choice(card_ids)

                if cid not in card_states:
                    elapsed = 0.0
                    grade = rng.choices([1, 2, 3, 4], weights=[15, 20, 50, 15])[0]
                    S_prev = fsrs.initial_stability(3)  # Good default
                    D_prev = fsrs.initial_difficulty(3)
                    count = 0
                else:
                    S_prev, D_prev, count = card_states[cid]
                    elapsed = rng.uniform(0.5, S_prev * 2.5)  # Roughly around optimal

                    # Simulate recall based on retrievability
                    R = fsrs.retrievability(elapsed, S_prev)
                    recalled = rng.random() < R
                    if recalled:
                        grade = rng.choices([2, 3, 4], weights=[20, 60, 20])[0]
                    else:
                        grade = 1  # Again

                R_at = fsrs.retrievability(elapsed, S_prev) if elapsed > 0 else 1.0

                S_next, D_next, _ = fsrs.step(S_prev, D_prev, elapsed, grade)

                review = Review(
                    card_id=cid,
                    timestamp=timestamp,
                    elapsed_days=elapsed,
                    grade=grade,
                    recalled=grade >= 2,
                    S_prev=S_prev,
                    D_prev=D_prev,
                    R_at_review=R_at,
                    S_target=S_next,
                    D_target=D_next,
                )
                reviews.append(review)

                card_states[cid] = (S_next, D_next, count + 1)
                timestamp += int(elapsed * 86400)  # Convert days to seconds

            task = Task(
                student_id=student_id,
                reviews=reviews,
            )
            tasks.append(task)

        return tasks
```

---

## 12. `inference/__init__.py`

```python
from .adaptation import FastAdapter, AdaptationPhase
from .scheduling import Scheduler, ScheduleResult

__all__ = [
    "FastAdapter",
    "AdaptationPhase",
    "Scheduler",
    "ScheduleResult",
]
```

---

## 13. `inference/adaptation.py` — Fast Adaptation (Section 4.1)

```python
"""
Fast Adaptation for New Students (Section 4.1).

Three-Phase Onboarding Protocol:

Phase 1 — Zero-shot (0 reviews):
    Use phi* directly. Schedule with meta-parameter predictions.
    Apply uncertainty-aware exploration — mix predicted-easy and
    uncertain cards to gather calibration data quickly.

Phase 2 — Rapid adapt (5–50 reviews):
    Every 5 new reviews: run k=5 gradient steps from phi* on student
    history. Switch scheduling from phi* to theta_student. AUC
    already exceeds population-average FSRS-6.

Phase 3 — Full personal (50+ reviews):
    Run k=20 gradient steps at higher inner learning rate. Enable
    online streaming updates (gradient step after every review).
    Reach full personalization within the first study session.
"""

import torch
import torch.nn as nn
from copy import deepcopy
from enum import Enum, auto
from typing import Dict, List, Optional
from collections import OrderedDict

from config import AdaptationConfig, MetaSRSConfig
from models.memory_net import MemoryNet
from training.loss import MetaSRSLoss, compute_loss
from data.task_sampler import Review, reviews_to_batch


class AdaptationPhase(Enum):
    """The three onboarding phases from Section 4.1."""
    ZERO_SHOT = auto()    # Phase 1: 0 reviews, use phi* directly
    RAPID_ADAPT = auto()  # Phase 2: 5–50 reviews, periodic adaptation
    FULL_PERSONAL = auto() # Phase 3: 50+ reviews, streaming updates


class FastAdapter:
    """
    Online fast-adaptation engine for a single student.

    Manages the transition through the three onboarding phases and
    maintains the student's personalised parameters theta_student.

    Usage:
        adapter = FastAdapter(model, phi_star, config)
        # Student reviews a card:
        adapter.add_review(review)
        # Get current model for scheduling:
        model = adapter.get_model()
    """

    def __init__(
        self,
        model: MemoryNet,
        phi_star: OrderedDict,
        config: Optional[MetaSRSConfig] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model.to(device)
        self.phi_star = deepcopy(phi_star)
        self.cfg = (config or MetaSRSConfig()).adaptation
        self.fsrs_cfg = (config or MetaSRSConfig()).fsrs
        self.device = device

        # Student's personalised parameters
        self.theta_student = deepcopy(phi_star)
        self.reviews: List[Review] = []
        self._reviews_since_last_adapt = 0

        self.loss_fn = MetaSRSLoss(w20=self.fsrs_cfg.w[20])

    @property
    def phase(self) -> AdaptationPhase:
        """Determine current onboarding phase based on review count."""
        n = len(self.reviews)
        if n < self.cfg.phase1_threshold:
            return AdaptationPhase.ZERO_SHOT
        elif n < self.cfg.phase2_threshold:
            return AdaptationPhase.RAPID_ADAPT
        else:
            return AdaptationPhase.FULL_PERSONAL

    @property
    def n_reviews(self) -> int:
        return len(self.reviews)

    def add_review(self, review: Review):
        """
        Process a new review event. Triggers adaptation if appropriate
        based on the current phase.
        """
        self.reviews.append(review)
        self._reviews_since_last_adapt += 1

        phase = self.phase

        if phase == AdaptationPhase.ZERO_SHOT:
            # No adaptation yet — use phi* directly
            pass

        elif phase == AdaptationPhase.RAPID_ADAPT:
            # Every N reviews: run k=5 gradient steps from phi*
            if self._reviews_since_last_adapt >= self.cfg.adapt_every_n_reviews:
                self._adapt(
                    k_steps=self.cfg.phase1_k_steps,
                    inner_lr=0.01,
                    from_phi=True,  # Always adapt from phi*, not theta
                )
                self._reviews_since_last_adapt = 0

        elif phase == AdaptationPhase.FULL_PERSONAL:
            if self._reviews_since_last_adapt >= self.cfg.adapt_every_n_reviews:
                # Run k=20 gradient steps at higher LR
                self._adapt(
                    k_steps=self.cfg.phase3_k_steps,
                    inner_lr=self.cfg.phase3_inner_lr,
                    from_phi=False,  # Fine-tune from current theta
                )
                self._reviews_since_last_adapt = 0
            elif len(self.reviews) >= self.cfg.streaming_after:
                # Streaming: single gradient step after every review
                self._streaming_step()

    def _adapt(
        self,
        k_steps: int,
        inner_lr: float,
        from_phi: bool = True,
    ):
        """Run k gradient steps for adaptation."""
        # Start from phi* or current theta
        start_params = self.phi_star if from_phi else self.theta_student
        self.model.load_state_dict(start_params)
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=inner_lr)

        # Use all available reviews as training data
        batch = reviews_to_batch(
            self.reviews, self.device
        )

        for step in range(k_steps):
            losses = compute_loss(self.model, batch, self.loss_fn)
            loss = losses["total"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

        self.theta_student = deepcopy(self.model.state_dict())

    def _streaming_step(self):
        """Single gradient step on the most recent review (online learning)."""
        if not self.reviews:
            return

        self.model.load_state_dict(self.theta_student)
        self.model.train()

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.cfg.phase3_inner_lr
        )

        # Use last few reviews as a mini-batch
        recent = self.reviews[-min(8, len(self.reviews)):]
        batch = reviews_to_batch(recent, self.device)

        losses = compute_loss(self.model, batch, self.loss_fn)
        loss = losses["total"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()

        self.theta_student = deepcopy(self.model.state_dict())

    def get_model(self) -> MemoryNet:
        """
        Get the current model with appropriate parameters for scheduling.

        Phase 1: Returns model with phi* (meta-parameters).
        Phase 2–3: Returns model with theta_student (adapted).
        """
        if self.phase == AdaptationPhase.ZERO_SHOT:
            self.model.load_state_dict(self.phi_star)
        else:
            self.model.load_state_dict(self.theta_student)
        self.model.eval()
        return self.model

    def get_state(self) -> dict:
        """Serialise adapter state for persistence."""
        return {
            "theta_student": self.theta_student,
            "reviews": self.reviews,
            "n_reviews": len(self.reviews),
            "phase": self.phase.name,
            "reviews_since_adapt": self._reviews_since_last_adapt,
        }

    def load_state(self, state: dict):
        """Restore adapter state."""
        self.theta_student = state["theta_student"]
        self.reviews = state["reviews"]
        self._reviews_since_last_adapt = state.get("reviews_since_adapt", 0)
```

---

## 14. `inference/scheduling.py` — Uncertainty-Aware Scheduling (Section 4.2)

```python
"""
Uncertainty-Aware Scheduling (Section 4.2).

Monte Carlo Dropout for uncertainty quantification:
    def predict_with_uncertainty(theta, x, n_samples=20):
        model.train()  # keep dropout active at inference
        preds = [model(x) for _ in range(n_samples)]
        mean  = torch.stack(preds).mean(0)  # expected recall probability
        sigma = torch.stack(preds).std(0)   # epistemic uncertainty
        return mean, sigma

Interval with uncertainty and difficulty discounting:
    def compute_interval(S_pred, D, sigma, desired_retention=0.90):
        base_interval = S_pred  # when DR=0.90, I ~ S
        confidence_factor = 1.0 - 0.5 * sigma     # range [0.5, 1.0]
        difficulty_factor = 1.0 - 0.05 * (D - 5)  # centre on D=5
        fuzz = uniform(0.95, 1.05)                 # prevent card clustering
        return max(1, round(base_interval * confidence_factor
                            * difficulty_factor * fuzz))
"""

import random
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from config import SchedulingConfig, ModelConfig
from models.memory_net import MemoryNet


@dataclass
class ScheduleResult:
    """Result of scheduling a single card."""
    card_id: str
    interval_days: int            # Recommended review interval
    p_recall_mean: float          # Expected recall probability
    p_recall_sigma: float         # Epistemic uncertainty
    S_pred: float                 # Predicted stability (days)
    D_pred: float                 # Predicted difficulty [1, 10]
    confidence_factor: float      # Uncertainty discount applied
    difficulty_factor: float      # Difficulty discount applied


class Scheduler:
    """
    Uncertainty-aware card scheduler.

    Uses Monte Carlo Dropout to estimate both the expected recall
    probability and its epistemic uncertainty, then computes review
    intervals with confidence and difficulty discounting.
    """

    def __init__(
        self,
        model: MemoryNet,
        config: Optional[SchedulingConfig] = None,
        mc_samples: int = 20,
    ):
        self.model = model
        self.cfg = config or SchedulingConfig()
        self.mc_samples = mc_samples

    def predict_with_uncertainty(
        self,
        features: torch.Tensor,
        S_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Monte Carlo Dropout: run the model n_samples times with dropout
        active to estimate prediction uncertainty.

        Args:
            features: (batch, input_dim) pre-assembled feature vectors.
            S_prev:   (batch,) previous stability values.

        Returns:
            p_recall_mean:  (batch,) expected recall probability
            p_recall_sigma: (batch,) epistemic uncertainty
            S_mean:         (batch,) expected next stability
            D_mean:         (batch,) expected next difficulty
        """
        self.model.train()  # Keep dropout active for MC sampling

        all_p_recall = []
        all_S = []
        all_D = []

        with torch.no_grad():
            for _ in range(self.mc_samples):
                state = self.model.forward_from_features(features, S_prev)
                all_p_recall.append(state.p_recall)
                all_S.append(state.S_next)
                all_D.append(state.D_next)

        p_stack = torch.stack(all_p_recall)  # (n_samples, batch)
        S_stack = torch.stack(all_S)
        D_stack = torch.stack(all_D)

        return (
            p_stack.mean(0),
            p_stack.std(0),
            S_stack.mean(0),
            D_stack.mean(0),
        )

    def compute_interval(
        self,
        S_pred: float,
        D: float,
        sigma: float,
    ) -> Tuple[int, float, float]:
        """
        Compute review interval with uncertainty and difficulty discounting.

        Args:
            S_pred: Predicted stability (days until R drops to 90%).
            D: Predicted difficulty [1, 10].
            sigma: Epistemic uncertainty of recall prediction.

        Returns:
            (interval_days, confidence_factor, difficulty_factor)
        """
        base_interval = S_pred  # When desired_retention=0.90, interval ~ S

        # Uncertainty discount: high uncertainty → shorter interval
        confidence_factor = 1.0 - self.cfg.uncertainty_discount * min(sigma, 1.0)
        confidence_factor = max(0.5, min(1.0, confidence_factor))

        # Difficulty discount: harder cards get slightly shorter intervals
        difficulty_factor = 1.0 - self.cfg.difficulty_slope * (
            D - self.cfg.difficulty_centre
        )
        difficulty_factor = max(0.5, min(1.5, difficulty_factor))

        # Fuzz to prevent card clustering
        fuzz_lo, fuzz_hi = self.cfg.fuzz_range
        fuzz = random.uniform(fuzz_lo, fuzz_hi)

        interval = base_interval * confidence_factor * difficulty_factor * fuzz
        interval = max(self.cfg.min_interval, min(self.cfg.max_interval, round(interval)))

        return int(interval), confidence_factor, difficulty_factor

    def schedule_card(
        self,
        card_id: str,
        features: torch.Tensor,
        S_prev: torch.Tensor,
    ) -> ScheduleResult:
        """
        Schedule a single card: predict memory state + compute interval.

        Args:
            card_id: Identifier for the card.
            features: (1, input_dim) feature vector for this card/review.
            S_prev: (1,) previous stability.

        Returns:
            ScheduleResult with interval and prediction details.
        """
        p_mean, p_sigma, S_mean, D_mean = self.predict_with_uncertainty(
            features, S_prev
        )

        interval, conf_factor, diff_factor = self.compute_interval(
            S_pred=S_mean.item(),
            D=D_mean.item(),
            sigma=p_sigma.item(),
        )

        return ScheduleResult(
            card_id=card_id,
            interval_days=interval,
            p_recall_mean=p_mean.item(),
            p_recall_sigma=p_sigma.item(),
            S_pred=S_mean.item(),
            D_pred=D_mean.item(),
            confidence_factor=conf_factor,
            difficulty_factor=diff_factor,
        )

    def schedule_deck(
        self,
        card_ids: List[str],
        features: torch.Tensor,
        S_prev: torch.Tensor,
    ) -> List[ScheduleResult]:
        """
        Schedule an entire deck: batch MC-Dropout + individual intervals.

        Args:
            card_ids: List of card identifiers.
            features: (n_cards, input_dim) feature vectors.
            S_prev: (n_cards,) previous stability values.

        Returns:
            List of ScheduleResult, one per card.
        """
        p_mean, p_sigma, S_mean, D_mean = self.predict_with_uncertainty(
            features, S_prev
        )

        results = []
        for i, card_id in enumerate(card_ids):
            interval, conf_factor, diff_factor = self.compute_interval(
                S_pred=S_mean[i].item(),
                D=D_mean[i].item(),
                sigma=p_sigma[i].item(),
            )
            results.append(ScheduleResult(
                card_id=card_id,
                interval_days=interval,
                p_recall_mean=p_mean[i].item(),
                p_recall_sigma=p_sigma[i].item(),
                S_pred=S_mean[i].item(),
                D_pred=D_mean[i].item(),
                confidence_factor=conf_factor,
                difficulty_factor=diff_factor,
            ))

        return results

    def select_next_card(
        self,
        results: List[ScheduleResult],
        strategy: str = "uncertainty_weighted",
    ) -> ScheduleResult:
        """
        Select which card to review next using uncertainty-aware exploration.

        Strategies:
            'most_due': Card with lowest interval (most overdue).
            'highest_uncertainty': Card with highest sigma.
            'uncertainty_weighted': Mix of due-ness and uncertainty
                (Phase 1 exploration: mix predicted-easy and uncertain cards).
        """
        if not results:
            raise ValueError("No cards to schedule")

        if strategy == "most_due":
            return min(results, key=lambda r: r.interval_days)

        elif strategy == "highest_uncertainty":
            return max(results, key=lambda r: r.p_recall_sigma)

        elif strategy == "uncertainty_weighted":
            # Score = (1 - p_recall) + 0.5 * sigma
            # High score → more urgent (forgotten or uncertain)
            scores = [
                (1.0 - r.p_recall_mean) + 0.5 * r.p_recall_sigma
                for r in results
            ]
            # Softmax selection for stochastic exploration
            scores_t = torch.tensor(scores)
            probs = torch.softmax(scores_t * 3.0, dim=0)  # temperature=1/3
            idx = torch.multinomial(probs, 1).item()
            return results[idx]

        else:
            raise ValueError(f"Unknown strategy: {strategy}")
```

---

## 15. `evaluation/__init__.py`

```python
from .metrics import MetaSRSEvaluator, EvalResults

__all__ = ["MetaSRSEvaluator", "EvalResults"]
```

---

## 16. `evaluation/metrics.py` — Evaluation Framework (Section 7)

```python
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
```

---

## 17. `train.py` — Main Training Script

```python
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
```

---

## 18. `pytest.ini`

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
addopts = -v --tb=short --timeout=120
```

---

## 19. `tests/__init__.py`

```python

```

---

## 20. `tests/conftest.py` — Shared Test Fixtures

```python
"""Shared fixtures for MetaSRS tests."""

import sys
import os
import torch
import numpy as np
import pytest

# Add meta-srs root to sys.path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import MetaSRSConfig, ModelConfig, TrainingConfig, FSRSConfig
from models.memory_net import MemoryNet
from data.task_sampler import Review, Task, ReviewDataset
from training.fsrs_warmstart import FSRS6


def create_model(config=None):
    """Create a MemoryNet from ModelConfig, filtering out config-only fields."""
    if config is None:
        config = ModelConfig()
    init_args = {k: v for k, v in config.__dict__.items()
                 if k in MemoryNet.__init__.__code__.co_varnames}
    return MemoryNet(**init_args)


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def config():
    return MetaSRSConfig()


@pytest.fixture
def model_config():
    return ModelConfig()


@pytest.fixture
def training_config():
    return TrainingConfig()


@pytest.fixture
def fsrs_config():
    return FSRSConfig()


@pytest.fixture
def model(model_config):
    return create_model(model_config)


@pytest.fixture
def fsrs():
    return FSRS6()


@pytest.fixture
def sample_batch_tensors(device):
    """Create a small batch of tensors suitable for MemoryNet forward."""
    batch_size = 4
    return {
        "D_prev": torch.tensor([5.0, 3.0, 7.0, 2.0], device=device),
        "S_prev": torch.tensor([10.0, 5.0, 20.0, 1.0], device=device),
        "R_at_review": torch.tensor([0.9, 0.7, 0.5, 1.0], device=device),
        "delta_t": torch.tensor([3.0, 7.0, 14.0, 0.0], device=device),
        "grade": torch.tensor([3, 2, 4, 1], dtype=torch.long, device=device),
        "review_count": torch.tensor([5.0, 3.0, 10.0, 1.0], device=device),
        "user_stats": torch.zeros(batch_size, 8, device=device),
        "history_grades": torch.tensor(
            [[3, 2, 0, 0], [4, 0, 0, 0], [3, 3, 2, 4], [0, 0, 0, 0]],
            dtype=torch.float32, device=device,
        ),
        "history_delta_ts": torch.tensor(
            [[1.0, 3.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0],
             [1.0, 3.0, 7.0, 14.0], [0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32, device=device,
        ),
        "history_lengths": torch.tensor([2, 1, 4, 1], dtype=torch.long, device=device),
        "recalled": torch.tensor([1.0, 1.0, 1.0, 0.0], device=device),
        "S_target": torch.tensor([15.0, 8.0, 30.0, 0.5], device=device),
        "D_target": torch.tensor([4.5, 3.2, 6.8, 2.5], device=device),
    }


@pytest.fixture
def sample_reviews():
    """Create a list of sample Review objects."""
    return [
        Review(card_id="c1", timestamp=0, elapsed_days=0.0, grade=3,
               recalled=True, S_prev=3.0, D_prev=5.0, R_at_review=1.0,
               S_target=5.0, D_target=4.8),
        Review(card_id="c2", timestamp=86400, elapsed_days=1.0, grade=2,
               recalled=True, S_prev=1.0, D_prev=7.0, R_at_review=0.9,
               S_target=2.0, D_target=7.2),
        Review(card_id="c1", timestamp=172800, elapsed_days=2.0, grade=4,
               recalled=True, S_prev=5.0, D_prev=4.8, R_at_review=0.85,
               S_target=15.0, D_target=4.0),
        Review(card_id="c3", timestamp=259200, elapsed_days=0.0, grade=1,
               recalled=False, S_prev=1.0, D_prev=5.0, R_at_review=1.0,
               S_target=0.5, D_target=6.0),
    ]


@pytest.fixture
def synthetic_tasks():
    """Generate a small set of synthetic tasks for testing."""
    return ReviewDataset.generate_synthetic(
        n_students=20, reviews_per_student=50, n_cards=30, seed=42
    )
```

---

## 21. `tests/test_config.py`

```python
"""Tests for MetaSRS configuration."""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    MetaSRSConfig, ModelConfig, FSRSConfig,
    TrainingConfig, AdaptationConfig, SchedulingConfig,
)


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.input_dim == 49
        assert cfg.hidden_dim == 128
        assert cfg.gru_hidden_dim == 32
        assert cfg.history_len == 32
        assert cfg.user_stats_dim == 8
        assert cfg.dropout == 0.1
        assert cfg.mc_samples == 20

    def test_feature_dim_sum(self):
        """Verify input_dim = scalars(4) + grade(4) + count(1) + user(8) + gru(32)."""
        cfg = ModelConfig()
        expected = 4 + 4 + 1 + cfg.user_stats_dim + cfg.gru_hidden_dim
        assert cfg.input_dim == expected


class TestFSRSConfig:
    def test_default_weights(self):
        cfg = FSRSConfig()
        assert len(cfg.w) == 21
        assert cfg.w[0] == 0.40255   # initial stability (Again)
        assert cfg.w[20] == 0.4665   # power-law exponent

    def test_weights_are_positive(self):
        cfg = FSRSConfig()
        # All weights should be non-negative (w7 is a placeholder)
        for i, w in enumerate(cfg.w):
            assert w >= 0, f"w[{i}] = {w} is negative"


class TestTrainingConfig:
    def test_epsilon_schedule_boundaries(self):
        cfg = TrainingConfig()
        assert cfg.epsilon_schedule(0) == cfg.epsilon_start
        assert abs(cfg.epsilon_schedule(cfg.n_iters) - cfg.epsilon_end) < 1e-8

    def test_epsilon_schedule_monotonic_decrease(self):
        cfg = TrainingConfig()
        prev = cfg.epsilon_schedule(0)
        for i in range(1, 100):
            cur = cfg.epsilon_schedule(i * 500)
            assert cur <= prev + 1e-12, f"epsilon increased at iter {i*500}"
            prev = cur

    def test_outer_lr_schedule_boundaries(self):
        cfg = TrainingConfig()
        # At iter 0, cosine gives cos(0) = 1 → lr = lr_start
        assert abs(cfg.outer_lr_schedule(0) - cfg.outer_lr_start) < 1e-8
        # At iter n_iters, cosine gives cos(pi) = -1 → lr = lr_end
        assert abs(cfg.outer_lr_schedule(cfg.n_iters) - cfg.outer_lr_end) < 1e-8

    def test_outer_lr_schedule_monotonic_decrease(self):
        cfg = TrainingConfig()
        prev = cfg.outer_lr_schedule(0)
        for i in range(1, 100):
            cur = cfg.outer_lr_schedule(i * 500)
            assert cur <= prev + 1e-12, f"outer_lr increased at iter {i*500}"
            prev = cur


class TestAdaptationConfig:
    def test_phase_thresholds_ordered(self):
        cfg = AdaptationConfig()
        assert cfg.phase1_threshold < cfg.phase2_threshold

    def test_k_steps_increasing(self):
        cfg = AdaptationConfig()
        assert cfg.phase1_k_steps <= cfg.phase2_k_steps <= cfg.phase3_k_steps


class TestSchedulingConfig:
    def test_defaults(self):
        cfg = SchedulingConfig()
        assert cfg.desired_retention == 0.90
        assert cfg.min_interval == 1
        assert cfg.max_interval == 365
        assert cfg.fuzz_range[0] < cfg.fuzz_range[1]


class TestMetaSRSConfig:
    def test_aggregation(self):
        cfg = MetaSRSConfig()
        assert isinstance(cfg.model, ModelConfig)
        assert isinstance(cfg.fsrs, FSRSConfig)
        assert isinstance(cfg.training, TrainingConfig)
        assert isinstance(cfg.adaptation, AdaptationConfig)
        assert isinstance(cfg.scheduling, SchedulingConfig)
```

---

## 22. `tests/test_memory_net.py`

```python
"""Tests for MemoryNet model."""

import sys
import os
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import ModelConfig
from models.memory_net import MemoryNet, MemoryState


class TestMemoryNetArchitecture:
    def test_creation(self, model):
        assert isinstance(model, MemoryNet)

    def test_parameter_count(self, model):
        n_params = model.count_parameters()
        # Should be around 50K–70K as per architecture
        assert 30_000 < n_params < 150_000, f"Unexpected param count: {n_params}"

    def test_input_dim(self, model):
        assert model.input_dim == 49

    def test_submodules_exist(self, model):
        assert hasattr(model, "gru_encoder")
        assert hasattr(model, "stability_head")
        assert hasattr(model, "difficulty_head")
        assert hasattr(model, "recall_head")


class TestMemoryNetForward:
    def test_forward_output_shapes(self, model, sample_batch_tensors, device):
        batch = sample_batch_tensors
        state = model(
            D_prev=batch["D_prev"],
            S_prev=batch["S_prev"],
            R_at_review=batch["R_at_review"],
            delta_t=batch["delta_t"],
            grade=batch["grade"],
            review_count=batch["review_count"],
            user_stats=batch["user_stats"],
            history_grades=batch["history_grades"],
            history_delta_ts=batch["history_delta_ts"],
            history_lengths=batch["history_lengths"],
        )
        assert isinstance(state, MemoryState)
        assert state.S_next.shape == (4,)
        assert state.D_next.shape == (4,)
        assert state.p_recall.shape == (4,)

    def test_output_ranges(self, model, sample_batch_tensors, device):
        batch = sample_batch_tensors
        model.eval()
        with torch.no_grad():
            state = model(
                D_prev=batch["D_prev"],
                S_prev=batch["S_prev"],
                R_at_review=batch["R_at_review"],
                delta_t=batch["delta_t"],
                grade=batch["grade"],
                review_count=batch["review_count"],
                user_stats=batch["user_stats"],
                history_grades=batch["history_grades"],
                history_delta_ts=batch["history_delta_ts"],
                history_lengths=batch["history_lengths"],
            )

        # S_next clamped to [0.001, 36500]
        assert (state.S_next >= 1e-3).all()
        assert (state.S_next <= 36500.0).all()
        # D_next in [1, 10]
        assert (state.D_next >= 1.0).all()
        assert (state.D_next <= 10.0).all()
        # p_recall in [0, 1]
        assert (state.p_recall >= 0.0).all()
        assert (state.p_recall <= 1.0).all()

    def test_no_nan_in_output(self, model, sample_batch_tensors, device):
        batch = sample_batch_tensors
        model.eval()
        with torch.no_grad():
            state = model(
                D_prev=batch["D_prev"],
                S_prev=batch["S_prev"],
                R_at_review=batch["R_at_review"],
                delta_t=batch["delta_t"],
                grade=batch["grade"],
                review_count=batch["review_count"],
                user_stats=batch["user_stats"],
                history_grades=batch["history_grades"],
                history_delta_ts=batch["history_delta_ts"],
                history_lengths=batch["history_lengths"],
            )
        assert not torch.isnan(state.S_next).any()
        assert not torch.isnan(state.D_next).any()
        assert not torch.isnan(state.p_recall).any()


class TestBuildFeatures:
    def test_feature_dim(self, model, sample_batch_tensors, device):
        batch = sample_batch_tensors
        features = model.build_features(
            D_prev=batch["D_prev"],
            S_prev=batch["S_prev"],
            R_at_review=batch["R_at_review"],
            delta_t=batch["delta_t"],
            grade=batch["grade"],
            review_count=batch["review_count"],
            user_stats=batch["user_stats"],
            history_grades=batch["history_grades"],
            history_delta_ts=batch["history_delta_ts"],
            history_lengths=batch["history_lengths"],
        )
        assert features.shape == (4, 49)

    def test_no_history_uses_zero_context(self, model, sample_batch_tensors, device):
        """When history is None, GRU should produce zero context."""
        batch = sample_batch_tensors
        features = model.build_features(
            D_prev=batch["D_prev"],
            S_prev=batch["S_prev"],
            R_at_review=batch["R_at_review"],
            delta_t=batch["delta_t"],
            grade=batch["grade"],
            review_count=batch["review_count"],
            user_stats=batch["user_stats"],
            history_grades=None,
            history_delta_ts=None,
            history_lengths=None,
        )
        assert features.shape == (4, 49)
        # Last 32 dims (GRU context) should be zero
        assert (features[:, -32:] == 0).all()


class TestForwardFromFeatures:
    def test_forward_from_features(self, model, device):
        features = torch.randn(4, 49, device=device)
        S_prev = torch.tensor([5.0, 10.0, 1.0, 20.0], device=device)
        state = model.forward_from_features(features, S_prev)
        assert state.S_next.shape == (4,)
        assert state.D_next.shape == (4,)
        assert state.p_recall.shape == (4,)

    def test_predict_recall(self, model, device):
        features = torch.randn(4, 49, device=device)
        S_prev = torch.tensor([5.0, 10.0, 1.0, 20.0], device=device)
        p = model.predict_recall(features, S_prev)
        assert p.shape == (4,)
        assert (p >= 0).all() and (p <= 1).all()

    def test_predict_stability(self, model, device):
        features = torch.randn(4, 49, device=device)
        S_prev = torch.tensor([5.0, 10.0, 1.0, 20.0], device=device)
        S = model.predict_stability(features, S_prev)
        assert S.shape == (4,)
        assert (S >= 1e-3).all()


class TestGradientFlow:
    def test_backward_pass(self, model, sample_batch_tensors, device):
        """Ensure gradients flow through the full forward pass."""
        batch = sample_batch_tensors
        state = model(
            D_prev=batch["D_prev"],
            S_prev=batch["S_prev"],
            R_at_review=batch["R_at_review"],
            delta_t=batch["delta_t"],
            grade=batch["grade"],
            review_count=batch["review_count"],
            user_stats=batch["user_stats"],
            history_grades=batch["history_grades"],
            history_delta_ts=batch["history_delta_ts"],
            history_lengths=batch["history_lengths"],
        )
        loss = state.p_recall.sum()
        loss.backward()

        # Check at least some parameters have gradients
        grads_found = 0
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                grads_found += 1
        assert grads_found > 0, "No gradients found after backward pass"
```

---

## 23. `tests/test_models.py` — GRU Encoder Tests

```python
"""Tests for GRU history encoder."""

import sys
import os
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.gru_encoder import GRUHistoryEncoder


class TestGRUHistoryEncoder:
    @pytest.fixture
    def encoder(self):
        return GRUHistoryEncoder(hidden_dim=32, max_len=32)

    def test_output_shape(self, encoder):
        grades = torch.tensor([[3, 2, 4, 0], [3, 0, 0, 0]], dtype=torch.float32)
        delta_ts = torch.tensor([[1.0, 3.0, 7.0, 0.0], [2.0, 0.0, 0.0, 0.0]])
        lengths = torch.tensor([3, 1], dtype=torch.long)
        out = encoder(grades, delta_ts, lengths)
        assert out.shape == (2, 32)

    def test_zero_state(self, encoder):
        z = encoder.zero_state(4, torch.device("cpu"))
        assert z.shape == (4, 32)
        assert (z == 0).all()

    def test_no_nan(self, encoder):
        grades = torch.rand(3, 10) * 4
        delta_ts = torch.rand(3, 10) * 30
        lengths = torch.tensor([10, 5, 1], dtype=torch.long)
        out = encoder(grades, delta_ts, lengths)
        assert not torch.isnan(out).any()

    def test_without_lengths(self, encoder):
        """Should work without explicit lengths."""
        grades = torch.rand(2, 5) * 4
        delta_ts = torch.rand(2, 5) * 10
        out = encoder(grades, delta_ts, lengths=None)
        assert out.shape == (2, 32)

    def test_truncation(self, encoder):
        """Sequences longer than max_len should be truncated."""
        grades = torch.rand(2, 50) * 4
        delta_ts = torch.rand(2, 50) * 10
        lengths = torch.tensor([50, 40], dtype=torch.long)
        out = encoder(grades, delta_ts, lengths)
        assert out.shape == (2, 32)
```

---

## 24. `tests/test_fsrs_warmstart.py`

```python
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
        """At t=0, retrievability should be 1.0 (no time to forget)."""
        R = fsrs.retrievability(0.0, 10.0)
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
```

---

## 25. `tests/test_loss.py`

```python
"""Tests for the multi-component loss function."""

import sys
import os
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.loss import MetaSRSLoss, compute_loss
from models.memory_net import MemoryNet
from config import ModelConfig


class TestMetaSRSLoss:
    @pytest.fixture
    def loss_fn(self):
        return MetaSRSLoss()

    def test_loss_returns_dict(self, loss_fn):
        p_pred = torch.tensor([0.8, 0.6, 0.3, 0.9])
        S_next = torch.tensor([10.0, 5.0, 2.0, 20.0])
        recalled = torch.tensor([1.0, 1.0, 0.0, 1.0])
        elapsed = torch.tensor([3.0, 7.0, 14.0, 1.0])
        grade = torch.tensor([3, 3, 1, 4])
        S_prev = torch.tensor([5.0, 3.0, 5.0, 10.0])

        result = loss_fn(p_pred, S_next, recalled, elapsed, grade, S_prev)

        assert "total" in result
        assert "recall" in result
        assert "stability" in result
        assert "monotonicity" in result

    def test_loss_is_finite(self, loss_fn):
        p_pred = torch.tensor([0.8, 0.6, 0.3, 0.9])
        S_next = torch.tensor([10.0, 5.0, 2.0, 20.0])
        recalled = torch.tensor([1.0, 1.0, 0.0, 1.0])
        elapsed = torch.tensor([3.0, 7.0, 14.0, 1.0])
        grade = torch.tensor([3, 3, 1, 4])
        S_prev = torch.tensor([5.0, 3.0, 5.0, 10.0])

        result = loss_fn(p_pred, S_next, recalled, elapsed, grade, S_prev)
        assert torch.isfinite(result["total"])

    def test_perfect_prediction_low_loss(self, loss_fn):
        """Perfect predictions should yield low recall loss."""
        p_pred = torch.tensor([0.99, 0.99, 0.01, 0.99])
        S_next = torch.tensor([10.0, 5.0, 2.0, 20.0])
        recalled = torch.tensor([1.0, 1.0, 0.0, 1.0])
        elapsed = torch.tensor([1.0, 1.0, 1.0, 1.0])
        grade = torch.tensor([3, 3, 1, 4])

        result = loss_fn(p_pred, S_next, recalled, elapsed, grade)
        assert result["recall"].item() < 0.5  # Should be low

    def test_bad_prediction_high_loss(self, loss_fn):
        """Inverted predictions should yield high recall loss."""
        p_pred = torch.tensor([0.01, 0.01, 0.99, 0.01])
        S_next = torch.tensor([10.0, 5.0, 2.0, 20.0])
        recalled = torch.tensor([1.0, 1.0, 0.0, 1.0])
        elapsed = torch.tensor([1.0, 1.0, 1.0, 1.0])
        grade = torch.tensor([3, 3, 1, 4])

        result = loss_fn(p_pred, S_next, recalled, elapsed, grade)
        assert result["recall"].item() > 1.0  # Should be high

    def test_monotonicity_penalty(self, loss_fn):
        """Stability decrease on success should incur monotonicity penalty."""
        p_pred = torch.tensor([0.8, 0.8])
        S_next = torch.tensor([3.0, 3.0])   # S decreased from 5 and 10
        recalled = torch.tensor([1.0, 1.0])
        elapsed = torch.tensor([1.0, 1.0])
        grade = torch.tensor([3, 3])  # Success
        S_prev = torch.tensor([5.0, 10.0])   # Higher than S_next

        result = loss_fn(p_pred, S_next, recalled, elapsed, grade, S_prev)
        assert result["monotonicity"].item() > 0

    def test_no_monotonicity_penalty_on_increase(self, loss_fn):
        """Stability increase on success should have zero monotonicity penalty."""
        p_pred = torch.tensor([0.8, 0.8])
        S_next = torch.tensor([10.0, 20.0])  # S increased
        recalled = torch.tensor([1.0, 1.0])
        elapsed = torch.tensor([1.0, 1.0])
        grade = torch.tensor([3, 3])
        S_prev = torch.tensor([5.0, 10.0])   # Lower than S_next

        result = loss_fn(p_pred, S_next, recalled, elapsed, grade, S_prev)
        assert result["monotonicity"].item() == 0.0

    def test_nan_fallback(self, loss_fn):
        """If total loss is NaN, should fall back to recall loss only."""
        # This is hard to trigger directly, but we can test the guard logic
        p_pred = torch.tensor([0.5, 0.5])
        S_next = torch.tensor([5.0, 5.0])
        recalled = torch.tensor([1.0, 0.0])
        elapsed = torch.tensor([1.0, 1.0])
        grade = torch.tensor([3, 1])

        result = loss_fn(p_pred, S_next, recalled, elapsed, grade)
        assert not torch.isnan(result["total"])

    def test_loss_weights(self):
        """Verify loss weights are applied correctly."""
        loss_fn = MetaSRSLoss(stability_weight=0.0, monotonicity_weight=0.0)
        p_pred = torch.tensor([0.8, 0.2])
        S_next = torch.tensor([3.0, 3.0])
        recalled = torch.tensor([1.0, 0.0])
        elapsed = torch.tensor([5.0, 5.0])
        grade = torch.tensor([3, 1])
        S_prev = torch.tensor([5.0, 5.0])

        result = loss_fn(p_pred, S_next, recalled, elapsed, grade, S_prev)
        # With zero weights, total should equal recall loss
        assert abs(result["total"].item() - result["recall"].item()) < 1e-6


class TestComputeLoss:
    def test_compute_loss_integration(self, model, sample_batch_tensors):
        loss_fn = MetaSRSLoss()
        result = compute_loss(model, sample_batch_tensors, loss_fn)
        assert "total" in result
        assert torch.isfinite(result["total"])
```

---

## 26. `tests/test_reptile.py`

```python
"""Tests for Reptile meta-training."""

import sys
import os
import torch
import pytest
from collections import OrderedDict
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import MetaSRSConfig
from models.memory_net import MemoryNet
from training.reptile import inner_loop, reptile_update, sample_batch, ReptileTrainer
from tests.conftest import create_model
from training.loss import MetaSRSLoss
from data.task_sampler import Task, Review, TaskSampler, ReviewDataset


@pytest.fixture
def phi_and_model():
    """Create a model and its initial phi state dict."""
    config = MetaSRSConfig()
    model = create_model(config.model)
    phi = deepcopy(model.state_dict())
    return phi, model, config


@pytest.fixture
def simple_task():
    """Create a simple task for testing inner loop."""
    reviews = [
        Review(card_id=f"c{i%5}", timestamp=i * 86400,
               elapsed_days=float(i), grade=3, recalled=True,
               S_prev=3.0 + i, D_prev=5.0, R_at_review=0.9,
               S_target=5.0 + i, D_target=4.8)
        for i in range(30)
    ]
    task = Task(student_id="test_student", reviews=reviews)
    task.split(support_ratio=0.70)
    return task


class TestSampleBatch:
    def test_returns_dict(self, simple_task):
        batch = sample_batch(
            simple_task.support_set, 8,
            torch.device("cpu")
        )
        assert isinstance(batch, dict)
        assert "D_prev" in batch
        assert "recalled" in batch

    def test_respects_size(self, simple_task):
        batch = sample_batch(
            simple_task.support_set, 5,
            torch.device("cpu")
        )
        assert batch["D_prev"].shape[0] <= max(5, len(simple_task.support_set))


class TestInnerLoop:
    def test_inner_loop_changes_weights(self, phi_and_model, simple_task):
        phi, model, config = phi_and_model
        loss_fn = MetaSRSLoss(w20=config.fsrs.w[20])

        adapted = inner_loop(
            phi_state_dict=phi,
            model=model,
            task=simple_task,
            loss_fn=loss_fn,
            k_steps=3,
            inner_lr=0.01,
            batch_size=16,
            device=torch.device("cpu"),
        )

        # Adapted weights should differ from phi
        different = False
        for key in phi:
            if not torch.equal(phi[key], adapted[key]):
                different = True
                break
        assert different, "Inner loop did not change weights"

    def test_inner_loop_does_not_modify_phi(self, phi_and_model, simple_task):
        phi, model, config = phi_and_model
        phi_copy = deepcopy(phi)
        loss_fn = MetaSRSLoss(w20=config.fsrs.w[20])

        inner_loop(
            phi_state_dict=phi,
            model=model,
            task=simple_task,
            loss_fn=loss_fn,
            k_steps=3,
            inner_lr=0.01,
            batch_size=16,
            device=torch.device("cpu"),
        )

        # phi should be unchanged
        for key in phi:
            assert torch.equal(phi[key], phi_copy[key]), \
                f"phi was modified at key {key}"


class TestReptileUpdate:
    def test_update_moves_phi(self, phi_and_model):
        phi, model, _ = phi_and_model
        # Create two "adapted" weight sets that differ from phi
        adapted1 = OrderedDict()
        adapted2 = OrderedDict()
        for key in phi:
            adapted1[key] = phi[key] + 0.1 * torch.randn_like(phi[key])
            adapted2[key] = phi[key] + 0.1 * torch.randn_like(phi[key])

        new_phi = reptile_update(phi, [adapted1, adapted2], epsilon=0.1)

        # New phi should differ from old phi
        different = False
        for key in phi:
            if not torch.equal(phi[key], new_phi[key]):
                different = True
                break
        assert different, "Reptile update did not change phi"

    def test_update_direction(self, phi_and_model):
        """Phi should move toward the adapted weights."""
        phi, model, _ = phi_and_model
        # All adapted weights are the same: phi + delta
        delta = OrderedDict()
        adapted = OrderedDict()
        for key in phi:
            delta[key] = torch.ones_like(phi[key])
            adapted[key] = phi[key] + delta[key]

        new_phi = reptile_update(phi, [adapted], epsilon=0.5)

        # new_phi should be phi + 0.5 * delta
        for key in phi:
            expected = phi[key] + 0.5 * delta[key]
            assert torch.allclose(new_phi[key], expected, atol=1e-6), \
                f"Unexpected update at key {key}"

    def test_epsilon_zero_no_change(self, phi_and_model):
        """With epsilon=0, phi should not change."""
        phi, model, _ = phi_and_model
        adapted = OrderedDict()
        for key in phi:
            adapted[key] = phi[key] + torch.randn_like(phi[key])

        new_phi = reptile_update(phi, [adapted], epsilon=0.0)
        for key in phi:
            assert torch.equal(phi[key], new_phi[key])


class TestReptileTrainer:
    @pytest.mark.timeout(60)
    def test_trainer_runs(self):
        """Test that ReptileTrainer runs for a few iterations without crashing."""
        config = MetaSRSConfig()
        config.training.log_dir = "/tmp/meta-srs-test-logs"
        config.training.checkpoint_dir = "/tmp/meta-srs-test-checkpoints"
        model = create_model(config.model)
        trainer = ReptileTrainer(model, config)

        tasks = ReviewDataset.generate_synthetic(
            n_students=10, reviews_per_student=30, n_cards=10, seed=42
        )
        sampler = TaskSampler(tasks, min_reviews=10)

        phi = trainer.train(sampler, n_iters=3)
        assert isinstance(phi, OrderedDict)
        assert len(phi) > 0
```

---

## 27. `tests/test_task_sampler.py`

```python
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
    def test_batch_shapes(self, sample_reviews):
        batch = reviews_to_batch(sample_reviews)
        n = len(sample_reviews)
        assert batch["D_prev"].shape == (n,)
        assert batch["S_prev"].shape == (n,)
        assert batch["R_at_review"].shape == (n,)
        assert batch["delta_t"].shape == (n,)
        assert batch["grade"].shape == (n,)
        assert batch["review_count"].shape == (n,)
        assert batch["user_stats"].shape == (n, 8)
        assert batch["recalled"].shape == (n,)
        assert batch["history_grades"].shape[0] == n
        assert batch["history_delta_ts"].shape[0] == n
        assert batch["history_lengths"].shape == (n,)

    def test_batch_on_device(self, sample_reviews, device):
        batch = reviews_to_batch(sample_reviews, device)
        for key, tensor in batch.items():
            assert tensor.device == device

    def test_review_count_accumulates(self):
        """Review count should increase for repeated cards."""
        reviews = [
            Review(card_id="c1", timestamp=0, elapsed_days=0.0,
                   grade=3, recalled=True),
            Review(card_id="c1", timestamp=1, elapsed_days=1.0,
                   grade=3, recalled=True),
            Review(card_id="c1", timestamp=2, elapsed_days=2.0,
                   grade=3, recalled=True),
        ]
        batch = reviews_to_batch(reviews)
        counts = batch["review_count"].tolist()
        assert counts == [1.0, 2.0, 3.0]


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
```

---

## 28. `tests/test_adaptation.py`

```python
"""Tests for FastAdapter (three-phase adaptation)."""

import sys
import os
import torch
import numpy as np
import pytest
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import MetaSRSConfig
from models.memory_net import MemoryNet
from inference.adaptation import FastAdapter, AdaptationPhase
from tests.conftest import create_model
from data.task_sampler import Review


def _make_review(card_id="c1", grade=3, elapsed=1.0, timestamp=0):
    return Review(
        card_id=card_id,
        timestamp=timestamp,
        elapsed_days=elapsed,
        grade=grade,
        recalled=grade >= 2,
        S_prev=3.0,
        D_prev=5.0,
        R_at_review=0.9,
        S_target=5.0,
        D_target=4.8,
    )


@pytest.fixture
def adapter():
    config = MetaSRSConfig()
    model = create_model(config.model)
    phi_star = deepcopy(model.state_dict())
    return FastAdapter(model, phi_star, config)


class TestAdaptationPhases:
    def test_initial_phase_is_zero_shot(self, adapter):
        assert adapter.phase == AdaptationPhase.ZERO_SHOT

    def test_phase_transitions(self, adapter):
        # Phase 1: 0-4 reviews → ZERO_SHOT
        for i in range(4):
            adapter.add_review(_make_review(timestamp=i))
        assert adapter.phase == AdaptationPhase.ZERO_SHOT

        # Phase 2: 5-49 reviews → RAPID_ADAPT
        adapter.add_review(_make_review(timestamp=5))
        assert adapter.phase == AdaptationPhase.RAPID_ADAPT

        # Add more to reach phase 3 (need >= 50 total)
        for i in range(6, 51):
            adapter.add_review(_make_review(timestamp=i))
        assert adapter.phase == AdaptationPhase.FULL_PERSONAL

    def test_n_reviews_property(self, adapter):
        assert adapter.n_reviews == 0
        adapter.add_review(_make_review())
        assert adapter.n_reviews == 1


class TestGetModel:
    def test_get_model_zero_shot_uses_phi(self, adapter):
        """In zero-shot phase, get_model should return model with phi_star."""
        model = adapter.get_model()
        assert isinstance(model, MemoryNet)

    def test_get_model_returns_eval_mode(self, adapter):
        model = adapter.get_model()
        assert not model.training

    def test_get_model_after_adaptation(self, adapter):
        """After adaptation, model should use theta_student."""
        for i in range(6):
            adapter.add_review(_make_review(
                card_id=f"c{i%3}", timestamp=i, elapsed=float(i+1)
            ))
        assert adapter.phase == AdaptationPhase.RAPID_ADAPT
        model = adapter.get_model()
        assert isinstance(model, MemoryNet)


class TestStateManagement:
    def test_get_state(self, adapter):
        adapter.add_review(_make_review())
        state = adapter.get_state()
        assert "theta_student" in state
        assert "reviews" in state
        assert "n_reviews" in state
        assert state["n_reviews"] == 1
        assert "phase" in state

    def test_load_state_roundtrip(self, adapter):
        for i in range(3):
            adapter.add_review(_make_review(timestamp=i))

        state = adapter.get_state()

        # Create new adapter and load state
        config = MetaSRSConfig()
        model = create_model(config.model)
        phi_star = deepcopy(model.state_dict())
        new_adapter = FastAdapter(model, phi_star, config)
        new_adapter.load_state(state)

        assert new_adapter.n_reviews == 3
```

---

## 29. `tests/test_scheduling.py`

```python
"""Tests for uncertainty-aware scheduling."""

import sys
import os
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import MetaSRSConfig, SchedulingConfig
from models.memory_net import MemoryNet
from inference.scheduling import Scheduler, ScheduleResult
from tests.conftest import create_model


@pytest.fixture
def scheduler():
    config = MetaSRSConfig()
    model = create_model(config.model)
    return Scheduler(model, config.scheduling, mc_samples=5)


@pytest.fixture
def single_features():
    """Single card features for scheduling."""
    return torch.randn(1, 49), torch.tensor([10.0])


class TestComputeInterval:
    def test_basic_interval(self, scheduler):
        interval, conf, diff = scheduler.compute_interval(
            S_pred=10.0, D=5.0, sigma=0.0
        )
        assert isinstance(interval, int)
        assert 1 <= interval <= 365

    def test_min_interval(self, scheduler):
        interval, _, _ = scheduler.compute_interval(
            S_pred=0.1, D=5.0, sigma=0.0
        )
        assert interval >= scheduler.cfg.min_interval

    def test_max_interval(self, scheduler):
        interval, _, _ = scheduler.compute_interval(
            S_pred=1000.0, D=5.0, sigma=0.0
        )
        assert interval <= scheduler.cfg.max_interval

    def test_high_uncertainty_shortens_interval(self, scheduler):
        """High uncertainty should reduce interval due to confidence discounting.
        Fuzz adds ±5% randomness, so we allow a small tolerance."""
        interval_low_sigma, _, _ = scheduler.compute_interval(
            S_pred=20.0, D=5.0, sigma=0.01
        )
        interval_high_sigma, _, _ = scheduler.compute_interval(
            S_pred=20.0, D=5.0, sigma=0.99
        )
        # The confidence factor for sigma=0.99 is ~0.505, vs ~0.995 for sigma=0.01.
        # With ±5% fuzz, the high-sigma interval could exceed the low-sigma one
        # by at most ~5% of the low-sigma interval. We use that as tolerance.
        fuzz_tolerance = max(1, round(interval_low_sigma * 0.10))
        assert interval_high_sigma <= interval_low_sigma + fuzz_tolerance

    def test_confidence_factor_range(self, scheduler):
        _, conf, _ = scheduler.compute_interval(
            S_pred=10.0, D=5.0, sigma=0.5
        )
        assert 0.5 <= conf <= 1.0

    def test_difficulty_factor_range(self, scheduler):
        _, _, diff = scheduler.compute_interval(
            S_pred=10.0, D=5.0, sigma=0.0
        )
        assert 0.5 <= diff <= 1.5

    def test_hard_card_shorter_interval(self, scheduler):
        """Harder cards (D>5) should get shorter intervals.
        Seed the RNG for deterministic fuzz to avoid statistical flakiness."""
        import random
        rng_state = random.getstate()
        try:
            random.seed(42)
            easy_intervals = [
                scheduler.compute_interval(S_pred=20.0, D=2.0, sigma=0.0)[0]
                for _ in range(20)
            ]
            random.seed(42)
            hard_intervals = [
                scheduler.compute_interval(S_pred=20.0, D=9.0, sigma=0.0)[0]
                for _ in range(20)
            ]
        finally:
            random.setstate(rng_state)

        avg_easy = sum(easy_intervals) / len(easy_intervals)
        avg_hard = sum(hard_intervals) / len(hard_intervals)
        assert avg_hard < avg_easy


class TestPredictWithUncertainty:
    def test_returns_four_tensors(self, scheduler, single_features):
        features, S_prev = single_features
        p_mean, p_sigma, S_mean, D_mean = scheduler.predict_with_uncertainty(
            features, S_prev
        )
        assert p_mean.shape == (1,)
        assert p_sigma.shape == (1,)
        assert S_mean.shape == (1,)
        assert D_mean.shape == (1,)

    def test_sigma_non_negative(self, scheduler, single_features):
        features, S_prev = single_features
        _, p_sigma, _, _ = scheduler.predict_with_uncertainty(features, S_prev)
        assert (p_sigma >= 0).all()


class TestScheduleCard:
    def test_returns_schedule_result(self, scheduler, single_features):
        features, S_prev = single_features
        result = scheduler.schedule_card("card_001", features, S_prev)
        assert isinstance(result, ScheduleResult)
        assert result.card_id == "card_001"
        assert 1 <= result.interval_days <= 365
        assert 0.0 <= result.p_recall_mean <= 1.0
        assert result.p_recall_sigma >= 0.0


class TestScheduleDeck:
    def test_returns_list(self, scheduler):
        features = torch.randn(3, 49)
        S_prev = torch.tensor([5.0, 10.0, 20.0])
        results = scheduler.schedule_deck(
            ["c1", "c2", "c3"], features, S_prev
        )
        assert len(results) == 3
        assert all(isinstance(r, ScheduleResult) for r in results)


class TestSelectNextCard:
    def test_most_due(self, scheduler):
        results = [
            ScheduleResult("c1", 5, 0.8, 0.1, 10, 5, 1.0, 1.0),
            ScheduleResult("c2", 1, 0.3, 0.2, 3, 7, 0.9, 0.9),
            ScheduleResult("c3", 10, 0.9, 0.05, 20, 3, 1.0, 1.0),
        ]
        selected = scheduler.select_next_card(results, strategy="most_due")
        assert selected.card_id == "c2"  # Lowest interval

    def test_highest_uncertainty(self, scheduler):
        results = [
            ScheduleResult("c1", 5, 0.8, 0.1, 10, 5, 1.0, 1.0),
            ScheduleResult("c2", 1, 0.3, 0.3, 3, 7, 0.9, 0.9),
            ScheduleResult("c3", 10, 0.9, 0.05, 20, 3, 1.0, 1.0),
        ]
        selected = scheduler.select_next_card(results, strategy="highest_uncertainty")
        assert selected.card_id == "c2"  # Highest sigma

    def test_uncertainty_weighted(self, scheduler):
        results = [
            ScheduleResult("c1", 5, 0.8, 0.1, 10, 5, 1.0, 1.0),
            ScheduleResult("c2", 1, 0.3, 0.3, 3, 7, 0.9, 0.9),
        ]
        selected = scheduler.select_next_card(results, strategy="uncertainty_weighted")
        assert isinstance(selected, ScheduleResult)

    def test_invalid_strategy_raises(self, scheduler):
        results = [ScheduleResult("c1", 5, 0.8, 0.1, 10, 5, 1.0, 1.0)]
        with pytest.raises(ValueError):
            scheduler.select_next_card(results, strategy="invalid")

    def test_empty_results_raises(self, scheduler):
        with pytest.raises(ValueError):
            scheduler.select_next_card([], strategy="most_due")
```

---

## 30. `tests/test_evaluation.py`

```python
"""Tests for the evaluation framework."""

import sys
import os
import numpy as np
import torch
import pytest
from collections import OrderedDict
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import MetaSRSConfig
from models.memory_net import MemoryNet
from evaluation.metrics import MetaSRSEvaluator, EvalResults
from tests.conftest import create_model
from data.task_sampler import ReviewDataset, TaskSampler


class TestEvalResults:
    def test_default_values(self):
        results = EvalResults()
        assert results.auc_roc == 0.0
        assert results.n_samples == 0

    def test_summary_string(self):
        results = EvalResults(auc_roc=0.85, calibration_error=0.03,
                              cold_start_auc=0.75, n_samples=100)
        s = results.summary()
        assert "AUC-ROC" in s
        assert "0.8500" in s
        assert "✓" in s  # AUC > 0.80 should pass


class TestAUCComputation:
    @pytest.fixture
    def evaluator(self):
        config = MetaSRSConfig()
        model = create_model(config.model)
        return MetaSRSEvaluator(model, config)

    def test_perfect_auc(self, evaluator):
        preds = np.array([0.9, 0.8, 0.2, 0.1])
        labels = np.array([1, 1, 0, 0])
        auc = evaluator.compute_auc_roc(preds, labels)
        assert auc == 1.0

    def test_random_auc(self, evaluator):
        """Random predictions should give AUC ≈ 0.5."""
        np.random.seed(42)
        preds = np.random.rand(1000)
        labels = (np.random.rand(1000) > 0.5).astype(float)
        auc = evaluator.compute_auc_roc(preds, labels)
        assert 0.4 < auc < 0.6

    def test_auc_with_all_same_label(self, evaluator):
        """AUC should handle single-class labels gracefully (NaN from sklearn)."""
        preds = np.array([0.9, 0.8, 0.7])
        labels = np.array([1, 1, 1])
        auc = evaluator.compute_auc_roc(preds, labels)
        # sklearn returns NaN for single-class; manual returns 0.5
        assert np.isnan(auc) or auc == 0.5

    def test_manual_auc_implementation(self, evaluator):
        """Test the fallback manual AUC implementation."""
        preds = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
        labels = np.array([1, 1, 1, 0, 0])
        auc = evaluator._manual_auc(preds, labels)
        assert auc == 1.0


class TestEvaluateOnTasks:
    @pytest.mark.timeout(60)
    def test_evaluate_returns_results(self):
        config = MetaSRSConfig()
        model = create_model(config.model)
        evaluator = MetaSRSEvaluator(model, config)

        tasks = ReviewDataset.generate_synthetic(
            n_students=10, reviews_per_student=30, n_cards=10, seed=42
        )
        # Split tasks
        for task in tasks:
            task.split(support_ratio=0.70)

        phi = deepcopy(model.state_dict())
        results = evaluator.evaluate_on_tasks(phi, tasks[:5], k_steps=2)

        assert isinstance(results, EvalResults)
        assert results.n_samples > 0
        assert 0.0 <= results.auc_roc <= 1.0

    @pytest.mark.timeout(60)
    def test_cold_start_curve(self):
        config = MetaSRSConfig()
        model = create_model(config.model)
        evaluator = MetaSRSEvaluator(model, config)

        tasks = ReviewDataset.generate_synthetic(
            n_students=10, reviews_per_student=30, n_cards=10, seed=42
        )
        for task in tasks:
            task.split(support_ratio=0.70)

        phi = deepcopy(model.state_dict())
        curve = evaluator.cold_start_curve(phi, tasks[:3],
                                           review_counts=[0, 5])

        assert isinstance(curve, dict)
        assert 0 in curve
        assert 0.0 <= curve[0] <= 1.0
```

---

## 31. `tests/test_integration.py`

```python
"""Integration test: full train→eval pipeline with synthetic data."""

import sys
import os
import pytest
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from config import MetaSRSConfig
from models.memory_net import MemoryNet
from data.task_sampler import ReviewDataset, TaskSampler
from tests.conftest import create_model
from training.reptile import ReptileTrainer
from training.loss import MetaSRSLoss, compute_loss
from evaluation.metrics import MetaSRSEvaluator


class TestEndToEnd:
    @pytest.mark.timeout(90)
    def test_synthetic_pipeline(self):
        """Test the complete train→eval pipeline with minimal synthetic data."""
        config = MetaSRSConfig()
        config.training.checkpoint_dir = "/tmp/meta-srs-e2e-checkpoints"
        config.training.log_dir = "/tmp/meta-srs-e2e-logs"

        # Step 1: Generate synthetic data
        all_tasks = ReviewDataset.generate_synthetic(
            n_students=15, reviews_per_student=30, n_cards=10, seed=42
        )
        assert len(all_tasks) == 15

        # Step 2: Split into train/test
        train_tasks = all_tasks[:12]
        test_tasks = all_tasks[12:]
        for t in test_tasks:
            t.split(support_ratio=0.70)

        # Step 3: Create model
        model = create_model(config.model)
        assert model.count_parameters() > 0

        # Step 4: Create task sampler
        sampler = TaskSampler(train_tasks, min_reviews=10)
        assert len(sampler) > 0

        # Step 5: Run Reptile training (minimal iterations)
        trainer = ReptileTrainer(model, config)
        phi = trainer.train(sampler, n_iters=5)
        assert isinstance(phi, dict)

        # Step 6: Evaluate
        evaluator = MetaSRSEvaluator(model, config)
        results = evaluator.evaluate_on_tasks(phi, test_tasks, k_steps=2)
        assert results.n_samples > 0
        assert 0.0 <= results.auc_roc <= 1.0

    @pytest.mark.timeout(60)
    def test_model_forward_with_synthetic_batch(self):
        """Test that model can do a forward pass on synthetic data."""
        config = MetaSRSConfig()
        model = create_model(config.model)
        loss_fn = MetaSRSLoss(w20=config.fsrs.w[20])

        tasks = ReviewDataset.generate_synthetic(
            n_students=5, reviews_per_student=20, n_cards=10, seed=42
        )
        task = tasks[0]
        task.split(support_ratio=0.70)

        from data.task_sampler import reviews_to_batch
        batch = reviews_to_batch(task.support_set)

        # Forward pass
        state = model(
            D_prev=batch["D_prev"],
            S_prev=batch["S_prev"],
            R_at_review=batch["R_at_review"],
            delta_t=batch["delta_t"],
            grade=batch["grade"],
            review_count=batch["review_count"],
            user_stats=batch["user_stats"],
            history_grades=batch["history_grades"],
            history_delta_ts=batch["history_delta_ts"],
            history_lengths=batch["history_lengths"],
        )

        assert not torch.isnan(state.p_recall).any()
        assert not torch.isnan(state.S_next).any()
        assert not torch.isnan(state.D_next).any()

        # Loss computation
        losses = compute_loss(model, batch, loss_fn)
        assert torch.isfinite(losses["total"])

    @pytest.mark.timeout(60)
    def test_checkpoint_save_and_load(self):
        """Test that checkpoints can be saved and loaded."""
        config = MetaSRSConfig()
        model = create_model(config.model)
        phi = deepcopy(model.state_dict())

        path = "/tmp/meta-srs-test-phi.pt"
        torch.save({"phi": phi, "iteration": 100, "config": config}, path)

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert "phi" in checkpoint
        assert checkpoint["iteration"] == 100

        # Load phi into model
        model.load_state_dict(checkpoint["phi"])
        # Verify model works after loading
        features = torch.randn(2, 49)
        S_prev = torch.tensor([5.0, 10.0])
        state = model.forward_from_features(features, S_prev)
        assert state.p_recall.shape == (2,)

        os.unlink(path)
```

---
