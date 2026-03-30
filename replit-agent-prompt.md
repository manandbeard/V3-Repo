# Replit Agent Prompt: Recreate MetaSRS — Meta-Learning Spaced Repetition Engine

> **Goal**: Build "MetaSRS" — a Python/PyTorch meta-learning spaced repetition scheduling engine with a Flask REST API. The system uses Reptile meta-learning to find neural network initialization parameters (φ\*) that fast-adapt to any new student in 5–20 gradient steps, achieving recall-prediction AUC > 0.80 that surpasses FSRS-6 (~0.78) while solving the cold-start problem. It serves as the back end for a Node.js web app used by teachers in online classrooms.

---

## 1. Project Structure

Create this exact directory layout inside the Replit project:

```
meta-srs/
├── config.py                    # All hyperparameters (5 dataclass configs)
├── api.py                       # Flask REST API (main entry point for Replit)
├── train.py                     # Training orchestration script (CLI)
├── requirements.txt             # Full dependencies
├── requirements-web.txt         # Slim CPU-only web dependencies (for Replit)
├── pytest.ini                   # Test configuration
├── models/
│   ├── __init__.py
│   ├── memory_net.py            # MemoryNet (~35K-param neural model)
│   └── gru_encoder.py           # GRU history encoder
├── training/
│   ├── __init__.py
│   ├── loss.py                  # Multi-component loss function
│   ├── reptile.py               # Reptile meta-training outer/inner loops
│   └── fsrs_warmstart.py        # FSRS-6 baseline + warm-start pre-training
├── data/
│   ├── __init__.py
│   └── task_sampler.py          # Review/Task dataclasses, batch construction, synthetic data
├── inference/
│   ├── __init__.py
│   ├── adaptation.py            # Three-phase fast adaptation for new students
│   └── scheduling.py            # Uncertainty-aware interval scheduler (MC Dropout)
├── evaluation/
│   ├── __init__.py
│   └── metrics.py               # AUC-ROC, calibration, cold-start curve
└── tests/
    ├── __init__.py              # Empty
    ├── conftest.py              # Shared fixtures
    ├── test_config.py
    ├── test_memory_net.py
    ├── test_models.py
    ├── test_loss.py
    ├── test_fsrs_warmstart.py
    ├── test_task_sampler.py
    ├── test_reptile.py
    ├── test_adaptation.py
    ├── test_scheduling.py
    ├── test_evaluation.py
    ├── test_integration.py
    └── test_api.py
```

Also create these files in the **project root** (parent of meta-srs/):

```
.replit                          # Replit run configuration
replit.nix                       # Nix packages for Replit
.gitignore
```

---

## 2. Dependencies

### `meta-srs/requirements.txt` (full — for training)
```
torch>=2.0
numpy>=1.24
scikit-learn>=1.3
tqdm
tensorboard
onnx
onnxruntime
```

### `meta-srs/requirements-web.txt` (slim — for Replit deployment)
```
# MetaSRS Web — Slim dependencies for Replit deployment
# CPU-only PyTorch keeps install under ~300MB (vs ~2GB with CUDA)
--extra-index-url https://download.pytorch.org/whl/cpu
torch>=2.0,<3.0
numpy>=1.24,<3.0
flask>=3.0,<4.0
gunicorn>=21.0,<24.0
```

### `meta-srs/pytest.ini`
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
addopts = -v --tb=short --timeout=120
```

---

## 3. Replit Configuration

### `.replit`
```
run = "cd meta-srs && python api.py"
entrypoint = "meta-srs/api.py"

[env]
PORT = "5000"
PYTHONPATH = "meta-srs"

[nix]
channel = "stable-24_05"

[nix.packages]
python = "python312"
```

### `replit.nix`
```nix
{ pkgs }: {
  deps = [
    pkgs.python312
    pkgs.python312Packages.pip
  ];
}
```

### `.gitignore`
```
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/
.eggs/
*.egg
checkpoints/
logs/
*.pt
*.npz
.env
.pytest_cache/
node_modules/
.upm/
.config/
```

---

## 4. Configuration (`meta-srs/config.py`)

Create 5 nested dataclass configs. All hyperparameters are defined here — no magic numbers elsewhere.

```python
"""
MetaSRS Configuration — All hyperparameters.
"""

from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class ModelConfig:
    """MemoryNet architecture."""
    input_dim: int = 49           # Feature vector dimension (4+4+1+8+32)
    hidden_dim: int = 128         # Hidden layer width
    gru_hidden_dim: int = 32      # GRU history encoder hidden state
    history_len: int = 32         # Max review history length for GRU
    user_stats_dim: int = 8       # User-level statistics features
    dropout: float = 0.1          # Dropout rate (used for MC Dropout uncertainty)
    mc_samples: int = 20          # Monte Carlo forward passes for uncertainty


@dataclass
class FSRSConfig:
    """FSRS-6 default population parameters — 21 optimizable weights (w0..w20)."""
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
    """Reptile meta-training hyperparameters."""
    # Outer loop
    n_iters: int = 50_000
    meta_batch_size: int = 16
    outer_lr_start: float = 1e-3
    outer_lr_end: float = 1e-4
    epsilon_start: float = 0.10
    epsilon_end: float = 0.01

    # Inner loop
    inner_lr: float = 0.01
    inner_steps_phase1: int = 5
    inner_steps_phase2: int = 20
    task_batch_size: int = 32
    support_ratio: float = 0.70

    # Loss weights
    stability_loss_weight: float = 0.10
    monotonicity_loss_weight: float = 0.01

    # Warm-start
    warmstart_epochs: int = 10
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
    """Online fast-adaptation settings."""
    phase1_threshold: int = 5
    phase2_threshold: int = 50
    phase1_k_steps: int = 5
    phase2_k_steps: int = 10
    phase3_k_steps: int = 20
    phase3_inner_lr: float = 0.02
    adapt_every_n_reviews: int = 5
    streaming_after: int = 50


@dataclass
class SchedulingConfig:
    """Interval computation settings."""
    desired_retention: float = 0.90
    min_interval: int = 1
    max_interval: int = 365
    fuzz_range: tuple = (0.95, 1.05)
    difficulty_centre: float = 5.0
    difficulty_slope: float = 0.05
    uncertainty_discount: float = 0.5


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

## 5. Models

### 5.1 GRU History Encoder (`meta-srs/models/gru_encoder.py`)

A 1-layer GRU that encodes a card's review history `[(grade, delta_t), ...]` into a fixed-size 32-dim context vector.

**Key details:**
- Input: 2 features per timestep — `grade/4.0` (normalised) and `log(delta_t + 1)` (log-scale)
- Has a small `input_proj = Linear(2, 2)` before the GRU
- Truncates sequences to `max_len` (keeps most recent reviews)
- Uses `pack_padded_sequence` when `lengths` is provided
- Returns `h_n.squeeze(0)` — final hidden state shape `(batch, hidden_dim)`
- Has a `zero_state(batch_size, device)` method returning zeros

### 5.2 MemoryNet (`meta-srs/models/memory_net.py`)

The core neural model (~35K parameters). A small feedforward network that predicts memory-state transitions.

**Input feature vector (dim = 49):**
```
x = concat([
    D_prev / 10.0,               # normalised difficulty          → 1
    log(S_prev),                  # log stability                  → 1
    R_at_review,                  # retrievability [0,1]           → 1
    log(delta_t + 1),             # log days since last review     → 1
    grade_onehot,                 # [Again, Hard, Good, Easy]      → 4
    log(review_count),            # log-scaled review count        → 1
    user_stats,                   # 8 user-level statistics        → 8
    gru_context_h,               # GRU hidden from review history → 32
])                                                            Total: 49
```

**Architecture:**
```
Linear(49, 128) + LayerNorm(128) + GELU + Dropout(0.1)
Linear(128, 128) + LayerNorm(128) + GELU + Dropout(0.1)
Linear(128, 64) + GELU + Dropout(0.1)
→ 3 output heads (each Linear(64, 1)):
    S_next   = Softplus(head) * S_prev   # multiplicative, clamped [0.001, 36500]
    D_next   = Sigmoid(head) * 9 + 1     # range [1, 10]
    p_recall = Sigmoid(head)             # range [0, 1]
```

**Output:** `MemoryState` NamedTuple with fields `S_next`, `D_next`, `p_recall`.

**Key methods:**
- `build_features(D_prev, S_prev, R_at_review, delta_t, grade, review_count, user_stats, history_grades=None, history_delta_ts=None, history_lengths=None) → (batch, 49)` — assembles feature vector. Has dynamic padding/truncation safety net.
- `forward_from_features(x, S_prev) → MemoryState` — runs network from pre-assembled features.
- `forward(D_prev, S_prev, ...) → MemoryState` — convenience: build_features + forward_from_features.
- `predict_recall(features, S_prev) → p_recall`
- `predict_stability(features, S_prev) → S_next`
- `count_parameters() → int`

### 5.3 `meta-srs/models/__init__.py`
```python
from .memory_net import MemoryNet
from .gru_encoder import GRUHistoryEncoder

__all__ = ["MemoryNet", "GRUHistoryEncoder"]
```

---

## 6. Data Pipeline (`meta-srs/data/task_sampler.py`)

### 6.1 Review dataclass
```python
@dataclass
class Review:
    card_id: str
    timestamp: int
    elapsed_days: float
    grade: int               # 1=Again, 2=Hard, 3=Good, 4=Easy
    recalled: bool           # grade >= 2
    S_prev: float = 1.0
    D_prev: float = 5.0
    R_at_review: float = 1.0
    S_target: float = 1.0   # Ground-truth next S (for warm-start)
    D_target: float = 5.0   # Ground-truth next D
```

### 6.2 Task dataclass
A Task represents one student's complete review history split into support/query sets.
- `student_id: str`, `reviews: List[Review]`
- `support_set`, `query_set` (populated by `split(support_ratio=0.70)`)
- `get_review_history(card_id, up_to_idx) → List[Review]`
- `unique_cards → List[str]` (property)

### 6.3 `reviews_to_batch(reviews, device=cpu, history_len=32) → Dict[str, Tensor]`
Converts a list of Review objects into a tensor dict for MemoryNet. Returns:
- `D_prev`, `S_prev`, `R_at_review`, `delta_t` — (batch,) tensors
- `grade` — (batch,) long tensor
- `review_count` — (batch,) accumulated per-card count
- `user_stats` — (batch, 8) with mean_D at `[:,0]` and log(mean_S) at `[:,1]`, rest zeros
- `recalled` — (batch,) float tensor
- `S_target`, `D_target` — (batch,) warm-start targets
- `history_grades`, `history_delta_ts` — (batch, history_len) for GRU
- `history_lengths` — (batch,) long tensor, clamped min=1

**Important detail:** The function builds per-card running histories as it iterates through reviews, so each review sees only its *prior* reviews of the same card.

### 6.4 TaskSampler
- Filters tasks with `< min_reviews` (default 10)
- Splits each task on init
- `sample(batch_size=16) → List[Task]` uses `rng.choices` (with replacement)

### 6.5 ReviewDataset
- `from_csv(csv_path) → List[Task]` — loads CSV with columns: student_id, card_id, timestamp, elapsed_days, grade
- `generate_synthetic(n_students=500, reviews_per_student=100, n_cards=200, seed=42) → List[Task]` — uses FSRS-6 to simulate realistic memory dynamics. Grade probabilities: first review = [15%, 20%, 50%, 15%]; recall = [20%, 60%, 20% for Hard/Good/Easy]; lapse = grade 1.

### 6.6 `meta-srs/data/__init__.py`
```python
from .task_sampler import Task, TaskSampler, ReviewDataset
__all__ = ["Task", "TaskSampler", "ReviewDataset"]
```

---

## 7. Training

### 7.1 FSRS-6 Baseline & Warm-Start (`meta-srs/training/fsrs_warmstart.py`)

#### FSRS6 class
Implements the 21-parameter FSRS-6 model:

- **Forgetting curve (power-law):** `R(t, S) = (0.9^(1/S))^(t^w20)`
- **Initial stability:** `S_0 = w[grade-1]` (w0–w3)
- **Initial difficulty:** `D_0 = w4 - (grade - 3) * w5`
- **Stability after success:** `S' = S * (e^w8 * (11-D) * S^(-w9) * (e^(w10*(1-R)) - 1) * grade_modifier + 1)` where grade_modifier = w15 for Hard, w16 for Easy, 1.0 for Good
- **Stability after lapse:** `S_f = w11 * D^(-w12) * ((S+1)^w13 - 1) * e^(w14*(1-R))`
- **Difficulty update:** `dD = -w6 * (grade-3)`, then `D'' = D + dD * (10-D)/9`, then mean-revert: `D' = w5 * D0(Easy) + (1-w5) * D''`, clamped to [1, 10]
- `step(S, D, elapsed_days, grade) → (S_next, D_next, R_at_review)` — full review event processing
- `simulate_student(reviews) → List[dict]` — augments reviews with S, D, R fields
- `retrievability_batch(t, S)` — vectorised PyTorch version

#### `warm_start_from_fsrs6(model, dataset, config, device)`
Pre-trains MemoryNet to reproduce FSRS-6 predictions. Loss = `BCE(p_recall, R_target) + 0.5 * MSE(log(S_next+1), log(S_target+1)) + 0.5 * MSE(D_next, D_target)`. Uses Adam with `warmstart_lr`, grad clip 1.0, for `warmstart_epochs` epochs.

### 7.2 Multi-Component Loss (`meta-srs/training/loss.py`)

#### MetaSRSLoss
Three loss components combined as `total = recall_L + 0.10 * stable_L + 0.01 * mono_L`:

1. **PRIMARY — Recall prediction:** `BCEWithLogitsLoss` on manually computed logits from `p_clamped = p.clamp(1e-6, 1-1e-6)`, `logits = log(p/(1-p))`
2. **AUXILIARY 1 — Stability-curve consistency:** Computed in log-space: `log_R = log(0.9) * t^w20 / S_pred`, then `R_from_S = exp(clamp(log_R, -20, 0))`, then `MSE(R_from_S, recalled)`
3. **AUXILIARY 2 — Monotonicity:** `ReLU(-(S_next - S_prev)) * success_mask).mean()` — penalises stability decrease on successful recall
4. **NaN guard:** Falls back to recall_L only if total is NaN

#### `compute_loss(model, batch, loss_fn) → Dict` 
Convenience function: runs model forward on batch dict, then loss_fn.

### 7.3 Reptile Meta-Training (`meta-srs/training/reptile.py`)

#### `sample_batch(reviews, size, device) → Dict`
Samples min(size, len(reviews)) reviews, converts via `reviews_to_batch`.

#### `inner_loop(phi_state_dict, model, task, loss_fn, k_steps, inner_lr, batch_size, device) → OrderedDict`
Per-student adaptation:
1. `model.load_state_dict(phi)` (copies, doesn't modify phi)
2. Adam optimizer, `k_steps` gradient steps on `task.support_set`
3. `clip_grad_norm_(1.0)` each step
4. Returns `deepcopy(model.state_dict())`

#### `reptile_update(phi, adapted_weights, epsilon) → OrderedDict`
`new_phi[key] = phi[key] + epsilon * mean(W_i[key] - phi[key])` for all keys.

#### ReptileTrainer class
Full orchestrator with:
- Linear epsilon decay, cosine outer LR decay
- TensorBoard logging (optional, graceful ImportError)
- Per-iteration: sample tasks → inner_loop each → reptile_update
- Evaluates adapted model on query sets for monitoring
- Periodic checkpointing (saves `{phi, iteration, config}`)
- Saves final `phi_star.pt`

### 7.4 `meta-srs/training/__init__.py`
```python
from .loss import compute_loss, MetaSRSLoss
from .reptile import inner_loop, reptile_update, ReptileTrainer
from .fsrs_warmstart import FSRS6, warm_start_from_fsrs6

__all__ = [
    "compute_loss", "MetaSRSLoss",
    "inner_loop", "reptile_update", "ReptileTrainer",
    "FSRS6", "warm_start_from_fsrs6",
]
```

---

## 8. Inference

### 8.1 Three-Phase Fast Adaptation (`meta-srs/inference/adaptation.py`)

#### AdaptationPhase enum
```python
class AdaptationPhase(Enum):
    ZERO_SHOT = auto()      # Phase 1: 0 reviews, use phi* directly
    RAPID_ADAPT = auto()    # Phase 2: 5–50 reviews, periodic adaptation
    FULL_PERSONAL = auto()  # Phase 3: 50+ reviews, streaming updates
```

#### FastAdapter class
Manages per-student personalisation:

- **Phase 1 (ZERO_SHOT, < 5 reviews):** No adaptation, use `phi_star` directly
- **Phase 2 (RAPID_ADAPT, 5–49 reviews):** Every 5 reviews, run k=5 gradient steps from `phi_star` on all student reviews. Always adapts from phi_star (not from previous theta).
- **Phase 3 (FULL_PERSONAL, 50+ reviews):** Every 5 reviews, run k=20 gradient steps at higher LR (0.02) from current `theta_student`. After 50 reviews, also does streaming single gradient steps on last 8 reviews.

**Key methods:**
- `add_review(review)` — appends review and triggers adaptation if thresholds met
- `get_model() → MemoryNet` — returns model with appropriate params (phi_star for Phase 1, theta_student for Phases 2-3), always in eval mode
- `get_state() → dict` / `load_state(state)` — serialisation for persistence

### 8.2 Uncertainty-Aware Scheduler (`meta-srs/inference/scheduling.py`)

#### ScheduleResult dataclass
```python
@dataclass
class ScheduleResult:
    card_id: str
    interval_days: int
    p_recall_mean: float
    p_recall_sigma: float
    S_pred: float
    D_pred: float
    confidence_factor: float
    difficulty_factor: float
```

#### Scheduler class
- `predict_with_uncertainty(features, S_prev)` — MC Dropout: runs model `mc_samples` times (default 20) with `model.train()` to keep dropout active, under `torch.no_grad()`. Returns mean and std of p_recall, S_next, D_next.
- `compute_interval(S_pred, D, sigma)` — `base = S_pred`, `confidence_factor = 1.0 - 0.5 * min(sigma, 1.0)` clamped [0.5, 1.0], `difficulty_factor = 1.0 - 0.05 * (D - 5.0)` clamped [0.5, 1.5], `fuzz = uniform(0.95, 1.05)`, `interval = round(base * conf * diff * fuzz)` clamped [1, 365].
- `schedule_card(card_id, features, S_prev) → ScheduleResult`
- `schedule_deck(card_ids, features, S_prev) → List[ScheduleResult]`
- `select_next_card(results, strategy)` — strategies: `"most_due"` (min interval), `"highest_uncertainty"` (max sigma), `"uncertainty_weighted"` (score = `(1-p_recall) + 0.5*sigma`, softmax with temperature=1/3, multinomial sampling).

### 8.3 `meta-srs/inference/__init__.py`
```python
from .adaptation import FastAdapter, AdaptationPhase
from .scheduling import Scheduler, ScheduleResult
__all__ = ["FastAdapter", "AdaptationPhase", "Scheduler", "ScheduleResult"]
```

---

## 9. Evaluation (`meta-srs/evaluation/metrics.py`)

### EvalResults dataclass
Fields: `auc_roc`, `calibration_error`, `cold_start_auc`, `adaptation_speed`, `retention_30d`, `rmse_stability`, `recall_loss`, `n_samples`, `phase1_auc`, `phase2_auc`, `phase3_auc`.
Has `summary()` method that prints targets with ✓/✗ indicators.

### MetaSRSEvaluator class
- `compute_auc_roc(predictions, labels)` — tries sklearn first, falls back to manual trapezoidal AUC
- `_manual_auc(scores, labels)` — simple AUC without sklearn; returns 0.5 for single-class
- `evaluate_on_tasks(phi, tasks, k_steps=5)` — for each task: (1) cold-start eval with phi, (2) inner-loop adapt on support, (3) eval on query. Guards against NaN with `nan_to_num`.
- `cold_start_curve(phi, tasks, review_counts=[0,5,10,20,30,50])` — AUC at different adaptation levels
- `ablation_study(phi_variants, tasks)` — evaluates multiple phi variants

### `meta-srs/evaluation/__init__.py`
```python
from .metrics import MetaSRSEvaluator, EvalResults
__all__ = ["MetaSRSEvaluator", "EvalResults"]
```

---

## 10. Flask REST API (`meta-srs/api.py`)

This is the **main entry point** for Replit. The API serves MetaSRS predictions over HTTP for a Node.js frontend.

### App Factory: `create_app(checkpoint_path=None, config=None)`

Creates and configures a Flask app with:

1. **Model initialisation:** Creates MemoryNet from config (filtering out `mc_samples` which MemoryNet doesn't accept). Loads checkpoint if available, otherwise uses fresh model.
2. **Per-student state:** In-memory dict `_adapters: dict[str, FastAdapter]` with a `threading.Lock` for thread safety. `_get_adapter(student_id)` creates a new FastAdapter (with its own MemoryNet copy) if one doesn't exist.
3. **FSRS-6 instance** for computing initial card states.

### Helper functions (inside create_app closure):
- `_card_state(adapter, card_id, elapsed_days, grade=3)` — looks up S_prev, D_prev, R_at_review from student's review history for a card. Falls back to FSRS-6 initial values for unseen cards.
- `_schedule_result_to_dict(r)` — serialises ScheduleResult to JSON-safe dict with 4-decimal rounding.
- `_validate_grade(grade)` — returns int 1-4 or None.
- `DEFAULT_GRADE = 3` module-level constant.

### Endpoints:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check. Returns `{status, model_params, active_students, timestamp}` |
| POST | `/api/review` | Submit a review. Body: `{student_id, card_id, grade, elapsed_days}`. Triggers adaptation, returns scheduling prediction. |
| POST | `/api/schedule` | Schedule a deck. Body: `{student_id, cards: [{card_id, elapsed_days}]}`. Returns scheduling for all cards. |
| GET | `/api/student/<id>/status` | Get student phase, n_reviews, unique_cards. |
| POST | `/api/student/<id>/reset` | Reset student personalisation. Deletes adapter. |
| POST | `/api/next-card` | Select next card. Body: `{student_id, cards, strategy?}`. Returns selected card via uncertainty-weighted exploration. |

### POST /api/review details:
1. Validate all fields (student_id, card_id, grade 1-4, elapsed_days >= 0)
2. Look up card state via `_card_state()`. If first review of card, set elapsed=0.
3. Compute FSRS-6 targets via `fsrs.step(S_prev, D_prev, elapsed_days, grade)`
4. Create Review object, call `adapter.add_review(review)` (triggers adaptation if thresholds met)
5. Get adapted model, create Scheduler, build features from review batch, schedule card
6. Return `{student_id, phase, n_reviews, schedule: {...}}`

### POST /api/schedule details:
1. For each card: look up state via `_card_state`, create Review with `DEFAULT_GRADE=3`
2. Batch all reviews, build features, `scheduler.schedule_deck()`
3. Return `{student_id, phase, n_reviews, cards: [...]}`

### Error handlers: 404 and 500 return JSON `{error: "..."}`.

### Module-level:
```python
app = create_app(
    checkpoint_path=os.environ.get("METASRS_CHECKPOINT", "checkpoints/phi_star.pt"),
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
```

---

## 11. Training Script (`meta-srs/train.py`)

CLI entry point for training (not used on Replit, but needed for model development):

```
python train.py --synthetic --n-students 100 --n-iters 500     # Quick test
python train.py --data reviews.csv --n-iters 50000              # Full training
python train.py --eval-only --eval-checkpoint checkpoints/phi_star.pt
```

Pipeline: Generate/load data → 80/20 student split → Create MemoryNet → FSRS-6 warm-start (optional) → Reptile meta-training → Final evaluation with cold-start curve.

Uses `sys.path.insert(0, ...)` to add project root, `argparse` for CLI, `set_seed()` for reproducibility.

---

## 12. Tests (139 total — 118 core + 21 API)

All test files start with `sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))`.

### `tests/conftest.py` — Shared fixtures

**Critical:** `create_model(config=None)` filters `ModelConfig.__dict__` to only include args that `MemoryNet.__init__` accepts (excludes `mc_samples`).

Fixtures: `device` (cpu), `config` (MetaSRSConfig), `model_config`, `training_config`, `fsrs_config`, `model` (via create_model), `fsrs` (FSRS6 instance), `sample_batch_tensors` (batch_size=4 dict), `sample_reviews` (4 Review objects), `synthetic_tasks` (20 students × 50 reviews).

### Test files and what they test:

1. **test_config.py** (11 tests) — defaults, feature_dim_sum, FSRS weight count, epsilon/LR schedule boundaries and monotonic decrease, phase thresholds ordered, k_steps increasing, scheduling defaults, config aggregation.

2. **test_memory_net.py** (12 tests) — creation, param count (30K-150K), input_dim=49, submodules exist, forward output shapes, output ranges (S in [0.001,36500], D in [1,10], p in [0,1]), no NaN, feature dim=49, no-history→zero GRU context, forward_from_features, predict_recall, predict_stability, gradient flow backward pass.

3. **test_models.py** (5 tests) — GRU encoder output shape, zero_state, no NaN, works without lengths, truncation for seq > max_len.

4. **test_loss.py** (9 tests) — returns dict with all keys, finite loss, perfect prediction → low loss, bad prediction → high loss, monotonicity penalty on S decrease, no penalty on S increase, NaN fallback, loss weight zeroing, compute_loss integration.

5. **test_fsrs_warmstart.py** (17 tests) — retrievability at t=0 (=1.0), decays with time, higher S → slower decay, R at t=S, zero S → 0.0, batch retrievability, initial S by grade (monotonic), initial D by grade (inverse), success increases S, Easy > Hard stability, lapse decreases S, Good maintains D, Again increases D, Easy decreases D, D bounds [1,10], step returns 3 values, first review R=1.0, success S > lapse S, simulate_student correctness.

6. **test_task_sampler.py** (10 tests) — Review creation/defaults, Task split, chronological ordering, get_review_history, unique_cards, batch shapes, batch on device, review_count accumulates (1,2,3), TaskSampler filters short histories, sample returns correct count, sampled tasks have splits, synthetic generation count/state/determinism.

7. **test_reptile.py** (7 tests) — sample_batch returns dict, respects size, inner_loop changes weights, inner_loop doesn't modify phi, reptile_update moves phi, update direction (phi + 0.5*delta), epsilon=0 no change, ReptileTrainer runs 3 iterations.

8. **test_adaptation.py** (8 tests) — initial phase is ZERO_SHOT, phase transitions (0-4→ZS, 5→RA, 50→FP), n_reviews property, get_model returns MemoryNet, returns eval mode, after adaptation uses theta, get_state has expected keys, load_state roundtrip.

9. **test_scheduling.py** (13 tests) — basic interval in [1,365], min/max interval, high uncertainty shortens (with fuzz tolerance), confidence/difficulty factor ranges, hard card shorter interval (seeded RNG), predict_with_uncertainty returns 4 tensors, sigma non-negative, schedule_card returns ScheduleResult, schedule_deck returns list, select_next_card strategies (most_due, highest_uncertainty, uncertainty_weighted), invalid strategy raises, empty results raises.

10. **test_evaluation.py** (6 tests) — EvalResults defaults, summary string with ✓, perfect AUC=1.0, random AUC≈0.5, all-same-label handled, manual AUC implementation, evaluate_on_tasks returns results, cold_start_curve returns dict.

11. **test_integration.py** (3 tests) — synthetic pipeline (generate→split→model→sampler→train 5 iters→evaluate), model forward with synthetic batch (no NaN, finite loss), checkpoint save/load roundtrip.

12. **test_api.py** (21 tests) — health returns ok, model_params in [25K,60K], submit first review, phase transition after 6 reviews, subsequent same card, missing fields → 400, invalid grade → 400, negative elapsed → 400, not-JSON → 400, schedule result has all expected keys, schedule deck, empty cards → 400, missing student → 400, new student status, status after reviews, reset clears state, reset nonexistent student, next card selection, custom strategy, empty cards → 400, 404 returns JSON.

---

## 13. Critical Implementation Details

### Import pattern
All modules use **bare imports** like `from config import ...` (not `from meta_srs.config`). The `api.py` and `conftest.py` add the meta-srs root to `sys.path` at the top:
```python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```

### ModelConfig.mc_samples
`mc_samples` is in `ModelConfig` but `MemoryNet.__init__` does NOT accept it. When constructing a model from config, filter it out:
```python
model_kwargs = {k: v for k, v in cfg.model.__dict__.items()
                if k in MemoryNet.__init__.__code__.co_varnames}
model = MemoryNet(**model_kwargs)
```

### Numerical stability
- Loss uses `BCEWithLogitsLoss` with manual `p → logit` conversion (not direct BCE)
- R_from_S computed in log-space: `log_R = log(0.9) * t^w20 / S_pred`, clamped [-20, 0]
- S_next clamped to [0.001, 36500] in MemoryNet
- Grad clipping at 1.0 in all optimisation loops
- NaN fallback in loss function

### Thread safety
The Flask API uses a `threading.Lock()` around the per-student adapter dict. Each student gets their own MemoryNet instance to prevent parameter clobbering.

### No offline preprocessing
The system is fully online-compatible. No BERT, no sentence-transformers, no file I/O at inference time. Only loads `phi_star.pt` at startup.

---

## 14. How to Run

### On Replit:
1. Install dependencies: `cd meta-srs && pip install -r requirements-web.txt`
2. The `.replit` file auto-runs `cd meta-srs && python api.py`
3. The API starts on port 5000 (reads from `PORT` env var)

### Run tests:
```bash
cd meta-srs && pip install pytest pytest-timeout flask && python -m pytest tests/ -v --tb=short
```
Expected: 139 tests pass in ~6 seconds.

### Train a model:
```bash
cd meta-srs && python train.py --synthetic --n-students 100 --n-iters 500
```

### Call the API from Node.js:
```javascript
// Submit a review
const res = await fetch('http://localhost:5000/api/review', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        student_id: 'student_001',
        card_id: 'card_042',
        grade: 3,
        elapsed_days: 2.5
    })
});
const data = await res.json();
// → { phase: "RAPID_ADAPT", n_reviews: 12, schedule: { interval_days: 7, p_recall_mean: 0.91, ... } }

// Schedule a deck
const deck = await fetch('http://localhost:5000/api/schedule', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        student_id: 'student_001',
        cards: [
            { card_id: 'c1', elapsed_days: 3.0 },
            { card_id: 'c2', elapsed_days: 0.0 }
        ]
    })
});

// Get next card to review
const next = await fetch('http://localhost:5000/api/next-card', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        student_id: 'student_001',
        cards: [
            { card_id: 'c1', elapsed_days: 3.0 },
            { card_id: 'c2', elapsed_days: 10.0 }
        ],
        strategy: 'uncertainty_weighted'
    })
});
```

---

## 15. Verification Checklist

After building, verify:

- [ ] `python -m pytest tests/ -v --tb=short` passes all 139 tests
- [ ] `python api.py` starts Flask server on port 5000
- [ ] `GET /api/health` returns `{"status": "ok", ...}`
- [ ] `POST /api/review` with valid JSON returns scheduling prediction
- [ ] `POST /api/schedule` batches multiple cards
- [ ] `POST /api/next-card` selects a card using uncertainty weighting
- [ ] `GET /api/student/x/status` returns phase and review count
- [ ] `POST /api/student/x/reset` clears state
- [ ] Invalid requests return 400 with error details
- [ ] Model has ~35K parameters (check via health endpoint)
- [ ] Feature vector dimension is 49
