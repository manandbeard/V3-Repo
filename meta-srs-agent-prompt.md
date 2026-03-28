# Agent Prompt: Build a Meta-Learning Spaced Repetition Engine from Scratch

> **Objective**: Build "MetaSRS" — a production-ready, meta-learning-based spaced repetition scheduling engine in Python/PyTorch. The system uses Reptile meta-learning to find neural network initialization parameters (φ\*) that can be fast-adapted to any new student in 5–20 gradient steps, achieving recall-prediction accuracy (AUC > 0.80) that surpasses FSRS-6 (~0.78) while solving the cold-start problem.

---

## System Overview

Build a spaced repetition system with five integrated subsystems:

1. **Neural Memory Model** (MemoryNet) — a small (~50K parameter) feedforward network that predicts memory state transitions (Stability, Difficulty, Recall probability)
2. **Reptile Meta-Trainer** — an outer/inner loop meta-learning system that trains MemoryNet across thousands of students to find optimal initialization parameters
3. **FSRS-6 Warm-Start** — a pre-training phase that initializes the neural network to reproduce the behavior of the FSRS-6 algorithm (current state-of-the-art hand-crafted SRS)
4. **Three-Phase Fast Adapter** — an inference-time component that progressively personalizes the model to individual students
5. **Uncertainty-Aware Scheduler** — a scheduling engine that uses Monte Carlo Dropout to quantify prediction confidence and compute optimal review intervals

---

## Part 1: Project Structure

Create the following directory layout:

```
meta-srs/
├── config.py                    # All hyperparameters (4 dataclass configs)
├── train.py                     # Main training orchestration script (CLI entry point)
├── requirements.txt             # Dependencies
├── pytest.ini                   # Test configuration
├── models/
│   ├── __init__.py
│   ├── memory_net.py            # MemoryNet — main neural DSR model
│   └── gru_encoder.py           # GRU-based review history encoder
├── training/
│   ├── __init__.py
│   ├── reptile.py               # Reptile outer/inner loop implementation
│   ├── loss.py                  # Multi-component loss function
│   └── fsrs_warmstart.py        # FSRS-6 baseline computations + warm-start
├── data/
│   ├── __init__.py
│   └── task_sampler.py          # Data structures, batch construction, synthetic data generation
├── inference/
│   ├── __init__.py
│   ├── adaptation.py            # FastAdapter — 3-phase student onboarding
│   └── scheduling.py            # Scheduler — uncertainty-aware interval computation
├── evaluation/
│   ├── __init__.py
│   └── metrics.py               # EvalResults, AUC computation, cold-start curves
└── tests/
    ├── __init__.py
    ├── conftest.py              # Shared fixtures
    ├── test_memory_net.py       # Architecture + forward pass tests
    ├── test_reptile.py          # Meta-training loop tests
    ├── test_loss.py             # Loss component tests
    ├── test_scheduling.py       # Interval computation tests
    ├── test_adaptation.py       # Phase transition tests
    ├── test_task_sampler.py     # Data pipeline tests
    ├── test_fsrs_warmstart.py   # FSRS-6 baseline tests
    └── test_evaluation.py       # Metrics computation tests
```

**Dependencies** (`requirements.txt`):
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

## Part 2: Configuration (`config.py`)

Create four `@dataclass` configuration classes, all composed into a single `MetaSRSConfig` root:

### ModelConfig — Neural Architecture
| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `input_dim` | int | 49 | Total feature vector dimension: 4 scalars + 4 grade one-hot + 1 count + 8 user stats + 32 GRU context = 49 (the model should dynamically pad if the assembled feature vector is slightly different) |
| `hidden_dim` | int | 128 | Width of hidden layers h1 and h2 |
| `gru_hidden_dim` | int | 32 | GRU hidden state dimension |
| `history_len` | int | 32 | Max number of past reviews per card for GRU |
| `user_stats_dim` | int | 8 | User-level aggregate statistics dimension |
| `dropout` | float | 0.1 | Dropout rate (enables MC Dropout for uncertainty) |
| `mc_samples` | int | 20 | Number of Monte Carlo forward passes during inference |

### FSRSConfig — Population Baseline (21 FSRS-6 weights)
Store the 21 default FSRS-6 population parameters `w[0]..w[20]` as a list field:
- `w0–w3`: Initial stability per grade (Again=0.40, Hard=1.18, Good=3.17, Easy=15.69)
- `w4`: Difficulty baseline (7.19)
- `w5`: Difficulty mean-reversion weight (0.53)
- `w6`: Difficulty update scaling (1.46)
- `w7`: Unused placeholder (0.0046)
- `w8–w10`: Stability growth factors on success (1.55, 0.174, 1.02)
- `w11–w14`: Lapse stability factors (1.94, 0.11, 0.296, 2.27)
- `w15–w16`: Grade modifiers Hard/Easy (0.23, 2.99)
- `w17–w19`: Short-term scheduling (0.517, 0.662, 0.060)
- `w20`: **Power-law exponent (0.4665)** — critical for the forgetting curve

### TrainingConfig — Reptile Meta-Learning
| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `n_iters` | int | 50000 | Total meta-iterations |
| `meta_batch_size` | int | 16 | Tasks (students) per outer update |
| `outer_lr_start` | float | 1e-3 | Adam LR for outer optimizer (cosine decayed) |
| `outer_lr_end` | float | 1e-4 | Final LR after cosine decay |
| `epsilon_start` | float | 0.10 | Reptile step size start (linear decay) |
| `epsilon_end` | float | 0.01 | Reptile step size end |
| `inner_lr` | float | 0.01 | Inner-loop (per-student) Adam LR |
| `inner_steps_phase1` | int | 5 | Gradient steps for rapid adaptation |
| `inner_steps_phase2` | int | 20 | Gradient steps for full personalization |
| `task_batch_size` | int | 32 | Reviews per mini-batch inside inner loop |
| `support_ratio` | float | 0.70 | Fraction of reviews for training (rest for evaluation) |
| `stability_loss_weight` | float | 0.10 | Weight for auxiliary stability-curve consistency loss |
| `monotonicity_loss_weight` | float | 0.01 | Weight for monotonicity constraint loss |
| `warmstart_epochs` | int | 10 | FSRS-6 pre-training epochs |
| `warmstart_lr` | float | 1e-3 | Warm-start learning rate |

Implement two schedule functions as methods:
- `epsilon_schedule(iteration)`: Linear decay from `epsilon_start` to `epsilon_end` over `n_iters`
- `outer_lr_schedule(iteration)`: Cosine decay from `outer_lr_start` to `outer_lr_end`

### AdaptationConfig — Three-Phase Onboarding
| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `phase1_threshold` | int | 5 | Reviews before entering Phase 2 |
| `phase2_threshold` | int | 50 | Reviews before entering Phase 3 |
| `phase1_k_steps` | int | 5 | Phase 1 gradient steps |
| `phase2_k_steps` | int | 10 | Phase 2 gradient steps |
| `phase3_k_steps` | int | 20 | Phase 3 gradient steps |
| `phase3_inner_lr` | float | 0.02 | Higher LR for full personalization |
| `adapt_every_n_reviews` | int | 5 | How often to re-adapt |
| `streaming_after` | int | 50 | When to start per-review gradient updates |

### SchedulingConfig — Interval Computation
| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `desired_retention` | float | 0.90 | Target recall probability |
| `min_interval` | int | 1 | Minimum days between reviews |
| `max_interval` | int | 365 | Maximum days between reviews |
| `fuzz_range` | tuple | (0.95, 1.05) | ±5% random fuzz to prevent clustering |
| `difficulty_centre` | float | 5.0 | Reference difficulty level |
| `difficulty_slope` | float | 0.05 | Interval penalty per unit of difficulty above center |
| `uncertainty_discount` | float | 0.5 | Max interval reduction from high uncertainty |

### Root Config
```python
@dataclass
class MetaSRSConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    fsrs: FSRSConfig = field(default_factory=FSRSConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    scheduling: SchedulingConfig = field(default_factory=SchedulingConfig)
```

---

## Part 3: Neural Architecture (`models/`)

### 3.1 MemoryNet (`models/memory_net.py`)

Build a feedforward neural network that takes a 49-dimensional feature vector (the model should handle slight dimension mismatches via dynamic padding) and outputs three memory-state predictions.

**Architecture**:
```
Input: (batch, input_dim)   # ~49 features

h1 = Linear(input_dim, 128) → LayerNorm(128) → GELU → Dropout(0.1)
h2 = Linear(128, 128)       → LayerNorm(128) → GELU → Dropout(0.1)
h3 = Linear(128, 64)        → GELU → Dropout(0.1)

Output heads (all from h3):
  stability_head:  Linear(64, 1) → Softplus → multiply by S_prev → clamp [0.001, 36500]
  difficulty_head: Linear(64, 1) → Sigmoid → multiply by 9, add 1 → range [1, 10]
  recall_head:     Linear(64, 1) → Sigmoid → range [0, 1]
```

**Design rationale**:
- Use **LayerNorm** (not BatchNorm) because inner-loop SGD may use batch size 1
- Use **GELU** activation for smoother gradients
- **~50K parameters** total — small enough that 5–20 gradient steps meaningfully move the weights
- Stability head uses **multiplicative update**: `S_next = Softplus(head_output) × S_prev` — this encodes the cognitive science principle that stability evolves relative to its current value
- Stability is clamped to `[0.001, 36500]` to prevent numerical instability during Reptile inner loops

**Feature Assembly** — Implement a `build_features()` method that constructs the input vector:

| Component | Dimensions | Normalization |
|-----------|-----------|---------------|
| D_prev | 1 | Divide by 10.0 → [0, 1] |
| S_prev | 1 | log(S_prev) for numerical stability |
| R_at_review | 1 | Already in [0, 1] |
| delta_t (elapsed days) | 1 | log(delta_t + 1) to handle t=0 |
| Grade | 4 | One-hot encoding: [Again=1, Hard=2, Good=3, Easy=4] |
| Review count | 1 | log(count) |
| User stats | 8 | Mean difficulty, mean stability, session stats, etc. |
| GRU context | 32 | Final hidden state of GRU history encoder |
| **Total** | **~49** | |

**Return type**: Create a `MemoryState` dataclass or named tuple with fields: `S_next`, `D_next`, `p_recall`.

**Key methods**:
- `forward(batch_dict)` — full pipeline: build features → network → outputs
- `forward_from_features(features, S_prev)` — fast path when features are pre-assembled
- `predict_recall(batch_dict)` — convenience: return only p_recall
- `predict_stability(batch_dict)` — convenience: return only S_next
- `build_features(batch_dict)` — assemble the 49-dim input vector
- `count_parameters()` — return total trainable parameter count

### 3.2 GRU History Encoder (`models/gru_encoder.py`)

Build a 1-layer GRU that encodes the per-card review history into a 32-dimensional context vector.

**Input**: Batch of review sequences, each sequence is `(seq_len, 2)` where the 2 features are:
- Normalized grade: `grade / 4.0` → [0, 1]
- Log-scaled elapsed time: `log(delta_t + 1)`

**Architecture**:
```python
GRU(input_size=2, hidden_size=32, num_layers=1, batch_first=True)
```

**Processing**:
1. Accept variable-length sequences (up to `history_len=32` reviews)
2. Use `pack_padded_sequence` to handle variable lengths efficiently
3. Extract and return the final hidden state `h_n` → shape `(batch, 32)`
4. For cards with no history (first review), return a zero vector

**Purpose**: Captures patterns like repeated "Again" presses (instability), regular review intervals (stability), or erratic spacing (uncertainty).

---

## Part 4: Training Subsystem (`training/`)

### 4.1 Multi-Component Loss (`training/loss.py`)

Implement `MetaSRSLoss` with three loss terms:

**PRIMARY — Recall Prediction (Binary Cross-Entropy)**:
```
L_recall = BCEWithLogitsLoss(p_pred_logit, recalled)
```
Convert the model's sigmoid output p_recall back to logits for numerical stability with BCEWithLogitsLoss.

**AUXILIARY 1 — Stability-Curve Consistency**:
```
R_from_S = exp(log(0.9) × (delta_t ^ w20) / S_next)
L_stability = MSE(R_from_S, recalled)
```
Where `w20 = 0.4665` (the FSRS-6 power-law exponent). Compute `R_from_S` in **log-space** for numerical stability. This loss enforces that the predicted stability S is consistent with the power-law forgetting curve from cognitive science.

**AUXILIARY 2 — Monotonicity Constraint**:
```
L_mono = mean(ReLU(-(S_next - S_prev)) × success_mask)
```
Where `success_mask = (grade >= 2)`. On successful recall, stability should NOT decrease. Penalize violations.

**Combined**:
```
L_total = L_recall + 0.10 × L_stability + 0.01 × L_monotonicity
```

**NaN guard**: If `L_total` is NaN (can happen during early Reptile training), fall back to `L_recall` only.

Return a dictionary: `{"total": L_total, "recall": L_recall, "stability": L_stability, "monotonicity": L_mono}`

### 4.2 Reptile Meta-Learner (`training/reptile.py`)

Implement `ReptileTrainer` with the following algorithm:

**Outer Loop** (50,000 meta-iterations):
```
φ ← initialize (from warm-start or random)

for iteration in range(n_iters):
    tasks ← TaskSampler.sample(meta_batch_size=16)  # sample 16 students
    
    accumulated_direction = zeros_like(φ)
    
    for task in tasks:
        # Inner loop: adapt φ to this student
        W_i ← inner_loop(φ, task, k_steps=5, inner_lr=0.01)
        accumulated_direction += (W_i - φ)
    
    # Reptile update: move φ toward average adapted weights
    ε = epsilon_schedule(iteration)    # linear decay 0.10 → 0.01
    φ ← φ + ε × accumulated_direction / len(tasks)
    
    # Periodic evaluation and checkpointing
    if iteration % 1000 == 0: evaluate(φ, test_tasks)
    if iteration % 5000 == 0: save_checkpoint(φ, iteration)
```

**Inner Loop** (per-student adaptation):
```python
def inner_loop(phi, task, k_steps=5, inner_lr=0.01):
    θ = deepcopy(phi)  # Don't modify original
    optimizer = Adam(θ.parameters(), lr=inner_lr)
    
    for step in range(k_steps):
        batch = sample(task.support_set, size=32)
        features = model.build_features(batch)
        state = model.forward_from_features(features, batch["S_prev"])
        loss = MetaSRSLoss(state, batch)
        
        optimizer.zero_grad()
        loss["total"].backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    return θ.state_dict()  # Adapted parameters W_i
```

**Key implementation details**:
- Use gradient clipping (norm ≤ 1.0) in the inner loop for training stability
- The Reptile update treats `(W_i - φ)` as a pseudo-gradient — apply it directly to φ (not through an optimizer)
- Support both the raw Reptile update and an Adam-based outer optimizer (the Reptile paper shows both work)
- Query set (30% of reviews) is used for monitoring inner-loop performance, not for the Reptile update itself

### 4.3 FSRS-6 Warm-Start (`training/fsrs_warmstart.py`)

Implement the full FSRS-6 algorithm as a reference baseline and warm-start source.

**FSRS-6 Forgetting Curve**:
```
R(t, S) = (0.9^(1/S))^(t^w20)
```
This is a **power-law** form (not pure exponential). At t=0: R=1. At t≈S: R≈0.9 (by design). The power-law exponent w20=0.4665 makes the curve steeper initially and flatter at longer intervals.

**Stability Update on Success** (grade ≥ 2):
```
S' = S × (exp(w8) × (11 - D) × S^(-w9) × (exp(w10×(1-R)) - 1) × grade_modifier + 1)
```
Grade modifiers: Hard(2) → w15=0.2315, Good(3) → 1.0, Easy(4) → w16=2.9898

**Stability Update on Lapse** (grade = 1, "Again"):
```
S_f = w11 × D^(-w12) × ((S+1)^w13 - 1) × exp(w14×(1-R))
```

**Difficulty Update** (all grades):
```
ΔD = -w6 × (grade - 3)
D'' = D + ΔD × (10 - D) / 9        # Linear damping toward D=10
D_easy = w4 - 2 × w5                # Default "Easy" difficulty
D' = w5 × D_easy + (1 - w5) × D''  # Mean-reversion
D' = clamp(D', 1, 10)
```

**Initial values** (first review of a card):
```
S_initial = w[grade - 1]   # w0=0.40 (Again), w1=1.18 (Hard), w2=3.17 (Good), w3=15.69 (Easy)
D_initial = w4 - exp(w5 × (grade - 1)) + 1   # Initial difficulty varies by first grade
```

**Warm-start function**: Pre-train MemoryNet for 10 epochs to minimize:
```
L_warmstart = BCE(p_recall_pred, R_target)
            + 0.5 × MSE(log(S_next_pred + 1), log(S_target + 1))
            + 0.5 × MSE(D_next_pred, D_target)
```
Where targets come from FSRS-6 computations. This initializes the neural network with cognitively-valid behavior before Reptile training begins, cutting meta-training time by ~60%.

---

## Part 5: Data Pipeline (`data/task_sampler.py`)

### Data Structures

**Review** (single review event):
```python
@dataclass
class Review:
    card_id: str
    timestamp: int              # Unix timestamp
    elapsed_days: float         # Days since last review (0.0 for first)
    grade: int                  # 1=Again, 2=Hard, 3=Good, 4=Easy
    recalled: bool              # grade >= 2
    S_prev: float = 1.0        # Current stability
    D_prev: float = 5.0        # Current difficulty
    R_at_review: float = 1.0   # Retrievability at review time
    S_target: float = 1.0      # FSRS-6 target S (for warm-start)
    D_target: float = 5.0      # FSRS-6 target D
```

**Task** (one student's data — the "task" in meta-learning):
```python
@dataclass
class Task:
    student_id: str
    reviews: List[Review]
    support_set: List[Review]               # First 70% (for inner-loop training)
    query_set: List[Review]                 # Last 30% (for evaluation)
    
    def split(support_ratio=0.70):
        """Chronological split into support/query sets."""
```

### Batch Construction

Implement `reviews_to_batch(reviews, device, history_len=32)` that converts a list of Reviews into a dictionary of tensors:

- `D_prev`: (batch,) float
- `S_prev`: (batch,) float
- `R_at_review`: (batch,) float
- `delta_t`: (batch,) float — elapsed days
- `grade`: (batch,) long
- `review_count`: (batch,) float — accumulated count per card within this batch
- `recalled`: (batch,) float — target labels
- `user_stats`: (batch, 8) float — aggregate stats (mean D, mean S, etc.)
- `history_grades`: (batch, history_len) float — past grades for GRU
- `history_delta_ts`: (batch, history_len) float — past elapsed times for GRU
- `history_lengths`: (batch,) long — actual sequence lengths
- `S_target`: (batch,) float — FSRS-6 target stability
- `D_target`: (batch,) float — FSRS-6 target difficulty

**Review history tracking**: As you iterate through reviews in chronological order, maintain a per-card history buffer. For each review, the history contains all *previous* reviews of that card (not including the current one). Then append the current review to the buffer for future iterations.

### Task Sampler

```python
class TaskSampler:
    def __init__(self, tasks, support_ratio=0.70, min_reviews=10):
        # Filter out students with fewer than min_reviews
        # Split each task into support/query sets
    
    def sample(self, batch_size=16):
        """Sample batch_size tasks with replacement for one meta-iteration."""
```

### Synthetic Data Generation

Implement `ReviewDataset.generate_synthetic()` that uses FSRS-6 to simulate realistic review data:

1. Create `n_students` (default 500), each with `reviews_per_student` (default 100) reviews
2. Create `n_cards` (default 200)
3. For each student, simulate reviews:
   - Randomly select a card
   - If first review: assign initial S and D from FSRS-6, grade drawn from weights [15%, 20%, 50%, 15%]
   - If subsequent review: compute R from FSRS-6 forgetting curve, simulate recall with Bernoulli(R), assign grade accordingly
   - Use FSRS-6 to compute S_target and D_target for the next state
4. Return list of Task objects with deterministic seeding for reproducibility

---

## Part 6: Inference Components (`inference/`)

### 6.1 Scheduler (`inference/scheduling.py`)

**Monte Carlo Dropout Uncertainty**:
```python
def predict_with_uncertainty(model, features, S_prev, mc_samples=20):
    model.train()  # Keep dropout active!
    
    predictions = []
    for _ in range(mc_samples):
        with torch.no_grad():
            state = model(features, S_prev)
            predictions.append(state)
    
    p_mean = mean of all p_recall predictions
    p_sigma = std of all p_recall predictions  # Epistemic uncertainty
    S_mean = mean of S_next predictions
    D_mean = mean of D_next predictions
    
    return p_mean, p_sigma, S_mean, D_mean
```

**Interval Computation**:
```python
def compute_interval(S_pred, D, sigma, config):
    base_interval = S_pred
    
    # Uncertainty discount: high σ → shorter interval (conservative)
    confidence_factor = 1.0 - uncertainty_discount × min(sigma, 1.0)
    confidence_factor = clamp(confidence_factor, 0.5, 1.0)
    
    # Difficulty discount: D > centre → shorter interval
    difficulty_factor = 1.0 - difficulty_slope × (D - difficulty_centre)
    difficulty_factor = clamp(difficulty_factor, 0.5, 1.5)
    
    # Fuzz: ±5% to prevent review clustering
    fuzz = uniform(fuzz_range[0], fuzz_range[1])
    
    interval = base_interval × confidence_factor × difficulty_factor × fuzz
    return clamp(round(interval), min_interval, max_interval)
```

**Card Selection Strategies** — implement three strategies:
1. `most_due`: Review the card with the shortest predicted interval
2. `highest_uncertainty`: Review the card with highest σ (explore)
3. `uncertainty_weighted`: Score = (1 - p_recall) + 0.5 × sigma, then sample from softmax(scores × 3.0) — balanced exploration

### 6.2 FastAdapter (`inference/adaptation.py`)

Implement `FastAdapter` — manages per-student model personalization across three phases:

**Phase 1: Zero-Shot** (0–4 reviews)
- Use φ\* directly, no adaptation
- Schedule using population model

**Phase 2: Rapid Adapt** (5–49 reviews)
- Every 5 reviews: run k=5 gradient steps starting from φ\* (always from meta-init)
- Use Adam with lr=0.01
- Inner loop trains on ALL reviews collected so far

**Phase 3: Full Personal** (50+ reviews)
- Every 5 reviews: run k=20 gradient steps starting from θ_student (from student's current params)
- Use Adam with higher lr=0.02
- **Plus streaming**: After every individual review (once past 50), do a single gradient step on the last 8 reviews

**State management**:
- Store `phi_star` (read-only meta-initialization)
- Store `theta_student` (student's personalized parameters, evolves over time)
- Store `reviews` (full history for this student)
- Implement `get_model()` that loads appropriate parameters based on current phase

---

## Part 7: Evaluation (`evaluation/metrics.py`)

### EvalResults Dataclass
```python
@dataclass
class EvalResults:
    auc_roc: float         # Main metric: recall prediction accuracy (target > 0.80)
    calibration_error: float  # |predicted R - actual R| (target < 0.05)
    cold_start_auc: float    # AUC with 0 adaptation reviews (target > 0.73)
    adaptation_speed: int    # Reviews to match FSRS-6 AUC (target < 30)
    retention_30d: float     # Recall rate at 30 days (target > 85%)
    rmse_stability: float    # RMSE(S_pred - S_target) (target < 2.0)
    phase1_auc: float        # Per-phase breakdowns
    phase2_auc: float
    phase3_auc: float
```

### MetaSRSEvaluator
```python
class MetaSRSEvaluator:
    def evaluate_on_tasks(phi, tasks, k_steps=5):
        """For each task: adapt φ → θ on support set, evaluate on query set."""
        # Compute AUC, calibration error, RMSE for both cold-start and adapted
    
    def cold_start_curve(phi, tasks, review_counts=[0, 5, 10, 20, 30]):
        """Measure AUC at different adaptation points."""
        # Returns: {n_reviews → AUC} dictionary
```

Use `sklearn.metrics.roc_auc_score` for AUC computation. Handle edge cases where all labels are the same class.

---

## Part 8: Training Script (`train.py`)

Build the main CLI entry point that orchestrates the full pipeline:

**CLI Arguments**:
| Argument | Type | Default | Purpose |
|----------|------|---------|---------|
| `--data` | str | None | Path to review CSV |
| `--synthetic` | flag | False | Use synthetic data |
| `--n-students` | int | 500 | Number of synthetic students |
| `--n-iters` | int | 50000 | Meta-iterations |
| `--inner-steps` | int | None | Override inner loop steps |
| `--batch-size` | int | None | Override meta batch size |
| `--resume` | str | None | Resume from checkpoint path |
| `--skip-warmstart` | flag | False | Skip FSRS-6 pre-training |
| `--device` | str | "auto" | Device (cpu/cuda/auto) |
| `--seed` | int | 42 | Random seed |
| `--eval-only` | flag | False | Run evaluation without training |
| `--eval-checkpoint` | str | None | Checkpoint for eval-only mode |

**Execution Flow**:
1. Parse arguments and set random seed
2. Load data (CSV or synthetic generation)
3. Split tasks 80/20 by student for train/test
4. Create MemoryNet model
5. \[Optional\] Warm-start from FSRS-6 (10 epochs pre-training)
6. Run Reptile meta-training (50K iterations)
7. Final evaluation on test tasks
8. Save φ\* checkpoint and print results summary

---

## Part 9: Test Suite (`tests/`)

Write comprehensive tests using pytest. Target 100+ tests across all modules.

### Test Fixtures (`conftest.py`)
- `device` fixture (cpu or cuda if available)
- `model_config` fixture returning ModelConfig()
- `model` fixture returning MemoryNet constructed from config (note: filter out `mc_samples` field from ModelConfig when constructing MemoryNet, since MemoryNet.__init__ doesn't accept it)
- `sample_batch_tensors` fixture returning a dict of tensors with proper shapes
- `synthetic_tasks` fixture returning a small set of synthetic tasks

### Key Test Patterns

**Architecture tests** (`test_memory_net.py`):
- Parameter count is approximately 50K
- Output ranges: S_next ∈ [0.001, 36500], D_next ∈ [1, 10], p_recall ∈ [0, 1]
- Forward pass produces correct shapes
- Backward pass produces non-zero gradients

**Loss tests** (`test_loss.py`):
- Each loss component is non-negative
- Total loss is finite (not NaN/Inf)
- NaN guard falls back to recall-only loss
- Monotonicity loss is zero when S_next >= S_prev on success

**Meta-learning tests** (`test_reptile.py`):
- Inner loop produces different parameters than input φ
- Reptile update moves φ toward adapted weights
- Trainer runs for 3 iterations without error

**Data tests** (`test_task_sampler.py`):
- Synthetic generation is deterministic with same seed
- Reviews have valid ranges (grade 1–4, elapsed >= 0, etc.)
- Task split produces correct support/query ratio

**Scheduling tests** (`test_scheduling.py`):
- Higher uncertainty → shorter intervals
- Higher difficulty → shorter intervals
- Intervals are within [min_interval, max_interval]

**Adaptation tests** (`test_adaptation.py`):
- Phase transitions occur at correct review counts
- Phase 1 uses φ\* directly
- Phase 2 adapts from φ\*
- Phase 3 uses streaming updates

**FSRS-6 tests** (`test_fsrs_warmstart.py`):
- Retrievability at t=0 is 1.0
- Retrievability decreases with time
- Stability increases on successful recall
- Stability drops sharply on lapse

**Evaluation tests** (`test_evaluation.py`):
- AUC is computed correctly (perfect predictions → AUC = 1.0)
- Cold-start curve returns values for each n

---

## Part 10: Mathematical Foundations Reference

### The Forgetting Curve (Power-Law)
```
R(t, S) = (0.9^(1/S))^(t^w20)

where:
  R = retrievability (recall probability) at time t
  S = stability (the interval in days at which R drops to 90%)
  t = elapsed time in days
  w20 = 0.4665 (power-law exponent)

Properties:
  - At t=0: R = 1 (perfect recall immediately after review)
  - At t ≈ S: R ≈ 0.9 (by design)
  - Power-law tail: slower decay than exponential for large t
```

**Why power-law, not exponential?** The population average of exponential curves with different stability values follows a power-law distribution. This has been empirically validated across millions of Anki reviews.

### Reptile Meta-Learning
```
Update rule: φ ← φ + ε × mean_i(W_i - φ)

Equivalently: φ moves toward the average of adapted weights.

Theoretical justification (Nichol & Schulman 2018):
  Reptile maximizes < ∇L(θ, D_test), ∇L(θ, D_train) >
  i.e., finds initialization that generalizes within each task.

First-order approximation to MAML — no Hessian computation needed.
```

### Multi-Component Loss Rationale
- **Recall loss (BCE)**: Primary objective — accurately predict whether a student will remember a card
- **Stability consistency**: Ensures the model's S predictions are consistent with the scientifically-established forgetting curve, providing regularization from cognitive science
- **Monotonicity**: Enforces the constraint that successful recall should not decrease memory stability — a well-established finding in memory research

---

## Part 11: Performance Targets and Validation

Run the test suite with:
```bash
cd meta-srs && python -m pytest tests/ -v --tb=short
```
Target: 100+ tests, all passing, ~6 seconds on CPU.

Validate the training pipeline with:
```bash
cd meta-srs && python train.py --synthetic --n-students 100 --n-iters 500
```
This should complete in ~4 minutes on CPU and demonstrate the full training → evaluation loop.

### Performance Benchmarks
| Metric | Target | FSRS-6 Baseline |
|--------|--------|-----------------|
| AUC-ROC | > 0.80 | ~0.78 |
| Calibration Error | < 0.05 | ~0.06 |
| Cold-Start AUC (0 reviews) | > 0.73 | 0.78* |
| Adaptation Speed | < 30 reviews | N/A |
| 30-day Retention | > 85% | ~87% |
| RMSE(Stability) | < 2.0 days | ~2.5 |

*\*FSRS-6 uses population parameters directly, so its "cold-start" is already personalized at the population level.*

---

## Key Implementation Notes

1. **Numerical stability is critical**: Use log-space computations for the forgetting curve, clamp stability to prevent explosion/vanishing, and add NaN guards throughout the loss function.

2. **The model must be small**: ~50K parameters is intentional — larger models overfit during the few-shot inner loop and defeat the purpose of meta-learning.

3. **LayerNorm over BatchNorm**: The inner loop may run with batch size as small as 1 during streaming updates. BatchNorm fails with single samples; LayerNorm works fine.

4. **Gradient clipping in inner loop**: Always clip gradients to norm ≤ 1.0 during inner-loop adaptation. Without this, rare outlier reviews can cause catastrophic parameter updates.

5. **Chronological splits only**: Never randomly shuffle reviews within a student's history. The support/query split must be chronological (first 70% / last 30%) to simulate real-world deployment where we train on past data and predict the future.

6. **MC Dropout requires `model.train()`**: During inference uncertainty estimation, call `model.train()` (not `model.eval()`) so that dropout remains active across the Monte Carlo forward passes.

7. **The FSRS-6 warm-start is optional but strongly recommended**: It cuts Reptile convergence time by ~60% and ensures the model starts from cognitively-valid predictions rather than random noise.

---

## Summary of Key Papers

| Paper | Year | Relevance |
|-------|------|-----------|
| **Reptile** (Nichol & Schulman, OpenAI) | 2018 | Core meta-learning algorithm |
| **MAML** (Finn et al., ICML) | 2017 | Theoretical foundation for Reptile |
| **FSRS-6** (open-spaced-repetition) | 2024–2025 | State-of-the-art SRS; DSR model with 21 params |
| **KAR³L** (Shu et al., EMNLP) | 2024 | Content-aware SRS with BERT embeddings |
| **Half-Life Regression** (Settles & Meeder, ACL) | 2016 | ML-based forgetting curves (Duolingo) |
| **DKT** (Piech et al., NeurIPS) | 2015 | Deep knowledge tracing with LSTM |
| **PLATIPUS** (Finn et al., NeurIPS) | 2018 | Bayesian extension of MAML/Reptile |
