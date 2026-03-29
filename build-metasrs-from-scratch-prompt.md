# Agent Prompt: Build MetaSRS — A Meta-Learning Spaced Repetition Engine — From Scratch

> **Objective**: Build "MetaSRS" — a production-ready, meta-learning-based spaced repetition scheduling engine in Python/PyTorch. The system uses Reptile meta-learning to find neural network initialization parameters (phi*) that fast-adapt to any new student in 5-20 gradient steps, achieving recall-prediction AUC > 0.80 that surpasses FSRS-6 (~0.78) while solving the cold-start problem.

---

## 1. Theoretical Foundation

### 1.1 The DSR Memory Model (Difficulty-Stability-Retrievability)

Every modern spaced repetition system is built on three latent memory variables per card per student:

| Variable | Symbol | Range | Meaning |
|----------|--------|-------|---------|
| **Difficulty** | D | [1, 10] | Intrinsic hardness of the card |
| **Stability** | S | (0, +inf) | Days until retention drops to 90% |
| **Retrievability** | R | [0, 1] | Current recall probability |

### 1.2 The FSRS-6 Forgetting Curve (Power-Law)

The state-of-the-art forgetting curve used by FSRS-6 (default in Anki since 2024) uses a power-law form:

```
R(t, S) = (0.9^(1/S))^(t^w20)
```

Where:
- `t` = elapsed days since last review
- `S` = stability
- `w20 = 0.4665` = power-law exponent (one of FSRS-6's 21 parameters)

**Properties:**
- At t=0: R = 1 (perfect recall immediately after review)
- At t ~ S: R ~ 0.9 (by definition of S)
- Power-law tail: slower decay than exponential for large t

**Why power-law, not exponential?** The population average of exponential forgetting curves with different stability values follows a power-law distribution. This has been empirically validated across millions of real Anki reviews.

### 1.3 FSRS-6 Stability Updates

**On Success (grade >= 2: Hard, Good, Easy):**
```
S' = S * (exp(w8) * (11 - D) * S^(-w9) * (exp(w10*(1-R)) - 1) * grade_modifier + 1)
```
Grade modifiers: Hard(2) -> w15=0.2315, Good(3) -> 1.0, Easy(4) -> w16=2.9898

**On Lapse (grade = 1, "Again"):**
```
S_f = w11 * D^(-w12) * ((S+1)^w13 - 1) * exp(w14*(1-R))
```

**Difficulty Update (all grades):**
```
delta_D = -w6 * (grade - 3)
D'' = D + delta_D * (10 - D) / 9        # Linear damping toward D=10
D_easy = initial_difficulty(grade=4)     # w4 - (4-3)*w5
D' = w5 * D_easy + (1 - w5) * D''       # Mean-reversion
D' = clamp(D', 1, 10)
```

**Initial values (first review of a card):**
```
S_initial = w[grade - 1]     # w0=0.40(Again), w1=1.18(Hard), w2=3.17(Good), w3=15.69(Easy)
D_initial = w4 - (grade - 3) * w5
```

### 1.4 Reptile Meta-Learning

Reptile (Nichol & Schulman, OpenAI 2018) is a first-order meta-learning algorithm. Unlike MAML (which requires second-order gradients/Hessians), Reptile only needs standard SGD.

**Core Algorithm:**
```
Initialize meta-parameters phi
for each meta-iteration:
    Sample task tau_i (= one student's review history)
    Perform k gradient steps on tau_i, starting from phi -> get W_i
    Update: phi <- phi + epsilon * (W_i - phi)
```

**Theoretical insight**: Reptile maximizes the inner product between gradients from different minibatches of the same task, promoting *within-task generalization*. This is equivalent to a first-order approximation of MAML.

**Why Reptile for SRS specifically:**
- Review histories vary wildly in length (1 to 10,000+) — Reptile handles variable-length inner loops
- No second-order differentiation needed — avoids memory explosion
- Student tasks are naturally i.i.d. across users (same card domain, different learners)
- Can plug any optimizer (Adam) into both inner and outer loops

### 1.5 The Cold-Start Problem

FSRS-6 requires hundreds to thousands of reviews per user before its 21 parameters converge. New users start from generic population averages. Meta-learning solves this: a student reviewing their first 5 cards already benefits from the forgetting patterns of thousands of prior learners.

---

## 2. System Architecture Overview

```
OFFLINE META-TRAINING:
  User Corpus -> [Task Sampler] -> Reptile Outer Loop -> phi*
  (N users, each with review history)

ONLINE FAST ADAPTATION (new student):
  0 reviews  -> use phi* directly (population prior, zero-shot)
  5 reviews  -> 5 gradient steps from phi* -> personalised theta_student
  50 reviews -> 20 gradient steps -> fully personalised theta_student

SCHEDULING:
  theta_student -> predict (R, S, D, uncertainty) -> compute interval -> show card
```

Five integrated subsystems:
1. **Neural Memory Model** (MemoryNet) — ~50K param feedforward net predicting DSR transitions
2. **Reptile Meta-Trainer** — outer/inner loop meta-learning finding optimal phi*
3. **FSRS-6 Warm-Start** — pre-training to reproduce FSRS-6 behavior (cuts training time ~60%)
4. **Three-Phase Fast Adapter** — inference-time progressive personalization
5. **Uncertainty-Aware Scheduler** — Monte Carlo Dropout for confidence + optimal intervals

---

## 3. Project Structure

Create this exact directory layout:

```
meta-srs/
├── config.py                    # All hyperparameters (5 dataclass configs)
├── train.py                     # Main CLI training orchestration script
├── requirements.txt             # Dependencies
├── pytest.ini                   # Test configuration
├── models/
│   ├── __init__.py              # Export MemoryNet, GRUHistoryEncoder
│   ├── memory_net.py            # MemoryNet — neural DSR model (~50K params)
│   └── gru_encoder.py           # GRU-based per-card review history encoder
├── training/
│   ├── __init__.py              # Export loss, reptile, warmstart components
│   ├── reptile.py               # Reptile outer/inner loop implementation
│   ├── loss.py                  # Multi-component loss (BCE + stability + monotonicity)
│   └── fsrs_warmstart.py        # FSRS-6 full algorithm + warm-start pre-training
├── data/
│   ├── __init__.py              # Export Task, TaskSampler, ReviewDataset
│   └── task_sampler.py          # Review/Task dataclasses, batch construction, synthetic data
├── inference/
│   ├── __init__.py              # Export FastAdapter, Scheduler, etc.
│   ├── adaptation.py            # FastAdapter — 3-phase student onboarding
│   └── scheduling.py            # Scheduler — MC Dropout uncertainty + interval computation
├── evaluation/
│   ├── __init__.py              # Export MetaSRSEvaluator, EvalResults
│   └── metrics.py               # AUC, calibration, cold-start curves, ablation
└── tests/
    ├── __init__.py
    ├── conftest.py              # Shared fixtures (model, config, sample data, etc.)
    ├── test_memory_net.py       # Architecture + forward/backward pass tests
    ├── test_models.py           # Additional model tests
    ├── test_reptile.py          # Meta-training loop tests
    ├── test_loss.py             # Loss component tests
    ├── test_scheduling.py       # Interval computation tests
    ├── test_adaptation.py       # Phase transition tests
    ├── test_task_sampler.py     # Data pipeline tests
    ├── test_fsrs_warmstart.py   # FSRS-6 baseline tests
    ├── test_evaluation.py       # Metrics computation tests
    ├── test_config.py           # Configuration tests
    └── test_integration.py      # End-to-end integration tests
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

**Test config** (`pytest.ini`):
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
addopts = -v --tb=short --timeout=120
```

---

## 4. Configuration System (`config.py`)

Create five `@dataclass` configuration classes composed into a single `MetaSRSConfig` root.

### 4.1 ModelConfig — Neural Architecture

```python
@dataclass
class ModelConfig:
    input_dim: int = 49           # Feature vector dimension: 4 scalars + 4 grade one-hot + 1 count + 8 user stats + 32 GRU context = 49
    hidden_dim: int = 128         # Width of hidden layers h1 and h2
    gru_hidden_dim: int = 32      # GRU hidden state dimension
    history_len: int = 32         # Max review history length for GRU encoder
    user_stats_dim: int = 8       # User-level aggregate statistics dimension
    dropout: float = 0.1          # Dropout rate (also enables MC Dropout at inference)
    mc_samples: int = 20          # Monte Carlo forward passes for uncertainty estimation
```

**Important**: `mc_samples` is used by the Scheduler but is NOT a parameter of MemoryNet's `__init__`. When constructing MemoryNet from ModelConfig, filter out `mc_samples`.

### 4.2 FSRSConfig — FSRS-6 Population Baseline (21 weights)

Store the full 21 FSRS-6 default population parameters as a list:

```python
@dataclass
class FSRSConfig:
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
```

### 4.3 TrainingConfig — Reptile Meta-Training

```python
@dataclass
class TrainingConfig:
    # Outer loop
    n_iters: int = 50_000                 # Total meta-iterations
    meta_batch_size: int = 16             # Tasks (students) per outer update
    outer_lr_start: float = 1e-3          # Outer Adam LR start (cosine decay)
    outer_lr_end: float = 1e-4            # Outer Adam LR end
    epsilon_start: float = 0.10           # Reptile step size start (linear decay)
    epsilon_end: float = 0.01             # Reptile step size end

    # Inner loop
    inner_lr: float = 0.01               # Per-student Adam LR
    inner_steps_phase1: int = 5           # k steps for rapid adaptation
    inner_steps_phase2: int = 20          # k steps for full personalization
    task_batch_size: int = 32             # Reviews per inner-loop mini-batch
    support_ratio: float = 0.70           # Fraction of reviews for inner-loop training

    # Loss weights
    stability_loss_weight: float = 0.10
    monotonicity_loss_weight: float = 0.01

    # Warm-start
    warmstart_epochs: int = 10            # Pre-train on FSRS-6 predictions
    warmstart_lr: float = 1e-3

    # Operational
    seed: int = 42
    log_every: int = 100
    eval_every: int = 1_000
    save_every: int = 5_000
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
```

Implement two schedule methods:
- `epsilon_schedule(iteration)`: **Linear decay** from `epsilon_start` to `epsilon_end` over `n_iters`
- `outer_lr_schedule(iteration)`: **Cosine decay** from `outer_lr_start` to `outer_lr_end`

### 4.4 AdaptationConfig — Three-Phase Onboarding

```python
@dataclass
class AdaptationConfig:
    phase1_threshold: int = 5             # Reviews before first adaptation
    phase2_threshold: int = 50            # Reviews before full personalization
    phase1_k_steps: int = 5              # Gradient steps used in Phase 2 rapid adaptation
    phase2_k_steps: int = 10             # Gradient steps used in Phase 2 (intermediate)
    phase3_k_steps: int = 20             # Phase 3 gradient steps
    phase3_inner_lr: float = 0.02         # Higher LR for full personalization
    adapt_every_n_reviews: int = 5        # Re-adapt frequency
    streaming_after: int = 50             # Enable per-review gradient steps
```

### 4.5 SchedulingConfig — Interval Computation

```python
@dataclass
class SchedulingConfig:
    desired_retention: float = 0.90
    min_interval: int = 1                 # Days
    max_interval: int = 365               # Days
    fuzz_range: tuple = (0.95, 1.05)      # +/-5% random fuzz to prevent card clustering
    difficulty_centre: float = 5.0        # D reference point for discounting
    difficulty_slope: float = 0.05        # Interval penalty per unit of D above centre
    uncertainty_discount: float = 0.5     # Max interval reduction from high uncertainty
```

### 4.6 Root Config

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

## 5. Neural Architecture (`models/`)

### 5.1 MemoryNet (`models/memory_net.py`)

Build a feedforward neural network (~50K parameters) that predicts memory-state transitions.

**Architecture:**
```
Input: (batch, 49)    # assembled feature vector

h1 = Linear(49, 128) -> LayerNorm(128) -> GELU -> Dropout(0.1)
h2 = Linear(128, 128) -> LayerNorm(128) -> GELU -> Dropout(0.1)
h3 = Linear(128, 64)  -> GELU -> Dropout(0.1)

Output heads (all from h3's 64-dim output):
  stability_head:  Linear(64, 1) -> Softplus -> multiply by S_prev -> clamp [0.001, 36500]
  difficulty_head: Linear(64, 1) -> Sigmoid -> multiply by 9, add 1 -> range [1, 10]
  recall_head:     Linear(64, 1) -> Sigmoid -> range [0, 1]
```

**Critical Design Rationale:**
- **LayerNorm** (NOT BatchNorm): Inner-loop SGD may use batch size 1 during streaming updates. BatchNorm fails with single samples; LayerNorm works.
- **GELU** activation: Smoother gradients than ReLU, better for meta-learning.
- **~50K parameters**: Intentionally small — larger models overfit during few-shot inner loop and defeat the purpose of meta-learning. 5-20 gradient steps must meaningfully move the weights.
- **Multiplicative stability head**: `S_next = Softplus(output) * S_prev` encodes the cognitive science principle that stability evolves *relative* to its current value, not in absolute terms.
- **Stability clamping** to `[0.001, 36500]`: Prevents NaN during Reptile inner loops from extreme values.

**Feature Assembly** — `build_features()` method constructs the 49-dim input vector:

| Component | Dims | Normalization |
|-----------|------|---------------|
| D_prev | 1 | Divide by 10.0 -> [0, 1] |
| S_prev | 1 | log(S_prev) — log-scale for numerical stability |
| R_at_review | 1 | Already in [0, 1] |
| delta_t (elapsed days) | 1 | log(delta_t + 1) — handles t=0 gracefully |
| Grade | 4 | One-hot encoding: grades 1-4 map to indices 0-3 in a 4-dim vector via `F.one_hot((grade-1).clamp(0,3), num_classes=4)` |
| Review count | 1 | log(count) — log-scaled |
| User stats | 8 | Mean difficulty, mean log-stability, plus zeros for other stats |
| GRU context | 32 | Final hidden state from GRU history encoder |
| **Total** | **49** | |

**Dynamic padding safety net**: If the assembled feature vector's dimension doesn't match `input_dim`, pad with zeros or truncate. This protects against dimension mismatches across feature changes.

**Return type**: `MemoryState` NamedTuple with fields: `S_next`, `D_next`, `p_recall`.

**Required methods:**
- `forward(D_prev, S_prev, R_at_review, delta_t, grade, review_count, user_stats, [history_grades, history_delta_ts, history_lengths])` — full pipeline
- `forward_from_features(x, S_prev)` — fast path when features are pre-assembled
- `build_features(...)` — assemble the 49-dim vector from individual tensor inputs
- `predict_recall(features, S_prev)` — convenience: return only p_recall
- `predict_stability(features, S_prev)` — convenience: return only S_next
- `count_parameters()` — total trainable params (target ~50K)

### 5.2 GRU History Encoder (`models/gru_encoder.py`)

A 1-layer GRU that encodes per-card review history into a 32-dim context vector.

**Input sequence per card**: `[(grade_1, delta_t_1), ..., (grade_n, delta_t_n)]`

Each timestep has 2 features:
- Normalized grade: `grade / 4.0` -> [0, 1]
- Log-scaled elapsed time: `log(delta_t + 1)`

**Architecture:**
```python
input_proj = Linear(2, 2)          # Small input projection for numerical conditioning
gru = GRU(input_size=2, hidden_size=32, num_layers=1, batch_first=True)
```

**Processing:**
1. Stack and normalize the input features into `(batch, seq_len, 2)` tensor
2. Apply input projection
3. Truncate to `max_len=32` (keep most recent reviews if longer)
4. Use `pack_padded_sequence` to handle variable-length sequences efficiently (with `enforce_sorted=False`)
5. Return the final hidden state `h_n` squeezed to `(batch, 32)`
6. Provide `zero_state(batch_size, device)` method for first-ever reviews (no history)

**Purpose**: Captures trajectory patterns — repeated "Again" presses, irregular gaps, consistent performance — that provide contextual signal beyond the current review event.

---

## 6. Training Subsystem (`training/`)

### 6.1 Multi-Component Loss (`training/loss.py`)

Implement `MetaSRSLoss(nn.Module)` with three loss terms. The constructor takes `stability_weight`, `monotonicity_weight`, and `w20` (the FSRS-6 power-law exponent).

**PRIMARY — Recall Prediction (Binary Cross-Entropy):**
```python
# Convert model's sigmoid output back to logits for numerical stability
p_clamped = p_recall_pred.clamp(1e-6, 1 - 1e-6)
logits = torch.log(p_clamped / (1 - p_clamped))
recall_L = BCEWithLogitsLoss(logits, recalled.float())
```

**AUXILIARY 1 — Stability-Curve Consistency:**
```python
# R_from_S = (0.9^(1/S))^(t^w20), computed in log-space:
# log(R) = log(0.9) * t^w20 / S
log_09 = -0.10536051565782628  # math.log(0.9)
t_pow = elapsed_days.clamp(min=1e-6) ** w20
log_R = log_09 * t_pow / S_next_pred.clamp(min=1e-3)
R_from_S = torch.exp(log_R.clamp(min=-20, max=0))
stable_L = MSELoss(R_from_S, recalled.float())
```
This loss ensures the predicted stability S is consistent with the power-law forgetting curve.

**AUXILIARY 2 — Monotonicity Constraint:**
```python
success_mask = (grade >= 2).float()
delta_S = S_next_pred - S_prev
mono_L = (F.relu(-delta_S) * success_mask).mean()
```
On successful recall, stability should NOT decrease. This is a well-established cognitive science finding.

**Combined:**
```python
total = recall_L + 0.10 * stable_L + 0.01 * mono_L
```

**NaN guard**: If `total` is NaN (can happen during early Reptile training with extreme S values), fall back to `recall_L` only.

**Return**: Dictionary `{"total", "recall", "stability", "monotonicity"}`.

Also implement a convenience function `compute_loss(model, batch, loss_fn)` that runs model forward pass then computes loss from the batch dict.

### 6.2 Reptile Meta-Learner (`training/reptile.py`)

Implement three components: `inner_loop()`, `reptile_update()`, and `ReptileTrainer` class.

**Inner Loop** — `inner_loop(phi_state_dict, model, task, loss_fn, k_steps, inner_lr, batch_size, device)`:
```python
def inner_loop(phi_state_dict, model, task, loss_fn, k_steps=5, inner_lr=0.01, batch_size=32, device="cpu"):
    model.load_state_dict(phi_state_dict, strict=True)   # Copy phi, don't modify original
    model.train()
    optimizer = Adam(model.parameters(), lr=inner_lr)

    for step in range(k_steps):
        batch = sample_batch(task.support_set, batch_size, device)
        losses = compute_loss(model, batch, loss_fn)
        optimizer.zero_grad()
        losses["total"].backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)  # CRITICAL for stability
        optimizer.step()

    return deepcopy(model.state_dict())   # Adapted parameters W_i
```

**Reptile Update** — `reptile_update(phi, adapted_weights, epsilon)`:
```python
# phi = phi + epsilon * mean(W_i - phi)
for key in phi:
    reptile_grad = sum(W_i[key] - phi[key] for W_i in adapted_weights) / n_tasks
    new_phi[key] = phi[key] + epsilon * reptile_grad
```

**ReptileTrainer** class orchestrates the full outer loop:
- Holds `self.phi` (meta-parameters as OrderedDict)
- Creates MetaSRSLoss with config weights
- Implements `train(task_sampler, n_iters, eval_fn, resume_from)`:
  1. Optional TensorBoard SummaryWriter setup
  2. Optional checkpoint resume
  3. For each iteration:
     a. Compute `epsilon` via linear decay schedule
     b. Sample `meta_batch_size` tasks
     c. For each task: run `inner_loop`, collect adapted weights
     d. Evaluate each adapted model on query set for monitoring
     e. Apply `reptile_update` to phi
     f. Log every `log_every` iterations
     g. Run eval callback every `eval_every` iterations
     h. Save checkpoint every `save_every` iterations
  4. Save final `phi_star.pt` with `{"phi": phi, "iteration": n, "config": config}`

### 6.3 FSRS-6 Warm-Start (`training/fsrs_warmstart.py`)

Implement the complete FSRS-6 algorithm as class `FSRS6` and a `warm_start_from_fsrs6()` function.

**FSRS6 class methods:**
- `retrievability(t, S)` — scalar forgetting curve computation
- `retrievability_batch(t, S)` — vectorized PyTorch tensor version
- `initial_stability(grade)` — `w[grade-1]` (w0..w3)
- `initial_difficulty(grade)` — `w4 - (grade-3) * w5`
- `stability_after_success(S, D, R, grade)` — full success update formula
- `stability_after_lapse(S, D, R)` — full lapse update formula
- `update_difficulty(D, grade)` — with damping and mean-reversion
- `step(S, D, elapsed_days, grade)` — process one review, return (S_next, D_next, R)
- `simulate_student(reviews)` — run FSRS-6 on full review history, return augmented records

**Warm-start function:**
```python
def warm_start_from_fsrs6(model, dataset, config, device):
    # Pre-train MemoryNet for warmstart_epochs to minimize:
    # L = BCE(p_recall, R_target) + 0.5*MSE(log(S_pred+1), log(S_target+1)) + 0.5*MSE(D_pred, D_target)
    # Uses Adam with warmstart_lr, gradient clipping to 1.0
    # Modifies model in-place and returns it
```

---

## 7. Data Pipeline (`data/task_sampler.py`)

### 7.1 Data Structures

**Review** dataclass — single review event:
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

**Task** dataclass — one student's complete review history:
```python
@dataclass
class Task:
    student_id: str
    reviews: List[Review]
    support_set: List[Review] = field(default_factory=list)
    query_set: List[Review] = field(default_factory=list)

    def split(self, support_ratio=0.70):
        """CHRONOLOGICAL split — first 70% support, last 30% query. Never random shuffle.
        This temporal ordering is essential: the model must predict future reviews from past history,
        simulating real-world deployment conditions."""
```

### 7.2 Batch Construction

Implement `reviews_to_batch(reviews, device, history_len=32)` that converts a list of Reviews into a dictionary of tensors:

**Output keys:**
- `D_prev`: (batch,) float
- `S_prev`: (batch,) float
- `R_at_review`: (batch,) float
- `delta_t`: (batch,) float — elapsed days
- `grade`: (batch,) long
- `review_count`: (batch,) float — accumulated count per card within this batch
- `recalled`: (batch,) float — binary labels
- `user_stats`: (batch, 8) float — [mean_D, log(mean_S), zeros...]
- `history_grades`: (batch, history_len) float — per-card grade history for GRU
- `history_delta_ts`: (batch, history_len) float — per-card elapsed time history
- `history_lengths`: (batch,) long — actual sequence lengths (clamped min=1 for pack_padded)
- `S_target`: (batch,) float
- `D_target`: (batch,) float

**Review history tracking**: As you iterate through reviews chronologically, maintain a per-card history buffer `Dict[card_id, List[Tuple[grade, delta_t]]]`. For each review:
1. Look up existing history for this card
2. Pad/truncate to `history_len`
3. Record lengths
4. THEN append the current review's (grade, elapsed_days) to the buffer for future reviews

### 7.3 TaskSampler

```python
class TaskSampler:
    def __init__(self, tasks, support_ratio=0.70, min_reviews=10, seed=42):
        # Filter students with fewer than min_reviews
        # Split each remaining task into support/query
    def sample(self, batch_size=16) -> List[Task]:
        # Sample with replacement using seeded RNG
```

### 7.4 Synthetic Data Generation

`ReviewDataset.generate_synthetic(n_students=500, reviews_per_student=100, n_cards=200, seed=42)`:

For each student, simulate reviews using FSRS-6:
1. Randomly select a card
2. If first review: draw grade from weights [Again=15%, Hard=20%, Good=50%, Easy=15%], use FSRS initial S/D for grade=3 (Good) as defaults
3. If subsequent review: compute R from forgetting curve, simulate recall with Bernoulli(R), assign grade accordingly (success -> weighted [Hard 20%, Good 60%, Easy 20%], failure -> Again)
4. Run FSRS step to compute S_next, D_next as targets
5. Accumulate timestamps (elapsed_days * 86400 seconds)

Also implement `ReviewDataset.from_csv(csv_path)` that loads real data from CSV with columns: `student_id, card_id, timestamp, elapsed_days, grade`.

---

## 8. Inference Components (`inference/`)

### 8.1 Scheduler (`inference/scheduling.py`)

**ScheduleResult** dataclass:
```python
@dataclass
class ScheduleResult:
    card_id: str
    interval_days: int            # Recommended review interval
    p_recall_mean: float          # Expected recall probability
    p_recall_sigma: float         # Epistemic uncertainty (from MC Dropout)
    S_pred: float                 # Predicted stability (days)
    D_pred: float                 # Predicted difficulty [1, 10]
    confidence_factor: float      # Uncertainty discount applied
    difficulty_factor: float      # Difficulty discount applied
```

**Scheduler class** — constructor takes `model`, `config` (SchedulingConfig), `mc_samples` (default 20).

**Monte Carlo Dropout Uncertainty** — `predict_with_uncertainty(features, S_prev)`:
```python
model.train()  # CRITICAL: keep dropout active at inference time!
with torch.no_grad():
    for _ in range(mc_samples):
        state = model.forward_from_features(features, S_prev)
        # Collect p_recall, S_next, D_next across all samples
# Return: mean and std of p_recall, mean of S and D
```

**Interval Computation** — `compute_interval(S_pred, D, sigma)`:
```python
base_interval = S_pred    # When desired_retention=0.90, interval ~ S

# Uncertainty discount: high uncertainty -> shorter interval (conservative)
confidence_factor = 1.0 - uncertainty_discount * min(sigma, 1.0)
confidence_factor = clamp(confidence_factor, 0.5, 1.0)

# Difficulty discount: harder cards get slightly shorter intervals
difficulty_factor = 1.0 - difficulty_slope * (D - difficulty_centre)
difficulty_factor = clamp(difficulty_factor, 0.5, 1.5)

# Fuzz: +/-5% random to prevent card review clustering
fuzz = uniform(fuzz_range[0], fuzz_range[1])

interval = round(base_interval * confidence_factor * difficulty_factor * fuzz)
interval = clamp(interval, min_interval, max_interval)
```

**Single card** — `schedule_card(card_id, features, S_prev)` -> ScheduleResult
**Batch** — `schedule_deck(card_ids, features, S_prev)` -> List[ScheduleResult]

**Card Selection Strategies** — `select_next_card(results, strategy)`:
1. `most_due`: Card with lowest interval (most overdue)
2. `highest_uncertainty`: Card with highest sigma (explore)
3. `uncertainty_weighted`: Score = `(1 - p_recall) + 0.5 * sigma`, sample from `softmax(scores * 3.0)` — balanced exploration/exploitation

### 8.2 FastAdapter (`inference/adaptation.py`)

**AdaptationPhase** enum: `ZERO_SHOT`, `RAPID_ADAPT`, `FULL_PERSONAL`

**FastAdapter class** — manages per-student model personalization:

Constructor takes `model`, `phi_star`, `config` (MetaSRSConfig), `device`.
Stores:
- `phi_star` (read-only deepcopy of meta-initialization)
- `theta_student` (student's personalized parameters, starts as copy of phi_star)
- `reviews` (List[Review] — full history for this student)
- `_reviews_since_last_adapt` counter

**Phase property** — determines current phase from `len(self.reviews)`:
- < phase1_threshold (5): `ZERO_SHOT`
- < phase2_threshold (50): `RAPID_ADAPT`
- >= phase2_threshold: `FULL_PERSONAL`

**`add_review(review)`** — the main entry point:
- Phase 1 (ZERO_SHOT): No adaptation, just accumulate reviews
- Phase 2 (RAPID_ADAPT): Every `adapt_every_n_reviews` (5): run k=5 gradient steps from phi_star (always from meta-init, not from theta)
- Phase 3 (FULL_PERSONAL): Every 5 reviews: run k=20 steps from theta_student (continuing from current personalized params). Plus: after 50 reviews, do a single streaming gradient step on the last 8 reviews after every individual review.

**`_adapt(k_steps, inner_lr, from_phi)`** — run gradient steps:
- Load either phi_star or theta_student
- Create Adam optimizer
- Run k_steps on ALL accumulated reviews
- Save adapted weights as theta_student

**`_streaming_step()`** — single gradient step on last min(8, total) reviews

**`get_model()`** — returns model loaded with appropriate parameters:
- Phase 1: load phi_star, set eval mode
- Phase 2/3: load theta_student, set eval mode

**`get_state()` / `load_state()`** — serialize/restore adapter state for persistence

---

## 9. Evaluation Framework (`evaluation/metrics.py`)

### 9.1 EvalResults Dataclass

```python
@dataclass
class EvalResults:
    auc_roc: float = 0.0              # Target > 0.80
    calibration_error: float = 0.0    # Target < 0.05
    cold_start_auc: float = 0.0       # Target > 0.73
    adaptation_speed: int = 0          # Target < 30 reviews
    retention_30d: float = 0.0        # Target > 85%
    rmse_stability: float = 0.0       # Target < 2.0 days
    recall_loss: float = 0.0
    n_samples: int = 0
    phase1_auc: float = 0.0
    phase2_auc: float = 0.0
    phase3_auc: float = 0.0

    def summary(self) -> str:
        # Format all metrics with target check marks
```

### 9.2 MetaSRSEvaluator Class

**`compute_auc_roc(predictions, labels)`**: Use sklearn.metrics.roc_auc_score if available; otherwise implement a manual AUC via trapezoidal rule on sorted predictions. Handle edge case where all labels are the same class (return 0.5).

**`evaluate_on_tasks(phi, tasks, k_steps=5)`**: For each test task:
1. **Cold-start**: Load phi, evaluate on query set (no adaptation)
2. **Adapted**: Run inner_loop on support set, evaluate adapted model on query set
3. Collect all predictions and labels
4. Handle NaN predictions (replace with 0.5, warn)
5. Compute AUC, calibration error, RMSE(S)

**`cold_start_curve(phi, tasks, review_counts=[0,5,10,20,30,50])`**:
- For each n_reviews count: create truncated tasks with only n support reviews
- n=0: zero-shot (use phi directly)
- n>0: adapt with min(5, n_reviews) gradient steps
- Return `{n_reviews: AUC}` dictionary

**`ablation_study(phi_variants, tasks)`**: Run evaluate_on_tasks for each named variant, return dict of results.

---

## 10. Training Script (`train.py`)

CLI entry point orchestrating the full pipeline.

**CLI Arguments:**
| Argument | Type | Default | Purpose |
|----------|------|---------|---------|
| `--data` | str | None | Path to review CSV |
| `--synthetic` | flag | False | Use synthetic data |
| `--n-students` | int | 500 | Synthetic student count |
| `--n-iters` | int | None | Override meta-iterations |
| `--inner-steps` | int | None | Override inner loop steps |
| `--batch-size` | int | None | Override meta batch size |
| `--resume` | str | None | Resume from checkpoint |
| `--skip-warmstart` | flag | False | Skip FSRS-6 pre-training |
| `--device` | str | "auto" | Device (cpu/cuda/auto) |
| `--seed` | int | 42 | Random seed |
| `--eval-only` | flag | False | Only evaluate |
| `--eval-checkpoint` | str | None | Checkpoint for eval-only |
| `--checkpoint-dir` | str | "checkpoints" | Checkpoint directory |
| `--log-dir` | str | "logs" | TensorBoard log directory |

**Execution Flow:**
1. Parse args, set seed (random, numpy, torch, cuda)
2. Determine device (auto -> cuda if available, else cpu)
3. Load data: `--synthetic` -> `ReviewDataset.generate_synthetic()`, `--data` -> `ReviewDataset.from_csv()`, else default synthetic
4. Split tasks 80/20 by student (shuffle first)
5. Create TaskSampler from train tasks
6. Create MemoryNet with explicit config parameters (filter out mc_samples)
7. **Eval-only mode**: Load checkpoint, run evaluator, print results + cold-start curve, return
8. **Warm-start** (unless --skip-warmstart or --resume): Generate FSRS-6 targets from first 200 train tasks, build features batches, call `warm_start_from_fsrs6()`
9. **Reptile meta-training**: Create ReptileTrainer, set up eval callback that runs evaluator on 50 test tasks, call `trainer.train()`
10. **Final evaluation**: Run evaluator on all test tasks, print results summary + cold-start curve

---

## 11. Test Suite (`tests/`)

Write comprehensive pytest tests. Target 100+ tests across all modules, all passing in ~6 seconds on CPU.

### 11.1 Test Fixtures (`conftest.py`)

```python
def create_model(config=None):
    """Create MemoryNet from ModelConfig, filtering out mc_samples (not a MemoryNet init param)."""
    if config is None:
        config = ModelConfig()
    import inspect
    valid_params = set(inspect.signature(MemoryNet.__init__).parameters.keys()) - {'self'}
    init_args = {k: v for k, v in config.__dict__.items() if k in valid_params}
    return MemoryNet(**init_args)
```

Fixtures:
- `device` -> torch.device("cpu")
- `config` -> MetaSRSConfig()
- `model_config` -> ModelConfig()
- `training_config` -> TrainingConfig()
- `fsrs_config` -> FSRSConfig()
- `model` -> create_model(model_config)
- `fsrs` -> FSRS6()
- `sample_batch_tensors` -> dict of properly-shaped tensors for batch_size=4
- `sample_reviews` -> list of 4 Review objects with realistic values
- `synthetic_tasks` -> ReviewDataset.generate_synthetic(n_students=20, reviews_per_student=50, n_cards=30, seed=42)

### 11.2 Test Categories and Key Assertions

**Architecture tests** (`test_memory_net.py`, `test_models.py`):
- Parameter count is approximately 50K (within reasonable range)
- Output shapes are correct: (batch,) for each head
- Output ranges: S_next in [0.001, 36500], D_next in [1, 10], p_recall in [0, 1]
- Forward pass produces finite outputs
- Backward pass produces non-zero gradients
- Feature assembly produces (batch, 49) tensors
- GRU encoder produces (batch, 32) context vectors
- Model handles zero-length history gracefully

**Loss tests** (`test_loss.py`):
- Each component is non-negative
- Total loss is finite (not NaN, not Inf)
- NaN guard falls back to recall-only loss
- Monotonicity loss is zero when S_next >= S_prev on all success reviews
- Loss with perfect predictions is near zero
- compute_loss convenience function works end-to-end

**Meta-learning tests** (`test_reptile.py`):
- Inner loop produces parameters different from phi
- Reptile update moves phi toward adapted weights
- Epsilon is between phi and adapted (interpolation, not extrapolation)
- ReptileTrainer runs for 3 iterations without error
- Checkpoint saving/loading works

**Data tests** (`test_task_sampler.py`):
- Synthetic generation is deterministic (same seed = same data)
- Reviews have valid ranges (grade 1-4, elapsed >= 0, S_prev > 0)
- Task split produces ~70/30 support/query ratio
- reviews_to_batch produces correct tensor shapes and types
- History tracking accumulates correctly across reviews of same card
- TaskSampler filters students with too few reviews

**Scheduling tests** (`test_scheduling.py`):
- Higher uncertainty -> shorter intervals
- Higher difficulty -> shorter intervals
- Intervals are within [min_interval, max_interval]
- MC Dropout produces non-zero uncertainty (sigma > 0)
- Card selection strategies return valid results
- Schedule results have all expected fields

**Adaptation tests** (`test_adaptation.py`):
- Phase transitions at correct review counts: 0-4=ZERO_SHOT, 5-49=RAPID_ADAPT, 50+=FULL_PERSONAL
- Phase 1 uses phi_star parameters
- Phase 2 triggers adaptation when threshold met
- Phase 3 supports streaming updates
- State serialization round-trips correctly

**FSRS-6 tests** (`test_fsrs_warmstart.py`):
- Retrievability at t=0 is 1.0
- Retrievability decreases with time (monotonically for fixed S)
- Stability increases on successful recall
- Stability drops sharply on lapse (Again)
- Difficulty updates are bounded [1, 10]
- Initial values are correct for each grade
- Warm-start reduces loss over epochs

**Evaluation tests** (`test_evaluation.py`):
- Perfect predictions -> AUC = 1.0
- Random predictions -> AUC near 0.5
- Cold-start curve returns values for each requested n
- All-same-label edge case is handled (AUC = 0.5)

**Config tests** (`test_config.py`):
- Default values are correct
- Schedule functions produce expected ranges
- Epsilon decays linearly
- LR decays via cosine

**Integration tests** (`test_integration.py`):
- Full pipeline: generate synthetic -> warm-start -> train 3 iters -> evaluate
- End-to-end forward pass through model + loss + adaptation + scheduling

---

## 12. Critical Implementation Notes

1. **Numerical stability is the #1 concern**: Use log-space for the forgetting curve, clamp stability to [0.001, 36500], add NaN guards in loss, use BCEWithLogitsLoss with manual logit conversion.

2. **The model MUST be small (~50K params)**: This is not a compromise — it's a feature. Larger models overfit in 5-20 gradient steps and defeat meta-learning.

3. **LayerNorm, never BatchNorm**: Inner loop and streaming updates may use batch size 1.

4. **Gradient clipping (norm <= 1.0) in ALL inner loops**: Without this, rare outlier reviews cause catastrophic parameter updates.

5. **Chronological splits ONLY**: Never randomly shuffle reviews within a student's history. Support = first 70%, query = last 30%. This simulates real deployment.

6. **MC Dropout requires model.train()**: During inference uncertainty estimation, call `model.train()` (NOT `model.eval()`) so dropout stays active across Monte Carlo forward passes. Use `torch.no_grad()` for efficiency.

7. **FSRS-6 warm-start is optional but cuts training time by ~60%**: It ensures the model starts from cognitively-valid predictions instead of random noise.

8. **Phase 2 always adapts from phi_star**: In rapid adaptation, always start gradient steps from the meta-initialization, not from the previous theta. This prevents catastrophic forgetting on limited data.

9. **Phase 3 adapts from theta_student**: Once enough data exists, continue fine-tuning from the student's current personalized parameters for efficiency.

10. **Multiplicative stability update** (`Softplus(x) * S_prev`): This is not arbitrary — it encodes the cognitive science principle that memory stability evolves relative to its current value. A card with S=100 days should not jump to S=101 the same way a card with S=1 day jumps to S=2.

---

## 13. Performance Targets and Validation

Run the test suite with:
```bash
cd meta-srs && python -m pytest tests/ -v --tb=short
```
Target: 100+ tests, all passing, ~6 seconds on CPU.

Validate the training pipeline with:
```bash
cd meta-srs && python train.py --synthetic --n-students 100 --n-iters 500
```
This should complete in ~4 minutes on CPU and demonstrate the full pipeline.

### Performance Benchmarks

| Metric | Target | FSRS-6 Baseline |
|--------|--------|-----------------|
| AUC-ROC | > 0.80 | ~0.78 |
| Calibration Error | < 0.05 | ~0.06 |
| Cold-Start AUC (0 reviews) | > 0.73 | 0.78* (see note) |
| Adaptation Speed | < 30 reviews | N/A |
| 30-day Retention | > 85% | ~87% |
| RMSE(Stability) | < 2.0 days | ~2.5 |

*FSRS-6's "cold-start" number (0.78) is misleading: FSRS-6 uses pre-optimized population parameters, so it's already personalized at the population level. MetaSRS's 0.73 target is for a truly zero-shot neural model that will rapidly surpass FSRS-6 after just 5 reviews.

---

## 14. Key Papers and References

| Paper | Year | Relevance |
|-------|------|-----------|
| **Reptile** (Nichol & Schulman, OpenAI) | 2018 | Core meta-learning algorithm — arXiv:1803.02999 |
| **MAML** (Finn et al., ICML) | 2017 | Theoretical foundation for Reptile |
| **FSRS-6** (open-spaced-repetition) | 2024-2025 | State-of-the-art SRS; DSR model with 21 params |
| **KAR3L** (Shu et al., EMNLP) | 2024 | Content-aware SRS; DKT + BERT — arXiv:2402.12291 |
| **Half-Life Regression** (Settles & Meeder, ACL) | 2016 | ML-based forgetting curves (Duolingo) |
| **DKT** (Piech et al., NeurIPS) | 2015 | Deep knowledge tracing with LSTM |
| **PLATIPUS** (Finn et al., NeurIPS) | 2018 | Bayesian extension of MAML/Reptile |

---

*A student reviewing their first 5 flash cards already benefits from the forgetting patterns of thousands of prior learners — all within the first study session.*
