# Replit Agent Prompt: Recreate MetaSRS

> Build **MetaSRS** — a Python/PyTorch meta-learning spaced repetition engine with a Flask REST API. Uses Reptile meta-learning to find initialization parameters (φ\*) that fast-adapt to new students in 5–20 gradient steps. Serves as a back end for a Node.js web app.

---

## 1. Project Layout

```
.replit                          # run = "cd meta-srs && python api.py"
replit.nix                       # python312 + pip
.gitignore                       # __pycache__, *.pt, checkpoints/, logs/

meta-srs/
├── config.py                    # 5 dataclass configs (all hyperparams live here)
├── api.py                       # Flask REST API — main Replit entry point
├── train.py                     # CLI training script
├── requirements.txt             # torch, numpy, scikit-learn, tqdm, tensorboard
├── requirements-web.txt         # CPU-only: torch, numpy, flask, gunicorn
├── pytest.ini                   # testpaths=tests, --timeout=120
├── models/
│   ├── __init__.py              # exports MemoryNet, GRUHistoryEncoder
│   ├── memory_net.py            # ~35K-param neural model
│   └── gru_encoder.py           # GRU review-history encoder
├── training/
│   ├── __init__.py              # exports MetaSRSLoss, ReptileTrainer, FSRS6, etc.
│   ├── loss.py                  # 3-component loss (recall + stability + monotonicity)
│   ├── reptile.py               # inner_loop, reptile_update, ReptileTrainer
│   └── fsrs_warmstart.py        # FSRS-6 baseline model + warm-start pre-training
├── data/
│   ├── __init__.py              # exports Task, TaskSampler, ReviewDataset
│   └── task_sampler.py          # Review/Task dataclasses, reviews_to_batch, synthetic data
├── inference/
│   ├── __init__.py              # exports FastAdapter, Scheduler, etc.
│   ├── adaptation.py            # 3-phase fast adaptation (ZERO_SHOT → RAPID → FULL)
│   └── scheduling.py            # MC-Dropout uncertainty-aware interval scheduler
├── evaluation/
│   ├── __init__.py              # exports MetaSRSEvaluator, EvalResults
│   └── metrics.py               # AUC-ROC, calibration, cold-start curve
└── tests/                       # 139 tests (118 core + 21 API)
    ├── conftest.py, test_config.py, test_memory_net.py, test_models.py,
    ├── test_loss.py, test_fsrs_warmstart.py, test_task_sampler.py,
    ├── test_reptile.py, test_adaptation.py, test_scheduling.py,
    ├── test_evaluation.py, test_integration.py, test_api.py
```

All Python modules use **bare imports** (`from config import ...`). Files that run standalone (`api.py`, `train.py`, `conftest.py`) add the meta-srs root to `sys.path`.

---

## 2. Configuration (`config.py`)

Five nested `@dataclass` configs — copy these **exactly** as the reference implementation `config.py` (129 lines). Key values:

- **ModelConfig**: `input_dim=49`, `hidden_dim=128`, `gru_hidden_dim=32`, `history_len=32`, `user_stats_dim=8`, `dropout=0.1`, `mc_samples=20`
- **FSRSConfig**: 21 FSRS-6 weights `w[0..20]` (initial S per grade, difficulty params, stability update factors, power-law exponent `w20=0.4665`)
- **TrainingConfig**: `n_iters=50000`, `meta_batch_size=16`, `inner_lr=0.01`, `inner_steps_phase1=5`, `support_ratio=0.70`, loss weights `0.10`/`0.01`, `warmstart_epochs=10`. Has `epsilon_schedule()` (linear decay) and `outer_lr_schedule()` (cosine decay).
- **AdaptationConfig**: phase thresholds `5`/`50`, k-steps `5`/`10`/`20`, `adapt_every_n_reviews=5`
- **SchedulingConfig**: `desired_retention=0.90`, intervals `[1, 365]`, `fuzz_range=(0.95, 1.05)`, `uncertainty_discount=0.5`
- **MetaSRSConfig**: aggregates all five

**⚠️ Gotcha:** `mc_samples` is in ModelConfig but `MemoryNet.__init__()` does NOT accept it. Filter it out when constructing:
```python
model_kwargs = {k: v for k, v in cfg.model.__dict__.items()
                if k in MemoryNet.__init__.__code__.co_varnames}
```

---

## 3. MemoryNet (`models/memory_net.py`)

~35K-param feedforward network predicting memory-state transitions.

**Input (dim=49):** `concat([D_prev/10, log(S_prev), R_at_review, log(delta_t+1), grade_onehot(4), log(review_count), user_stats(8), gru_context(32)])`

**Architecture:** `Linear→LayerNorm→GELU→Dropout` × 2 (128-wide), then `Linear(128,64)→GELU→Dropout`, then 3 heads:
- `S_next = Softplus(head) * S_prev` clamped `[0.001, 36500]`
- `D_next = Sigmoid(head) * 9 + 1` → range `[1, 10]`
- `p_recall = Sigmoid(head)` → range `[0, 1]`

Returns `MemoryState` NamedTuple `(S_next, D_next, p_recall)`.

**Methods:** `build_features(...)→(B,49)`, `forward_from_features(x, S_prev)→MemoryState`, `forward(D_prev, S_prev, ...)→MemoryState`, `predict_recall`, `predict_stability`, `count_parameters`.

### GRU History Encoder (`models/gru_encoder.py`)
1-layer GRU encoding card review history into 32-dim context. Input: `grade/4.0` + `log(delta_t+1)` per timestep. Has `input_proj=Linear(2,2)`, uses `pack_padded_sequence`, returns `h_n.squeeze(0)`.

---

## 4. Data Pipeline (`data/task_sampler.py`)

**Review** dataclass: `card_id, timestamp, elapsed_days, grade(1-4), recalled, S_prev, D_prev, R_at_review, S_target, D_target`

**Task** dataclass: one student's review history with `split(support_ratio)` → `support_set`/`query_set`.

**`reviews_to_batch(reviews, device, history_len=32)`** → tensor dict for MemoryNet. Builds per-card running histories so each review only sees its *prior* reviews. Returns: `D_prev, S_prev, R_at_review, delta_t, grade, review_count, user_stats(B,8), recalled, S_target, D_target, history_grades(B,32), history_delta_ts(B,32), history_lengths`.

**TaskSampler**: filters tasks with `<10` reviews, splits on init, `sample(batch_size)` with replacement.

**ReviewDataset**: `from_csv(path)` and `generate_synthetic(n_students, reviews_per_student, n_cards, seed)` using FSRS-6 simulation.

---

## 5. FSRS-6 Baseline (`training/fsrs_warmstart.py`)

**FSRS6 class** — implements the 21-parameter model:
- Forgetting curve: `R(t,S) = (0.9^(1/S))^(t^w20)`
- `step(S, D, elapsed, grade) → (S_next, D_next, R)` — handles success/lapse stability updates, difficulty mean-reversion
- `simulate_student(reviews)` — augments with S, D, R for synthetic data
- `retrievability_batch(t, S)` — vectorised PyTorch version

**`warm_start_from_fsrs6(model, dataset, config, device)`** — pre-trains MemoryNet to match FSRS-6 outputs. Loss = `BCE(p, R_target) + 0.5*MSE(log(S+1), log(S_target+1)) + 0.5*MSE(D, D_target)`.

---

## 6. Loss & Reptile Training

### MetaSRSLoss (`training/loss.py`)
`total = recall_L + 0.10 * stability_L + 0.01 * monotonicity_L`
1. **Recall:** `BCEWithLogitsLoss` on manual logits (`p → log(p/(1-p))`, clamped)
2. **Stability consistency:** log-space `R_from_S = exp(log(0.9) * t^w20 / S)`, then `MSE(R_from_S, recalled)`
3. **Monotonicity:** penalise `S_next < S_prev` on successful recalls
4. **NaN guard:** falls back to recall-only if total is NaN

### Reptile (`training/reptile.py`)
- `inner_loop(phi, model, task, ...)` — loads phi, Adam, k gradient steps on support set, clip_grad_norm 1.0, returns adapted weights
- `reptile_update(phi, adapted_list, epsilon)` — `phi += epsilon * mean(W_i - phi)`
- **ReptileTrainer**: orchestrates sampling → inner loops → reptile updates with epsilon decay, cosine LR, TensorBoard, checkpointing. Saves `phi_star.pt`.

---

## 7. Inference

### FastAdapter (`inference/adaptation.py`)
Per-student personalisation with 3 phases:
- **ZERO_SHOT** (< 5 reviews): use φ\* directly
- **RAPID_ADAPT** (5–49): every 5 reviews, k=5 steps from φ\* on all reviews
- **FULL_PERSONAL** (50+): k=20 steps at LR=0.02 from current θ, plus streaming single-step updates

Methods: `add_review()`, `get_model()→MemoryNet` (eval mode), `get_state()`/`load_state()`

### Scheduler (`inference/scheduling.py`)
MC Dropout uncertainty: runs model `mc_samples` times in train mode under `no_grad()`, returns mean±std.

`compute_interval(S, D, sigma)`: `base=S × confidence(σ) × difficulty(D) × fuzz(0.95–1.05)`, clamped [1, 365].

`select_next_card(results, strategy)`: `"most_due"`, `"highest_uncertainty"`, `"uncertainty_weighted"` (softmax sampling).

---

## 8. Evaluation (`evaluation/metrics.py`)

**EvalResults**: auc_roc, calibration_error, cold_start_auc, rmse_stability, etc. Has `summary()` with ✓/✗.

**MetaSRSEvaluator**: `evaluate_on_tasks(phi, tasks)` — per-task cold-start + adapted eval on query set. `cold_start_curve()` — AUC at [0, 5, 10, 20, 30, 50] reviews. Falls back to manual AUC if sklearn unavailable.

---

## 9. Flask REST API (`api.py`)

**Entry point for Replit.** `create_app(checkpoint_path, config)` factory pattern.

- Loads MemoryNet + checkpoint at startup (env: `METASRS_CHECKPOINT`)
- Per-student `FastAdapter` dict with `threading.Lock` for thread safety
- FSRS-6 instance for initial card states

| Method | Path | Body | Returns |
|--------|------|------|---------|
| GET | `/api/health` | — | `{status, model_params, active_students}` |
| POST | `/api/review` | `{student_id, card_id, grade, elapsed_days}` | `{phase, n_reviews, schedule: {...}}` |
| POST | `/api/schedule` | `{student_id, cards: [{card_id, elapsed_days}]}` | `{cards: [ScheduleResult...]}` |
| GET | `/api/student/<id>/status` | — | `{phase, n_reviews, unique_cards}` |
| POST | `/api/student/<id>/reset` | — | `{status: "reset"}` |
| POST | `/api/next-card` | `{student_id, cards, strategy?}` | `{selected: ScheduleResult}` |

Invalid requests → 400 with details. 404/500 → JSON errors.

Module-level: reads `PORT` env (default 5000), `METASRS_CHECKPOINT` (default `checkpoints/phi_star.pt`).

---

## 10. Training Script (`train.py`)

```bash
python train.py --synthetic --n-students 100 --n-iters 500   # Quick test
python train.py --data reviews.csv --n-iters 50000            # Full run
python train.py --eval-only --eval-checkpoint checkpoints/phi_star.pt
```

Pipeline: load/generate data → 80/20 student split → MemoryNet → FSRS-6 warm-start → Reptile meta-training → final evaluation.

---

## 11. Tests (139 total)

All tests use `sys.path.insert` for bare imports. `conftest.py` provides `create_model()` (filters out mc_samples), `device`, `config`, `model`, `fsrs`, `sample_batch_tensors`, `sample_reviews`, `synthetic_tasks` fixtures.

| File | # | What it covers |
|------|---|----------------|
| test_config.py | 11 | Defaults, schedule boundaries/monotonicity, aggregation |
| test_memory_net.py | 12 | Shapes, ranges, NaN-free, features dim=49, gradient flow |
| test_models.py | 5 | GRU encoder shapes, truncation, zero_state |
| test_loss.py | 9 | Components, weights, NaN fallback, compute_loss |
| test_fsrs_warmstart.py | 17 | Forgetting curve, S/D updates, simulate_student |
| test_task_sampler.py | 10 | Batch construction, review_count, synthetic data |
| test_reptile.py | 7 | Inner loop, reptile_update direction, trainer runs |
| test_adaptation.py | 8 | Phase transitions, get_model, state roundtrip |
| test_scheduling.py | 13 | Intervals, uncertainty, MC dropout, card selection |
| test_evaluation.py | 6 | AUC, cold-start curve, manual fallback |
| test_integration.py | 3 | End-to-end pipeline, checkpoint save/load |
| test_api.py | 21 | All endpoints, validation, phase transitions |

---

## 12. Key Implementation Details

1. **Numerical stability:** BCEWithLogitsLoss with manual p→logit, log-space R_from_S, S clamped [0.001, 36500], grad clip 1.0 everywhere, NaN fallback in loss
2. **Thread safety:** Flask API uses `threading.Lock` around adapter dict; each student gets own MemoryNet copy
3. **Fully online:** No BERT, no file I/O at inference. Only loads `phi_star.pt` at startup.
4. **mc_samples filtering:** ModelConfig has it, MemoryNet doesn't accept it — always filter when constructing

---

## 13. Verification

```bash
cd meta-srs && pip install -r requirements-web.txt pytest pytest-timeout
python -m pytest tests/ -v --tb=short   # 139 pass in ~6s
python api.py                            # Flask on :5000
```

- [ ] All 139 tests pass
- [ ] `/api/health` returns `{"status": "ok"}` with ~35K model_params
- [ ] `/api/review` accepts grade 1-4, returns schedule with interval_days
- [ ] Feature vector dimension is 49
