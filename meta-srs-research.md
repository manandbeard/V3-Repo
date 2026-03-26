# Meta-Initialized Spaced Repetition Scheduler
## From-Scratch Algorithm Design with Reptile Meta-Learning

> **Goal**: A state-of-the-art SRS that learns a meta-initialization from multi-user review histories (Reptile) and fast-adapts to new students with just a handful of gradient steps.

---

## 1. Foundations: What We're Building On

### 1.1 The Three-Component Memory Model (DSR)

Every modern high-performance SRS is built on three latent memory variables per card per student:

| Variable | Symbol | Meaning |
|---|---|---|
| **Difficulty** | D ∈ [1, 10] | Intrinsic hardness of the card |
| **Stability** | S ∈ ℝ⁺ | Days until retention drops to 90% |
| **Retrievability** | R ∈ [0, 1] | Current recall probability |

**FSRS-6 Forgetting Curve** (state of the art as of 2025, default in Anki):
```
R(t, S) = (0.9^(1/S))^(t^w20)   where w20 ∈ (0.1, 0.8)
```
This power-law form beats exponential decay because the population average of two exponential curves with different S values is better approximated by a power function—a key insight that took the field years to discover.

**Stability Update (success)**:
```
S' = S · (e^w8 · (11 - D) · S^(-w9) · (e^(w10·(1-R)) - 1) · w15_grade · w16_grade + 1)
```
**Stability Update (lapse/Again)**:
```
S_f = w11 · D^(-w12) · ((S+1)^w13 - 1) · e^(w14·(1-R))
```
**Difficulty Update**:
```
ΔD = -w6 · (G - 3)    [grade-based delta]
D'' = D + ΔD · (10 - D) / 9    [linear damping toward max]
D' = w5 · D_0(G=4) + (1 - w5) · D''    [mean reversion]
```

FSRS-6 has **21 learnable parameters (w0–w20)** optimized per user on their own review history using gradient descent with binary cross-entropy loss.

### 1.2 The Cold-Start Problem

FSRS-6 is excellent but **requires hundreds to thousands of reviews per user** before the parameters converge. New users start from a generic population average. This is the exact problem meta-learning solves.

### 1.3 Reptile: Scalable Meta-Learning (Nichol & Schulman, OpenAI 2018)

Reptile is a first-order meta-learning algorithm. Unlike MAML (which requires second-order gradients), Reptile only needs standard SGD.

**Reptile Pseudocode**:
```
Initialize meta-parameters φ
for each meta-iteration:
    Sample a task τᵢ (e.g., one student's data)
    Perform k steps of SGD on τᵢ, starting from φ → get Wᵢ
    Update: φ ← φ + ε · (Wᵢ - φ)
```
The key insight: Reptile's update maximizes the inner product between gradients from different mini-batches of the same task, promoting within-task generalization. It approximates MAML's update with far less compute.

**Why Reptile over MAML for SRS**:
- Review histories vary in length (1 to 10,000 reviews) — Reptile handles variable-length inner loops gracefully
- No need to store or differentiate through computation graphs
- Student tasks are naturally i.i.d. across users (same card domain, different learners)
- Can plug any optimizer (Adam) into both inner and outer loops

---

## 2. System Architecture

### 2.1 High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    OFFLINE META-TRAINING                     │
│                                                              │
│  User Corpus ──► [Task Sampler] ──► Reptile Loop            │
│  (N users, each with review history H_i)                    │
│                                   ↓                          │
│                          Meta-Parameters φ*                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   ONLINE FAST ADAPTATION                     │
│                                                              │
│  New Student ──► [5–20 gradient steps on their reviews]     │
│                   starting from φ*                           │
│                              ↓                               │
│              Personalized θ_student ──► Schedule            │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Neural Memory Model (The Inner Network)

We replace FSRS's hand-crafted formulas with a small, differentiable neural network that predicts the memory state transition.

**Architecture**: `MemoryNet(θ)`

```
Input features at each review event:
  x = [
    D_prev,              # current difficulty (1-10)
    log(S_prev),         # log stability (numerical stability)
    R_at_review,         # retrievability when reviewed
    log(Δt + 1),         # time since last review
    grade,               # one-hot: [Again, Hard, Good, Easy]
    review_count,        # total reviews of this card
    card_embed,          # BERT/sentence-transformer embedding (dim 64, projected)
    user_stats,          # aggregate statistics (mean D, mean S, session length)
  ]
  dim(x) ≈ 64 + 8 + user_stats_dim

Architecture:
  h1 = LayerNorm(Linear(x, 128) + GELU)
  h2 = LayerNorm(Linear(h1, 128) + GELU)
  h3 = Linear(h2, 64) + GELU

Heads:
  S_next = Softplus(Linear(h3, 1)) · S_prev   # stability can only grow on success
  D_next = Sigmoid(Linear(h3, 1)) · 9 + 1     # D ∈ [1, 10]
  p_recall = Sigmoid(Linear(h3, 1))            # auxiliary recall prediction

Total params: ~50K (fast to adapt, low risk of overfitting on few shots)
```

**Key design choices**:
- `S_next = Softplus(...) · S_prev` enforces the stability-growth constraint from cognitive science
- LayerNorm instead of BatchNorm (works with batch size = 1 for inner-loop SGD)
- Small footprint so 5–20 gradient steps meaningfully move the parameters

### 2.3 Content-Aware Card Embeddings

Following KAR³L (Shu et al., EMNLP 2024), we embed the card's text content:

```python
# Offline: embed all cards once
card_embedding = SentenceTransformer("all-MiniLM-L6-v2").encode(card_front + " | " + card_back)
# 384-dim → project to 64-dim via trained linear layer
card_embed = Linear(384, 64)(card_embedding)
```

This enables **semantic transfer**: if a student knows "France → Paris", their model infers they're likely to know "Germany → Berlin" without a single review, because both cards cluster together in embedding space.

### 2.4 History Encoder (Sequence Context)

To capture the student's recent trajectory (not just per-card state), add a lightweight sequence model:

```
Review sequence for one card:
  [(grade_1, Δt_1), (grade_2, Δt_2), ..., (grade_n, Δt_n)]
  
Architecture: 1-layer GRU, hidden_dim=32
  context_h = GRU(reviews)[-1]   # final hidden state
  x = concat(x_static, context_h)
```

This captures patterns like "student has been pressing 'Again' repeatedly → instability signal."

---

## 3. The Reptile Training Loop

### 3.1 Task Definition

Each **task τᵢ** = one student's review history for a deck of N cards.

```python
Task = {
    "student_id": str,
    "reviews": List[{
        "card_id": str,
        "timestamp": int,
        "grade": int,          # 1=Again, 2=Hard, 3=Good, 4=Easy
        "elapsed_days": float,
    }],
    "card_embeddings": Dict[str, np.ndarray]
}
```

Each task provides a **support set** (first 70% of reviews chronologically) and **query set** (last 30%).

### 3.2 Inner Loop (Per-Student Adaptation)

```python
def inner_loop(phi, task, k_steps=5, inner_lr=0.01):
    """k gradient steps on support set, starting from meta-params phi"""
    theta = copy(phi)
    optimizer = Adam(theta, lr=inner_lr)
    
    for step in range(k_steps):
        batch = sample_batch(task.support_set, size=32)
        loss = compute_loss(theta, batch)  # binary cross-entropy on p_recall
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    return theta   # adapted parameters W_i
```

### 3.3 Loss Function

```python
def compute_loss(theta, batch):
    # Primary: recall prediction (binary cross-entropy)
    p_pred = MemoryNet(theta)(batch.features)
    recall_loss = BCELoss(p_pred, batch.recalled)
    
    # Auxiliary: stability consistency loss
    # Predicted S should match empirically observed forgetting curve
    s_pred = MemoryNet(theta).predict_stability(batch.features)
    r_from_s = forgetting_curve(s_pred, batch.elapsed_days)  # power law
    stability_loss = MSELoss(r_from_s, batch.recalled.float())
    
    # Auxiliary: monotonicity constraint on stability for success reviews
    # S' should be >= S when grade >= 2 (Hard, Good, Easy)
    mono_violation = ReLU(-(s_next - s_prev)[batch.grade >= 2]).mean()
    
    return recall_loss + 0.1 * stability_loss + 0.01 * mono_violation
```

### 3.4 Outer Loop (Meta-Update)

```python
def meta_train(dataset, n_meta_iters=50000, meta_lr=0.001, k=10, epsilon=0.1):
    phi = initialize_params()  # can warm-start from FSRS-6 population params
    meta_optimizer = Adam(phi, lr=meta_lr)
    
    for iteration in range(n_meta_iters):
        # Sample a minibatch of tasks (students)
        tasks = sample_tasks(dataset, batch_size=16)
        
        meta_gradient = zeros_like(phi)
        for task in tasks:
            W_i = inner_loop(phi, task, k_steps=k)
            meta_gradient += (W_i - phi)   # Reptile update direction
        
        # Apply average Reptile gradient
        # Equivalent to: phi = phi + epsilon * mean(W_i - phi)
        phi = phi + epsilon * meta_gradient / len(tasks)
        
        # Optional: evaluate on query sets for logging
        if iteration % 100 == 0:
            val_loss = evaluate_query_sets(phi, tasks)
            log(f"Iter {iteration}: val_loss={val_loss:.4f}")
    
    return phi  # meta-initialized parameters
```

**Practical detail**: Use `epsilon` schedule — start at 0.1, decay to 0.001. The Reptile paper shows Adam can be used instead of the raw update, treating `(W_i - phi)` as a gradient.

### 3.5 Warm-Starting from FSRS-6

Instead of random initialization, bootstrap φ from FSRS-6's population parameters:

```python
# Map FSRS-6's 21 parameters to MemoryNet's weight space
# by pre-training MemoryNet to match FSRS-6's predictions on a large corpus
phi = pretrain_to_match_fsrs6(MemoryNet, fsrs6_population_params, corpus)
# Then run Reptile from this warm start
phi_meta = meta_train(dataset, init=phi)
```

This dramatically accelerates convergence and ensures the model starts from a cognitively valid baseline.

---

## 4. Fast Adaptation for New Students

### 4.1 Onboarding Protocol

When a new student starts:

```
Phase 1: Zero-shot initialization (0 reviews)
  → Use φ* directly (meta-params = best population prior)
  → Schedule cards using φ* predictions + uncertainty-aware exploration

Phase 2: Rapid adaptation (5–50 reviews)
  → Every N new reviews, run k=5 gradient steps on that student's data
  → θ_student ← inner_loop(φ*, student_history, k=5)
  → Switch from φ* to θ_student for scheduling

Phase 3: Full personalization (50+ reviews)
  → Run k=20 gradient steps, larger learning rate
  → Optionally: continue gradient streaming (online learning)
```

### 4.2 Confidence / Uncertainty Estimation

Use **Monte Carlo Dropout** at inference time for uncertainty quantification:

```python
def predict_with_uncertainty(theta, x, n_samples=20):
    # Enable dropout at inference
    model.train()
    predictions = [model(x) for _ in range(n_samples)]
    mean_recall = torch.stack(predictions).mean(0)
    uncertainty = torch.stack(predictions).std(0)
    return mean_recall, uncertainty
```

High uncertainty → schedule the card sooner (conservative policy). This is especially valuable for new cards before the model has calibrated.

### 4.3 Scheduling Policy

Given the predicted recall probability R̂ and uncertainty σ, compute interval:

```python
def compute_interval(S_pred, desired_retention=0.90, sigma=0.0):
    # Base interval from predicted stability
    # Solve: desired_retention = (0.9^(1/S))^(t^w20) for t
    base_interval = S_pred  # when DR=0.9, I ≈ S

    # Uncertainty discount: review earlier when less confident
    confidence_factor = 1.0 - 0.5 * sigma  # [0.5, 1.0]
    
    # Difficulty boost: harder cards reviewed slightly more often
    difficulty_factor = 1.0 - 0.05 * (D - 5)  # center around D=5
    
    interval = base_interval * confidence_factor * difficulty_factor
    
    # Apply fuzzing (±5%) to prevent clustering
    interval *= uniform(0.95, 1.05)
    
    return max(1, round(interval))
```

---

## 5. Advanced Extensions

### 5.1 Transformer-Based Student Model (KAR³L-style)

For richer modeling, replace the GRU history encoder with a Transformer:

```
Input sequence: all of student's cross-card reviews (up to last 128)
Each token: [card_embed | grade_embed | time_embed | position_embed]

Architecture:
  - 2-layer Transformer encoder, 4 heads, dim=128
  - Output: query head attends to current card, produces recall prediction
  
Training: same Reptile outer loop, but inner loop is longer (k=15)
Adaptation cost: ~300ms on CPU for 20 steps — acceptable for background sync
```

This approach (inspired by KAR³L, EMNLP 2024) provides **cross-card interference modeling**: getting card A wrong affects predicted recall for semantically similar card B.

### 5.2 Bayesian Meta-Learning (PLATIPUS / MAML + Bayes)

Instead of a point estimate for φ, maintain a distribution p(θ | φ, Σ):

```
Meta-parameters: (φ, Σ) — mean and covariance of the parameter prior
Adaptation: posterior update given student's reviews via Laplace approximation
Scheduling: Thompson sampling from parameter posterior → natural exploration
```

This is the principled way to handle uncertainty and exploration simultaneously, though it adds complexity.

### 5.3 Reinforcement Learning Policy Head

Layer a policy network on top of the memory model using RL:

```
State: [S, D, R, card_embed, time_budget, deck_size, session_progress]
Action: which card to review next (or "end session")
Reward: Δ(long-term retention) measured at future checkpoints

Algorithm: PPO with card-model as the world model (model-based RL)
```

This enables the system to optimize for long-term learning goals, not just immediate recall accuracy — the key insight being that reviewing at R=70% (vs. R=90%) yields better stability gains.

### 5.4 Knowledge Concept Graph

Extend with a knowledge graph to model prerequisite relationships:

```
Cards → Knowledge Concepts (via embedding clustering or manual tagging)
Concepts → Dependency Graph (e.g., "multiplication" → prerequisite of "algebra")

Modified D update: If prerequisite concept has high D, boost current card's initial D
Modified scheduling: Prioritize prerequisite cards when mastery < threshold
```

### 5.5 Cross-Lingual / Cross-Domain Transfer

For language-learning apps: share meta-parameters across similar card types (vocabulary) using domain-conditioned initialization:

```python
# Domain-conditional Reptile
phi_vocab, phi_grammar, phi_kanji = separate_meta_params

# During meta-training, route each task to its domain head
# At inference, initialize from domain-appropriate φ
phi_init = domain_router(card_type) → phi_vocab or phi_grammar etc.
```

---

## 6. Data Requirements & Training Setup

### 6.1 Dataset Requirements

| Stage | Minimum | Recommended |
|---|---|---|
| FSRS-6 warm-start | 10K reviews, 100 users | 1M reviews, 10K users |
| Reptile meta-training | 50K reviews, 500 users | 5M reviews, 100K users |
| Per-user fast adaptation | 5 reviews | 20–50 reviews |

**Public datasets to bootstrap on**:
- **Duolingo SLAM** dataset (12.8M exercises, 6K students)
- **KAR³L dataset**: 123K study logs on trivia flashcards (GitHub: Pinafore/fact-repetition)
- **Anki review logs**: Large anonymous datasets shared by Anki community members

### 6.2 Data Schema

```sql
CREATE TABLE reviews (
    review_id     UUID PRIMARY KEY,
    user_id       UUID NOT NULL,
    card_id       UUID NOT NULL,
    reviewed_at   TIMESTAMP NOT NULL,
    elapsed_days  FLOAT NOT NULL,      -- days since last review (0 = first review)
    grade         SMALLINT NOT NULL,   -- 1=Again, 2=Hard, 3=Good, 4=Easy
    recalled      BOOLEAN NOT NULL,    -- grade >= 2
    session_idx   INT,                 -- which study session (for session-level features)
    INDEX(user_id, card_id, reviewed_at)
);

CREATE TABLE cards (
    card_id       UUID PRIMARY KEY,
    front_text    TEXT NOT NULL,
    back_text     TEXT NOT NULL,
    deck_id       UUID,
    card_embed    BYTEA,               -- pre-computed 64-dim float32 embedding
    domain        VARCHAR(64)          -- vocab, grammar, math, science, etc.
);
```

### 6.3 Training Infrastructure

```yaml
Hardware:
  Meta-training: 1x A100 40GB (or 4x V100 16GB)
  Training time: ~12 hours for 50K meta-iterations on 100K-user corpus
  Inference: CPU (50ms per schedule computation)
  Fast adaptation: CPU (~200ms for 10 gradient steps)

Hyperparameters:
  inner_lr: 0.01 (Adam)
  outer_lr: 0.001 → 0.0001 (cosine decay)
  inner_steps k: 5 (adaptation phase 1), 20 (adaptation phase 2)
  meta_batch_size: 16 tasks
  task_batch_size: 32 reviews
  epsilon: 0.1 → 0.01 (linear decay)
  hidden_dim: 128
  card_embed_dim: 64
  history_len: 32 (GRU) or 128 (Transformer)
```

---

## 7. Evaluation Framework

### 7.1 Metrics

| Metric | Description | Target |
|---|---|---|
| **AUC-ROC** | Recall prediction accuracy | > 0.80 (FSRS-6 baseline: ~0.78) |
| **Calibration Error** | |predicted R - actual R| | < 0.05 |
| **RMSE (stability)** | Error in stability estimation | < 2.0 days |
| **Cold-Start AUC** | AUC with only N reviews | AUC(N=5) > 0.73 |
| **Adaptation Speed** | Reviews to match FSRS-6 AUC | < 30 reviews |
| **Long-term Retention** | Student recall at 30/60/90 days | > 85% at 30 days |
| **Review Efficiency** | Learning per review event | > FSRS-6 baseline |

### 7.2 Ablation Studies

Run the following ablations to justify each component:

1. **No Reptile** (FSRS-6 population params only) — cold-start baseline
2. **Reptile only, no content embedding** — quantify semantic transfer value
3. **Reptile + content, no GRU history** — per-card vs. cross-card modeling
4. **Full model (Reptile + content + GRU)** — proposed system
5. **Reptile + Transformer** — cost/benefit of richer history encoder

### 7.3 Simulation Protocol

Since ground truth long-term retention is hard to collect, use a simulator:

```python
# Student simulator based on ACT-R / DSR model
class StudentSimulator:
    def __init__(self, true_params):
        self.D = {card: sample_difficulty() for card in cards}
        self.S = {card: 0.0 for card in cards}
    
    def review(self, card, elapsed_days):
        R = forgetting_curve(self.S[card], elapsed_days)
        recalled = random() < R  # Bernoulli
        self.S[card] = update_stability(self.D[card], self.S[card], R, recalled)
        return recalled
```

Run 1000 simulated students through both FSRS-6 and MetaSRS for 6 simulated months, compare retention rates and total reviews needed.

---

## 8. Implementation Roadmap

### Phase 1: Memory Model (Weeks 1–3)
- [ ] Implement DSR memory model as differentiable PyTorch nn.Module
- [ ] Verify it reproduces FSRS-6 behavior when given FSRS-6 parameters
- [ ] Add card embedding layer (SentenceTransformer → projected 64-dim)
- [ ] Add GRU history encoder

### Phase 2: Reptile Infrastructure (Weeks 4–6)
- [ ] Build task sampler from review database
- [ ] Implement inner loop (per-student SGD)
- [ ] Implement outer loop (Reptile meta-update)
- [ ] Set up evaluation pipeline (AUC, calibration)
- [ ] Warm-start from FSRS-6 population parameters

### Phase 3: Meta-Training Run (Weeks 7–9)
- [ ] Collect / license multi-user review dataset (Duolingo SLAM + KAR³L)
- [ ] Run full meta-training (12–24 hours on GPU)
- [ ] Hyperparameter search (Optuna or W&B sweeps)
- [ ] Ablation studies

### Phase 4: Fast Adaptation + Scheduling (Weeks 10–12)
- [ ] Implement online adaptation pipeline (background thread)
- [ ] Implement interval computation with uncertainty discounting
- [ ] A/B test: MetaSRS vs. FSRS-6 on real users (if available)
- [ ] Optional: RL policy head on top of memory model

### Phase 5: Production (Weeks 13–16)
- [ ] Optimize inference (ONNX export, quantization)
- [ ] Build deck-specific domain routing
- [ ] Ship Transformer variant as optional "deep mode"
- [ ] Continuous meta-training pipeline (retrain φ monthly on new user data)

---

## 9. Key Papers & References

| Paper | Relevance |
|---|---|
| **Reptile** (Nichol & Schulman, 2018) | Core meta-learning algorithm |
| **MAML** (Finn et al., 2017) | Theoretical grounding for Reptile |
| **FSRS-6** (Ye et al., open-spaced-repetition, 2024–2025) | SRS state of the art; DSR model |
| **KAR³L** (Shu et al., EMNLP 2024) | Content-aware SRS; DKT + BERT |
| **Half-Life Regression** (Settles & Meeder, ACL 2016) | ML-based forgetting curve; Duolingo |
| **DKT** (Piech et al., NeurIPS 2015) | Deep knowledge tracing with LSTM |
| **ACT-R** (Anderson et al.) | Cognitive model underlying spacing effect |
| **CPF** (arXiv 2404.12127) | Personalized forgetting with concept hierarchy |
| **PLATIPUS** (Finn et al., 2018) | Bayesian meta-learning (Bayesian extension of MAML/Reptile) |

---

## 10. Competitive Differentiation

| Feature | SM-2 (Anki default) | FSRS-6 | **MetaSRS (proposed)** |
|---|---|---|---|
| Memory model | Heuristic EF formula | DSR neural | Neural DSR + content |
| Personalization | Per-card EF factor | Full per-user params | Meta-initialized, few-shot |
| Cold start | Population avg | Population avg | **≤5 reviews to adapt** |
| Content-aware | No | No | **Yes (card embeddings)** |
| Cross-card transfer | No | No | **Yes (semantic similarity)** |
| Uncertainty-aware | No | No | **Yes (MC dropout)** |
| Training | None (rule-based) | Per-user gradient descent | **Multi-user Reptile** |

The meta-initialization is the killer feature: a student reviewing their **first 5 flash cards** already benefits from the forgetting patterns of thousands of prior students, while still adapting uniquely to their own retention profile within the first session.

---

*Document compiled from: FSRS-6 technical specification (expertium.github.io), OpenAI Reptile paper (arXiv:1803.02999), KAR³L (arXiv:2402.12291, EMNLP 2024), Duolingo HLR (Settles & Meeder 2016), open-spaced-repetition GitHub, and related cognitive science literature.*
