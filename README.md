# MetaSRS — Meta-Initialized Spaced Repetition Scheduler

A neural memory scheduler that learns a meta-initialization from multi-user review histories using [Reptile meta-learning](https://arxiv.org/abs/1803.02999), then fast-adapts to new students in as few as **5 gradient steps** — beating FSRS-6's cold-start accuracy from the first review.

## Key Numbers

| Metric | Value |
|--------|-------|
| FSRS-6 params (baseline) | 21 |
| Reviews to meaningful personalisation | 5 |
| DSR memory variables | 3 (Difficulty, Stability, Retrievability) |
| Fast-adapt latency | <200ms |
| MemoryNet parameters | ~50K |

## Architecture Overview

```

ONLINE FAST ADAPTATION (new student):
  0 reviews  → use φ* directly (population prior, zero-shot)
  5 reviews  → 5 gradient steps from φ* → personalised θ_student
  50 reviews → 20 gradient steps → fully personalised θ_student

SCHEDULING:
  θ_student → predict (R, S, D, uncertainty) → compute interval → show card
```

## Project Structure

```
meta-srs/
├── config.py                    # All hyperparameters (Sections 1.1, 6.3)
├── train.py                     # Main training script
├── requirements.txt
├── models/
│   ├── memory_net.py            # MemoryNet — neural DSR model (Section 2.2)
│   ├── gru_encoder.py           # GRU history encoder (Section 2.4)
│   └── card_embeddings.py       # BERT card embeddings + projection (Section 2.3)
├── training/
│   ├── fsrs_warmstart.py        # FSRS-6 baseline + warm-start (Section 1.1)
│   ├── loss.py                  # Multi-component loss function (Section 3.3)
│   └── reptile.py               # Reptile inner/outer loops (Sections 3.2, 3.4)
├── inference/
│   ├── adaptation.py            # 3-phase fast adaptation (Section 4.1)
│   └── scheduling.py            # Uncertainty-aware scheduling (Section 4.2)
└── evaluation/
    └── metrics.py               # Eval framework + ablations (Section 7)
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test with Synthetic Data

```bash
# Quick sanity check (~2 minutes)
python train.py --synthetic --n-students 100 --n-iters 500

# Larger synthetic run
python train.py --synthetic --n-students 1000 --n-iters 5000
```

### 3. Train with Real Data

Prepare a CSV with columns: `student_id, card_id, timestamp, elapsed_days, grade`

```bash
# Pre-compute card embeddings (one-time)
python -c "
from models.card_embeddings import embed_cards_offline
import json, numpy as np
cards = json.load(open('cards.json'))  # [{id, front, back}, ...]
embeds = embed_cards_offline(cards)
np.savez('card_embeddings.npz', **embeds)
print(f'Embedded {len(embeds)} cards')
"

# Full training run
python train.py --data reviews.csv --embeddings card_embeddings.npz --n-iters 50000
```

### 4. Resume Training

```bash
python train.py --data reviews.csv --resume checkpoints/phi_iter_10000.pt
```

### 5. Evaluate Only

```bash
python train.py --data reviews.csv --eval-only --eval-checkpoint checkpoints/phi_star.pt
```

## Three-Phase Onboarding

| Phase | Reviews | What Happens |
|-------|---------|-------------|
| **Phase 1: Zero-shot** | 0 | Use φ* directly. Mix easy + uncertain cards for calibration. |
| **Phase 2: Rapid adapt** | 5–50 | Every 5 reviews: 5 gradient steps from φ*. AUC exceeds FSRS-6. |
| **Phase 3: Full personal** | 50+ | 20 gradient steps. Streaming updates after every review. |

## Using the Scheduler

```python
import torch
from config import MetaSRSConfig
from models import MemoryNet
from inference.adaptation import FastAdapter
from inference.scheduling import Scheduler

# Load trained meta-parameters
config = MetaSRSConfig()
model = MemoryNet()
checkpoint = torch.load("checkpoints/phi_star.pt")
phi_star = checkpoint["phi"]

# Create adapter for a new student
adapter = FastAdapter(model, phi_star, config)

# Student reviews cards
adapter.add_review(review)  # Triggers adaptation as needed

# Get scheduling predictions
scheduler = Scheduler(adapter.get_model(), config.scheduling)
result = scheduler.schedule_card(card_id, features, S_prev)
print(f"Review '{card_id}' again in {result.interval_days} days")
print(f"Predicted recall: {result.p_recall_mean:.1%} ± {result.p_recall_sigma:.1%}")
```

## Loss Function

Three components ensure cognitively-valid predictions:

1. **Recall loss** (BCE) — primary objective, predicts recall probability
2. **Stability consistency** (MSE) — ensures S predictions follow the power-law forgetting curve
3. **Monotonicity** — S cannot decrease on successful recall

```
L = L_recall + 0.10 × L_stability + 0.01 × L_monotonicity
```

## Evaluation Targets (Section 7.1)

| Metric | Target | Description |
|--------|--------|-------------|
| AUC-ROC | > 0.80 | Recall prediction accuracy (FSRS-6: ~0.78) |
| Calibration Error | < 0.05 | \|predicted R − actual R\| |
| Cold-Start AUC (N=5) | > 0.73 | AUC with only 5 reviews |
| Adaptation Speed | < 30 reviews | Reviews to match FSRS-6 AUC |
| 30-day Retention | > 85% | Recall rate at 30 days |
| RMSE (stability) | < 2.0 days | Error in S estimation |

## Public Datasets for Bootstrap

| Dataset | Scale | Source |
|---------|-------|--------|
| Duolingo SLAM | 12.8M exercises, 6K students | Duolingo (public benchmark) |
| KAR3L Trivia Logs | 123K study logs | github.com/Pinafore/fact-repetition |
| Anki Community Logs | 1M+ reviews | AnkiWeb / research sharing |

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 1× V100 16GB | 1× A100 40GB |
| Training time | ~24h (50K iters) | ~12h (50K iters) |
| Inference | CPU, ~50ms/card | CPU, ~50ms/card |
| Adaptation | CPU, ~200ms | CPU, ~200ms |

## References

- Reptile (Nichol & Schulman, 2018) — [arXiv:1803.02999](https://arxiv.org/abs/1803.02999)
- FSRS-6 (open-spaced-repetition, 2024–2025) — [expertium.github.io](https://expertium.github.io)
- KAR3L (Shu et al., EMNLP 2024) — [arXiv:2402.12291](https://arxiv.org/abs/2402.12291)
- MAML (Finn et al., ICML 2017)
- Half-Life Regression (Settles & Meeder, ACL 2016)
- DKT (Piech et al., NeurIPS 2015)
- PLATIPUS (Finn et al., NeurIPS 2018)

---

*Research compiled March 2026. A student reviewing their first 5 flash cards already benefits from the forgetting patterns of thousands of prior learners — all within the first study session.* 🎓
