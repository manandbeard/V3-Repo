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
