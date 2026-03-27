"""
Content-Aware Card Embeddings (Section 2.3).

Following KAR3L (Shu et al., EMNLP 2024), each card's text is embedded via
a sentence transformer to enable semantic transfer across cards.

If a student has reviewed 'France → Paris', the model infers partial knowledge
of 'Germany → Berlin' because both cluster in embedding space — without a
single direct review.

Pipeline:
    1. Offline: embed all cards once with SentenceTransformer('all-MiniLM-L6-v2')
       → 384-dim vectors.
    2. Online: a trainable Linear(384, 64) projection (part of phi, updated
       during meta-training) produces the 64-dim card_embed feature.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


class CardEmbeddingProjector(nn.Module):
    """
    Trainable projection from raw BERT embeddings to compact card features.
    This layer is part of the meta-parameters phi and gets updated during
    Reptile training.
    """

    def __init__(self, raw_dim: int = 384, embed_dim: int = 64):
        super().__init__()
        self.projection = nn.Linear(raw_dim, embed_dim)

    def forward(self, raw_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            raw_embeddings: (batch, 384) — pre-computed BERT embeddings.

        Returns:
            card_embed: (batch, 64) — L2-normalised projected embeddings.
        """
        projected = self.projection(raw_embeddings)
        return F.normalize(projected, p=2, dim=-1)


def embed_cards_offline(
    cards: List[Dict[str, str]],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
    device: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute BERT embeddings for all cards offline (run once).

    Args:
        cards: List of dicts with 'id', 'front', 'back' keys.
        model_name: SentenceTransformer model name.
        batch_size: Encoding batch size.
        device: Device for encoding (None = auto).

    Returns:
        Dict mapping card_id → ndarray(384,)
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for offline embedding. "
            "Install with: pip install sentence-transformers"
        )

    model = SentenceTransformer(model_name, device=device)

    texts = [f"{card['front']} | {card['back']}" for card in cards]
    ids = [card["id"] for card in cards]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,  # We L2-normalise in the projection layer
    )

    return {card_id: emb for card_id, emb in zip(ids, embeddings)}


class CardEmbeddingStore:
    """
    In-memory lookup for pre-computed card embeddings.
    Used during training and inference to retrieve embeddings by card_id.
    """

    def __init__(self, embeddings: Dict[str, np.ndarray]):
        self.embeddings = embeddings
        self._dim = next(iter(embeddings.values())).shape[0] if embeddings else 384

    def lookup(
        self, card_ids: List[str], device: torch.device
    ) -> torch.Tensor:
        """
        Retrieve embeddings for a batch of card IDs.

        Args:
            card_ids: List of card ID strings.
            device: Target torch device.

        Returns:
            Tensor of shape (len(card_ids), raw_dim)
        """
        vecs = []
        for cid in card_ids:
            if cid in self.embeddings:
                vecs.append(self.embeddings[cid])
            else:
                # Unknown card: zero vector (will project to zero)
                vecs.append(np.zeros(self._dim, dtype=np.float32))
        return torch.tensor(np.stack(vecs), dtype=torch.float32, device=device)

    def __len__(self) -> int:
        return len(self.embeddings)

    @classmethod
    def from_file(cls, path: str) -> "CardEmbeddingStore":
        """Load embeddings from a .npz file."""
        data = np.load(path, allow_pickle=True)
        embeddings = {str(k): data[k] for k in data.files}
        return cls(embeddings)

    def save(self, path: str):
        """Save embeddings to a .npz file."""
        np.savez(path, **self.embeddings)
