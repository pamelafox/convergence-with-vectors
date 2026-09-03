"""Embedding model registry and cached vocabulary embeddings.

Models are loaded with :mod:`sentence_transformers` on CPU (these MiniLM
models are small enough that no GPU is needed). Vocabulary embeddings are
cached to disk so repeated runs don't re-embed the whole candidate list
every time, since that's the most expensive step for a large vocabulary.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

#: Short, CLI-friendly names mapped to their Hugging Face model ids.
#: Add new models here to make them available to every script and CLI flag.
MODELS: dict[str, str] = {
    "minilm-l6": "sentence-transformers/all-MiniLM-L6-v2",
    "minilm-l12": "sentence-transformers/all-MiniLM-L12-v2",
}

#: Default on-disk location for cached vocabulary embeddings.
DEFAULT_CACHE_DIR = Path(".embedding_cache")


@cache
def load_model(model_name: str) -> SentenceTransformer:
    """Load (and memoize in-process) a sentence-transformers model.

    Args:
        model_name: Either a short name from :data:`MODELS` or a full
            Hugging Face model id.
    """
    hf_id = MODELS.get(model_name, model_name)
    return SentenceTransformer(hf_id, device="cpu")


def embed_texts(model_name: str, texts: list[str]) -> np.ndarray:
    """Embed a list of texts, returning L2-normalized rows.

    Normalizing means the dot product of any two rows equals their cosine
    similarity, so downstream code can use matrix multiplication instead of
    repeated norm computations.
    """
    model = load_model(model_name)
    embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(embeddings, dtype=np.float32)


@dataclass
class VocabEmbeddings:
    """Normalized embeddings for every word in a vocabulary, for one model."""

    words: list[str]
    vectors: np.ndarray  # shape (len(words), dim), L2-normalized rows
    index: dict[str, int]

    def similarities_to(self, query: np.ndarray) -> np.ndarray:
        """Cosine similarity of every vocabulary word to a normalized query vector."""
        return self.vectors @ query

    def vector_for(self, word: str) -> np.ndarray | None:
        """Return the cached vector for ``word`` if it is in the vocabulary."""
        idx = self.index.get(word)
        return None if idx is None else self.vectors[idx]


def _cache_key(model_name: str, words: list[str]) -> str:
    digest = hashlib.sha256("\n".join(words).encode("utf-8")).hexdigest()[:16]
    safe_model = model_name.replace("/", "__")
    return f"{safe_model}__{digest}"


def get_vocab_embeddings(
    model_name: str,
    words: list[str],
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
) -> VocabEmbeddings:
    """Return normalized embeddings for ``words``, using a disk cache.

    The cache key combines the model name and a hash of the vocabulary
    contents, so any change to either produces a fresh cache entry instead
    of silently reusing stale embeddings.

    Args:
        model_name: Short or full model name (see :data:`MODELS`).
        words: The vocabulary, in the order returned by :func:`vocab.load_vocab`.
        cache_dir: Directory used to store cached ``.npz``/``.json`` files.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    key = _cache_key(model_name, words)
    npz_path = cache_path / f"{key}.npz"
    meta_path = cache_path / f"{key}.json"

    if npz_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("words") == words and meta.get("model") == model_name:
            vectors = np.load(npz_path)["vectors"]
            return VocabEmbeddings(words=words, vectors=vectors, index={w: i for i, w in enumerate(words)})

    vectors = embed_texts(model_name, words)
    np.savez_compressed(npz_path, vectors=vectors)
    meta_path.write_text(json.dumps({"model": model_name, "words": words}), encoding="utf-8")
    return VocabEmbeddings(words=words, vectors=vectors, index={w: i for i, w in enumerate(words)})
