"""Embedding model registry and cached vocabulary embeddings.

Models come from two providers:

* ``sentence-transformers`` models (MiniLM, BGE, Harrier, mxbai, Qwen3) run in-process on CPU,
  so they work anywhere, including the deployed container.
* ``ollama`` models are embedded by a local Ollama server through its
  OpenAI-compatible API, so they're only available where Ollama is running.

Vocabulary embeddings are cached to disk so repeated runs don't re-embed the
whole candidate list every time, since that's the most expensive step for a
large vocabulary.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

import numpy as np
import openai
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class EmbeddingModel:
    """How to run one embedding model."""

    model_id: str  # Hugging Face id or Ollama model name
    label: str  # Display name for the web pages
    provider: Literal["sentence-transformers", "ollama"] = "sentence-transformers"


#: Short, CLI-friendly names for every model.
#: Add new models here to make them available to every script, CLI flag, and the web app.
MODELS: dict[str, EmbeddingModel] = {
    "minilm-l6": EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2", "MiniLM L6"),
    "minilm-l12": EmbeddingModel("sentence-transformers/all-MiniLM-L12-v2", "MiniLM L12"),
    "bge-small": EmbeddingModel("BAAI/bge-small-en-v1.5", "BGE Small"),
    "harrier-270m": EmbeddingModel("microsoft/harrier-oss-v1-270m", "Harrier 270M"),
    "mxbai-embed-large": EmbeddingModel("mixedbread-ai/mxbai-embed-large-v1", "mxbai Embed Large"),
    "qwen3-embedding-0.6b": EmbeddingModel("Qwen/Qwen3-Embedding-0.6B", "Qwen3 Embedding 0.6B"),
}

#: OpenAI-compatible endpoint of the Ollama server used for "ollama" models.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")

#: Models the command-line scripts use by default: small, in-process, and always available.
DEFAULT_MODELS: list[str] = ["minilm-l6", "minilm-l12"]

#: Default on-disk location for cached vocabulary embeddings.
DEFAULT_CACHE_DIR = Path(".embedding_cache")


class ModelUnavailableError(RuntimeError):
    """Raised when a model's provider can't be reached (e.g. Ollama isn't running)."""


# functools.cache isn't thread-safe, and the backend embeds from worker threads
_load_lock = threading.Lock()


@cache
def _load_model(model_name: str) -> SentenceTransformer:
    hf_id = MODELS[model_name].model_id if model_name in MODELS else model_name
    return SentenceTransformer(hf_id, device="cpu")


def load_model(model_name: str) -> SentenceTransformer:
    """Load (and memoize in-process) a sentence-transformers model.

    Args:
        model_name: Either a short name from :data:`MODELS` or a full
            Hugging Face model id.
    """
    with _load_lock:
        return _load_model(model_name)


def _ollama_client() -> openai.OpenAI:
    return openai.OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama", timeout=30, max_retries=0)


def _embed_with_ollama(model_id: str, texts: list[str]) -> np.ndarray:
    try:
        response = _ollama_client().embeddings.create(model=model_id, input=texts)
    except (openai.APIConnectionError, openai.NotFoundError) as err:
        raise ModelUnavailableError(f"Ollama model {model_id!r} isn't available at {OLLAMA_BASE_URL}") from err
    vectors = np.array([item.embedding for item in response.data], dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def is_available(model_name: str) -> bool:
    """Whether ``model_name`` can embed right now (always true for in-process models)."""
    info = MODELS.get(model_name)
    if info is None or info.provider != "ollama":
        return True
    try:
        _embed_with_ollama(info.model_id, ["test"])
    except ModelUnavailableError:
        return False
    return True


def embed_texts(model_name: str, texts: list[str]) -> np.ndarray:
    """Embed a list of texts, returning L2-normalized rows.

    Normalizing means the dot product of any two rows equals their cosine
    similarity, so downstream code can use matrix multiplication instead of
    repeated norm computations.
    """
    info = MODELS.get(model_name)
    if info is not None and info.provider == "ollama":
        return _embed_with_ollama(info.model_id, texts)
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
            with np.load(npz_path) as data:
                vectors = data["vectors"]
            return VocabEmbeddings(words=words, vectors=vectors, index={w: i for i, w in enumerate(words)})

    vectors = embed_texts(model_name, words)
    np.savez_compressed(npz_path, vectors=vectors)
    meta_path.write_text(json.dumps({"model": model_name, "words": words}), encoding="utf-8")
    return VocabEmbeddings(words=words, vectors=vectors, index={w: i for i, w in enumerate(words)})
