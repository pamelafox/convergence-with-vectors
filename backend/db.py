"""SQLite storage for embeddings.

A single table holds both the candidate vocabulary (``in_vocab = 1``) and
anything embedded on demand, like off-vocabulary input words or "textual"
phrases (``in_vocab = 0``). Vectors are stored as float32 bytes and are
already L2-normalized, so dot products are cosine similarities.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import numpy as np

from embeddings import VocabEmbeddings
from operators import EmbedFn

SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    model    TEXT NOT NULL,
    text     TEXT NOT NULL,
    in_vocab INTEGER NOT NULL DEFAULT 0,
    vector   BLOB NOT NULL,
    PRIMARY KEY (model, text)
)
"""


def connect(path: Path | str) -> sqlite3.Connection:
    """Open (creating if needed) the embeddings database."""
    # Shared across FastAPI's worker threads; EmbeddingStore serializes access with a lock.
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute(SCHEMA)
    return conn


def _to_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def _from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def store_vocab(conn: sqlite3.Connection, model: str, words: list[str], vectors: np.ndarray) -> None:
    """Make ``words`` the candidate vocabulary for ``model``, replacing any previous vocabulary."""
    with conn:
        conn.execute("UPDATE embeddings SET in_vocab = 0 WHERE model = ?", (model,))
        conn.executemany(
            "INSERT INTO embeddings (model, text, in_vocab, vector) VALUES (?, ?, 1, ?) " "ON CONFLICT (model, text) DO UPDATE SET in_vocab = 1, vector = excluded.vector",
            [(model, word, _to_blob(vector)) for word, vector in zip(words, vectors, strict=True)],
        )


def vocab_models(conn: sqlite3.Connection) -> list[str]:
    """Models that have a candidate vocabulary in the database."""
    return [row[0] for row in conn.execute("SELECT DISTINCT model FROM embeddings WHERE in_vocab = 1 ORDER BY model")]


def load_vocab(conn: sqlite3.Connection, model: str) -> VocabEmbeddings:
    """Load ``model``'s candidate vocabulary into memory as a numpy matrix."""
    rows = conn.execute("SELECT text, vector FROM embeddings WHERE model = ? AND in_vocab = 1 ORDER BY text", (model,)).fetchall()
    words = [text for text, _ in rows]
    vectors = np.stack([_from_blob(blob) for _, blob in rows])
    return VocabEmbeddings(words=words, vectors=vectors, index={w: i for i, w in enumerate(words)})


class EmbeddingStore:
    """Looks up embeddings in SQLite, computing and caching any that are missing."""

    def __init__(self, conn: sqlite3.Connection, embed_fn: EmbedFn):
        self._conn = conn
        self._embed_fn = embed_fn
        self._lock = threading.Lock()

    def embed(self, model: str, texts: list[str]) -> np.ndarray:
        """Return one normalized row per text, in order. Matches :data:`operators.EmbedFn`."""
        unique = list(dict.fromkeys(texts))
        placeholders = ",".join("?" * len(unique))
        with self._lock:
            rows = self._conn.execute(f"SELECT text, vector FROM embeddings WHERE model = ? AND text IN ({placeholders})", (model, *unique)).fetchall()
        found = {text: _from_blob(blob) for text, blob in rows}

        missing = [text for text in unique if text not in found]
        if missing:
            # Run the model outside the lock so slow embeddings don't block cache reads.
            vectors = self._embed_fn(model, missing)
            with self._lock, self._conn:
                self._conn.executemany(
                    "INSERT OR IGNORE INTO embeddings (model, text, in_vocab, vector) VALUES (?, ?, 0, ?)",
                    [(model, text, _to_blob(vector)) for text, vector in zip(missing, vectors, strict=True)],
                )
            found.update(zip(missing, np.asarray(vectors, dtype=np.float32), strict=True))

        return np.stack([found[text] for text in texts])
