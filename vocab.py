"""Vocabulary loading utilities for the word-combination experiments.

The vocabulary is the closed candidate space: every combining operator must
return a word from this list, so parsing it correctly and deterministically
matters as much as the similarity math.
"""

from __future__ import annotations

from pathlib import Path


def load_vocab(path: str | Path) -> list[str]:
    """Load a newline-delimited UTF-8 vocabulary file.

    Each non-blank line is treated as one candidate word. Leading/trailing
    whitespace is stripped, blank lines and ``#``-prefixed comment lines are
    skipped, and duplicate words are removed while preserving first-seen
    order. Matching is case-sensitive, so "Fire" and "fire" are kept as
    distinct candidates if both appear.

    Args:
        path: Path to the vocabulary file.

    Returns:
        The ordered list of unique vocabulary words.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file contains no usable words.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Vocabulary file not found: {file_path}")

    seen: set[str] = set()
    words: list[str] = []
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        word = raw_line.strip()
        if not word or word.startswith("#"):
            continue
        if word not in seen:
            seen.add(word)
            words.append(word)

    if not words:
        raise ValueError(f"Vocabulary file has no usable words: {file_path}")

    return words
