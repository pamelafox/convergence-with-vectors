"""Discrete word-combination operators over a closed vocabulary.

Every operator answers the same question -- "which vocabulary word(s) sit at
the intersection of words ``a`` and ``b``?" -- but combines their embeddings
differently. All embeddings here are L2-normalized, so a dot product *is* the
cosine similarity; the pure scoring functions below take that as given and
never call the network or a model, which keeps them fast and easy to test.

Interpretive note: these are *centroid-style* combinations (nearest neighbor
to some combined representation of ``a`` and ``b``). They are not the same
thing as Word2Vec-style relational analogy arithmetic (e.g. "king - man +
woman"), which relies on consistent linear relational offsets that MiniLM
sentence embeddings are not trained to preserve. Do not read convergence or
non-convergence here as evidence for or against analogy arithmetic.
"""

from __future__ import annotations

import string
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from embeddings import VocabEmbeddings, embed_texts

#: Signature of an embedding function: (model_name, texts) -> L2-normalized rows.
EmbedFn = Callable[[str, list[str]], np.ndarray]

#: Names of every operator, in the order they should be presented by default.
OPERATOR_NAMES: list[str] = ["centroid", "maximin", "product", "textual"]

#: Default phrase templates for the "textual" operator. ``{a}`` and ``{b}``
#: are substituted with the two input words.
DEFAULT_TEMPLATES: list[str] = [
    "{a} and {b}",
    "something associated with both {a} and {b}",
    "the concept connecting {a} and {b}",
]


@dataclass(frozen=True)
class CandidateScore:
    """One ranked candidate word produced by an operator."""

    rank: int
    candidate: str
    score: float
    sim_a: float
    sim_b: float


def normalize(vector: np.ndarray) -> np.ndarray:
    """L2-normalize a vector, leaving zero vectors untouched."""
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def maximin_scores(sim_a: np.ndarray, sim_b: np.ndarray) -> np.ndarray:
    """min(cos(w, a), cos(w, b)) for every candidate -- rewards being close to *both*."""
    return np.minimum(sim_a, sim_b)


def product_scores(sim_a: np.ndarray, sim_b: np.ndarray) -> np.ndarray:
    """Product of the two cosine similarities, with negatives clamped to 0.

    Multiplying two negatives would give a misleading positive score, so any
    negative cosine similarity is clamped to 0 *before* multiplying. A
    candidate that is dissimilar (negative cosine) to either input word scores
    exactly 0. Older models like MiniLM occasionally produce small negatives.
    """
    clipped_a = np.clip(sim_a, 0.0, None)
    clipped_b = np.clip(sim_b, 0.0, None)
    return clipped_a * clipped_b


#: Registry of "pairwise" operators: pure functions of (sim_a, sim_b) arrays.
#: Add an entry here to support a new pairwise-similarity operator without
#: touching any CLI or reporting code.
PAIRWISE_OPERATORS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "maximin": maximin_scores,
    "product": product_scores,
}


def rank_candidates(
    words: list[str],
    scores: np.ndarray,
    sim_a: np.ndarray,
    sim_b: np.ndarray,
    exclude: set[str] | None,
    top_k: int,
) -> list[CandidateScore]:
    """Sort candidates by score (descending) with deterministic alphabetical tie-breaking.

    Args:
        words: Vocabulary words, parallel to ``scores``/``sim_a``/``sim_b``.
        scores: The operator's combined score for each vocabulary word.
        sim_a: Cosine similarity of each vocabulary word to input word ``a``.
        sim_b: Cosine similarity of each vocabulary word to input word ``b``.
        exclude: Words to drop from the results (e.g. the two input words).
        top_k: Maximum number of candidates to return.
    """
    exclude = exclude or set()
    # Round scores before comparing so floating-point noise never overrides
    # the alphabetical tie-break for candidates that are "really" tied.
    order = sorted(range(len(words)), key=lambda i: (-round(float(scores[i]), 9), words[i]))
    kept = [i for i in order if words[i] not in exclude][:top_k]
    return [
        CandidateScore(
            rank=rank,
            candidate=words[i],
            score=float(scores[i]),
            sim_a=float(sim_a[i]),
            sim_b=float(sim_b[i]),
        )
        for rank, i in enumerate(kept, start=1)
    ]


def render_template(template: str, word_a: str, word_b: str) -> str:
    """Fill a "textual" phrase template, allowing only bare ``{a}`` and ``{b}`` placeholders.

    Rejecting anything else (``{a.__class__}``, ``{0}``, format specs) keeps
    user-supplied templates from reaching into Python objects via ``str.format``.
    """
    for _, field, spec, conversion in string.Formatter().parse(template):
        if field is None:
            continue
        if field not in ("a", "b") or spec or conversion:
            raise ValueError("Template may only contain the placeholders {a} and {b}")
    return template.format(a=word_a, b=word_b)


def apply_operator(
    operator_name: str,
    model_name: str,
    vocab: VocabEmbeddings,
    word_a: str,
    word_b: str,
    *,
    top_k: int = 10,
    exclude: set[str] | None = None,
    template: str = "{a} and {b}",
    embed: EmbedFn | None = None,
) -> list[CandidateScore]:
    """Embed ``word_a``/``word_b`` (and, for "textual", a phrase) and rank the vocabulary.

    Args:
        operator_name: One of :data:`OPERATOR_NAMES`.
        model_name: Short or full sentence-transformers model name.
        vocab: Pre-embedded candidate vocabulary for this model.
        word_a: First input word.
        word_b: Second input word.
        top_k: Number of ranked candidates to return.
        exclude: Words to exclude from results (defaults to none here; callers
            typically pass ``{word_a, word_b}`` unless ``--include-inputs`` is set).
        template: Phrase template used only by the "textual" operator.
        embed: Function used to embed the inputs (and phrase); defaults to running the model directly.
    """
    if operator_name not in OPERATOR_NAMES:
        raise ValueError(f"Unknown operator {operator_name!r}; choose from {OPERATOR_NAMES}")
    embed = embed or embed_texts

    vec_a, vec_b = embed(model_name, [word_a, word_b])
    sim_a = vocab.similarities_to(vec_a)
    sim_b = vocab.similarities_to(vec_b)

    if operator_name == "centroid":
        query = normalize(vec_a + vec_b)
        scores = vocab.similarities_to(query)
    elif operator_name == "textual":
        phrase = render_template(template, word_a, word_b)
        phrase_vec = embed(model_name, [phrase])[0]
        scores = vocab.similarities_to(phrase_vec)
    else:
        scores = PAIRWISE_OPERATORS[operator_name](sim_a, sim_b)

    return rank_candidates(vocab.words, scores, sim_a, sim_b, exclude, top_k)
