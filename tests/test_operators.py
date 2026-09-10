"""Tests for the scoring math and ranking behavior of the combining operators.

These tests exercise the pure numpy scoring functions and ``rank_candidates``
directly, and exercise ``apply_operator`` with a monkeypatched ``embed_texts``
so the operator wiring is covered without ever downloading a real model.
"""

from __future__ import annotations

import numpy as np
import pytest

import operators as ops
from embeddings import VocabEmbeddings


def test_balanced_scores_is_elementwise_min():
    sim_a = np.array([0.9, 0.1, 0.5])
    sim_b = np.array([0.2, 0.8, 0.5])
    np.testing.assert_allclose(ops.balanced_scores(sim_a, sim_b), [0.2, 0.1, 0.5])


def test_mean_scores_is_elementwise_average():
    sim_a = np.array([1.0, 0.0])
    sim_b = np.array([0.0, 1.0])
    np.testing.assert_allclose(ops.mean_scores(sim_a, sim_b), [0.5, 0.5])


def test_geometric_mean_scores_matches_sqrt_product_for_positive_inputs():
    sim_a = np.array([0.64, 0.25])
    sim_b = np.array([0.25, 0.64])
    np.testing.assert_allclose(ops.geometric_mean_scores(sim_a, sim_b), [0.4, 0.4])


def test_geometric_mean_scores_clamps_negative_similarities_to_zero():
    # A candidate dissimilar to 'a' (negative cosine) should score 0, not a
    # spurious positive value from multiplying two negatives together.
    sim_a = np.array([-0.5, -0.5, 0.5])
    sim_b = np.array([0.8, -0.5, -0.5])
    scores = ops.geometric_mean_scores(sim_a, sim_b)
    np.testing.assert_allclose(scores, [0.0, 0.0, 0.0])


def test_normalize_handles_zero_vector():
    zero = np.zeros(4)
    np.testing.assert_allclose(ops.normalize(zero), zero)


def test_normalize_produces_unit_vector():
    vector = np.array([3.0, 4.0])
    normalized = ops.normalize(vector)
    assert np.isclose(np.linalg.norm(normalized), 1.0)


def test_rank_candidates_orders_by_score_descending():
    words = ["alpha", "beta", "gamma"]
    scores = np.array([0.1, 0.9, 0.5])
    result = ops.rank_candidates(words, scores, scores, scores, exclude=None, top_k=3)
    assert [c.candidate for c in result] == ["beta", "gamma", "alpha"]
    assert [c.rank for c in result] == [1, 2, 3]


def test_rank_candidates_breaks_ties_alphabetically():
    words = ["zebra", "apple", "mango"]
    scores = np.array([0.5, 0.5, 0.5])
    result = ops.rank_candidates(words, scores, scores, scores, exclude=None, top_k=3)
    assert [c.candidate for c in result] == ["apple", "mango", "zebra"]


def test_rank_candidates_excludes_given_words():
    words = ["alpha", "beta", "gamma"]
    scores = np.array([0.9, 0.5, 0.1])
    result = ops.rank_candidates(words, scores, scores, scores, exclude={"alpha"}, top_k=3)
    assert [c.candidate for c in result] == ["beta", "gamma"]


def test_rank_candidates_respects_top_k():
    words = ["a", "b", "c", "d"]
    scores = np.array([0.4, 0.3, 0.2, 0.1])
    result = ops.rank_candidates(words, scores, scores, scores, exclude=None, top_k=2)
    assert [c.candidate for c in result] == ["a", "b"]


def test_rank_candidates_reports_sim_a_and_sim_b_independently_of_score():
    words = ["alpha", "beta"]
    scores = np.array([1.0, 0.0])
    sim_a = np.array([0.9, 0.1])
    sim_b = np.array([0.2, 0.8])
    result = ops.rank_candidates(words, scores, sim_a, sim_b, exclude=None, top_k=2)
    winner = result[0]
    assert winner.candidate == "alpha"
    assert winner.sim_a == pytest.approx(0.9)
    assert winner.sim_b == pytest.approx(0.2)


def _make_vocab(words: list[str], vectors: np.ndarray) -> VocabEmbeddings:
    return VocabEmbeddings(words=words, vectors=vectors, index={w: i for i, w in enumerate(words)})


def test_apply_operator_centroid_prefers_the_midpoint_word(monkeypatch):
    # Three 2D unit vectors: 'a' points along x, 'b' along y, 'mid' at 45 degrees.
    vec_a = np.array([1.0, 0.0])
    vec_b = np.array([0.0, 1.0])
    vec_mid = ops.normalize(np.array([1.0, 1.0]))
    vec_far = np.array([-1.0, 0.0])
    words = ["mid", "far", "a_like", "b_like"]
    vectors = np.stack([vec_mid, vec_far, vec_a, vec_b])
    vocab = _make_vocab(words, vectors)

    def fake_embed_texts(model_name, texts):
        table = {"a": vec_a, "b": vec_b}
        return np.stack([table[t] for t in texts])

    monkeypatch.setattr(ops, "embed_texts", fake_embed_texts)

    result = ops.apply_operator("centroid", "fake-model", vocab, "a", "b", top_k=4, exclude=set())
    assert result[0].candidate == "mid"


def test_apply_operator_unknown_name_raises():
    vocab = _make_vocab(["a"], np.array([[1.0]]))
    with pytest.raises(ValueError, match="Unknown operator"):
        ops.apply_operator("nonsense", "fake-model", vocab, "a", "b")


def test_apply_operator_textual_uses_formatted_template(monkeypatch):
    vec_a = np.array([1.0, 0.0])
    vec_b = np.array([0.0, 1.0])
    vec_phrase = ops.normalize(np.array([1.0, 1.0]))
    words = ["phrase_word", "other"]
    vectors = np.stack([vec_phrase, -vec_phrase])
    vocab = _make_vocab(words, vectors)

    seen_texts: list[str] = []

    def fake_embed_texts(model_name, texts):
        seen_texts.extend(texts)
        table = {"a": vec_a, "b": vec_b, "a and b": vec_phrase}
        return np.stack([table[t] for t in texts])

    monkeypatch.setattr(ops, "embed_texts", fake_embed_texts)

    result = ops.apply_operator("textual", "fake-model", vocab, "a", "b", top_k=2, exclude=set(), template="{a} and {b}")
    assert "a and b" in seen_texts
    assert result[0].candidate == "phrase_word"
