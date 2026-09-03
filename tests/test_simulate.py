"""Tests for the convergence simulation loop (convergence, loop, and round-limit outcomes).

``apply_operator`` and ``get_vocab_embeddings`` are monkeypatched so the
control flow can be tested deterministically without downloading a real
model or vocabulary.
"""

from __future__ import annotations

from pathlib import Path

import simulate_convergence as sim
from operators import CandidateScore


def _fake_result(candidate: str) -> list[CandidateScore]:
    return [CandidateScore(rank=1, candidate=candidate, score=1.0, sim_a=1.0, sim_b=1.0)]


def _patch_vocab(monkeypatch):
    # get_vocab_embeddings is only used to build the (unused, by our fake
    # apply_operator) VocabEmbeddings objects, so a trivial stub is enough.
    monkeypatch.setattr(sim, "get_vocab_embeddings", lambda model_name, words, cache_dir: object())
    monkeypatch.setattr(sim, "load_vocab", lambda path: ["placeholder"])


def test_simulate_converges_when_models_agree(monkeypatch):
    _patch_vocab(monkeypatch)

    def fake_apply_operator(operator_name, model_name, vocab, word_a, word_b, *, top_k, exclude, template="{a} and {b}"):
        return _fake_result("mountain")

    monkeypatch.setattr(sim, "apply_operator", fake_apply_operator)

    path, outcome, rows = sim.simulate_convergence(Path("unused.txt"), "fire", "whale", models=["m1", "m2"], max_rounds=5)

    assert outcome == "converged"
    assert path[-1]["pair_out"] == ("mountain", "mountain")
    assert all(row.outcome == "converged" for row in rows)


def test_simulate_detects_loop(monkeypatch):
    _patch_vocab(monkeypatch)

    # Round 1: (fire, whale) -> (a, b). Round 2: (a, b) -> (fire, whale) again -> loop.
    responses = {
        ("m1", ("fire", "whale")): "a",
        ("m2", ("fire", "whale")): "b",
        ("m1", ("a", "b")): "fire",
        ("m2", ("a", "b")): "whale",
    }

    def fake_apply_operator(operator_name, model_name, vocab, word_a, word_b, *, top_k, exclude, template="{a} and {b}"):
        return _fake_result(responses[(model_name, (word_a, word_b))])

    monkeypatch.setattr(sim, "apply_operator", fake_apply_operator)

    path, outcome, rows = sim.simulate_convergence(Path("unused.txt"), "fire", "whale", models=["m1", "m2"], max_rounds=5)

    assert outcome == "loop"
    assert path[-1]["pair_out"] == ("fire", "whale")


def test_simulate_hits_round_limit(monkeypatch):
    _patch_vocab(monkeypatch)

    counter = {"n": 0}

    def fake_apply_operator(operator_name, model_name, vocab, word_a, word_b, *, top_k, exclude, template="{a} and {b}"):
        counter["n"] += 1
        # Always invent a brand-new word so the pair never repeats and never matches.
        suffix = "a" if model_name == "m1" else "b"
        return _fake_result(f"word{counter['n']}{suffix}")

    monkeypatch.setattr(sim, "apply_operator", fake_apply_operator)

    path, outcome, rows = sim.simulate_convergence(Path("unused.txt"), "fire", "whale", models=["m1", "m2"], max_rounds=3)

    assert outcome == "round_limit"
    assert path[-1]["round"] == 3


def test_simulate_excludes_current_pair_by_default(monkeypatch):
    _patch_vocab(monkeypatch)
    seen_excludes = []

    def fake_apply_operator(operator_name, model_name, vocab, word_a, word_b, *, top_k, exclude, template="{a} and {b}"):
        seen_excludes.append(exclude)
        return _fake_result("mountain")

    monkeypatch.setattr(sim, "apply_operator", fake_apply_operator)

    sim.simulate_convergence(Path("unused.txt"), "fire", "whale", models=["m1", "m2"], max_rounds=1)

    assert seen_excludes[0] == {"fire", "whale"}


def test_simulate_include_inputs_disables_exclusion(monkeypatch):
    _patch_vocab(monkeypatch)
    seen_excludes = []

    def fake_apply_operator(operator_name, model_name, vocab, word_a, word_b, *, top_k, exclude, template="{a} and {b}"):
        seen_excludes.append(exclude)
        return _fake_result("mountain")

    monkeypatch.setattr(sim, "apply_operator", fake_apply_operator)

    sim.simulate_convergence(Path("unused.txt"), "fire", "whale", models=["m1", "m2"], max_rounds=1, include_inputs=True)

    assert seen_excludes[0] == set()
