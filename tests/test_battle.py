"""Tests for the model battle, using a tiny database of 2D vectors."""

from __future__ import annotations

import numpy as np

from backend import db
from model_battle import play_battle, summarize
from operators import normalize

WORDS = {"east": [1.0, 0.0], "north": [0.0, 1.0], "northeast": normalize(np.array([1.0, 1.0])), "west": [-1.0, 0.0]}


def test_battle_plays_every_pairing_including_self_play(tmp_path):
    conn = db.connect(tmp_path / "battle.db")
    for model in ["minilm-l6", "minilm-l12"]:
        db.store_vocab(conn, model, list(WORDS), np.array(list(WORDS.values()), dtype=np.float32))

    games = play_battle(conn, ["minilm-l6", "minilm-l12"], [("east", "north")], max_rounds=3)

    assert [(g["model_a"], g["model_b"]) for g in games] == [("minilm-l6", "minilm-l6"), ("minilm-l6", "minilm-l12"), ("minilm-l12", "minilm-l12")]
    # Identical models always pick the same word, so they converge in round 1
    assert games[0]["outcome"] == "converged" and games[0]["rounds"] == 1
    assert games[0]["path"] == [["east", "north"], ["northeast", "northeast"]]


def test_summarize_reports_rates_per_pairing():
    games = [
        {"model_a": "a", "model_b": "b", "outcome": "converged", "rounds": 2},
        {"model_a": "a", "model_b": "b", "outcome": "converged", "rounds": 4},
        {"model_a": "a", "model_b": "b", "outcome": "loop", "rounds": 3},
        {"model_a": "a", "model_b": "b", "outcome": "round_limit", "rounds": 10},
    ]

    assert summarize(games) == {("a", "b"): {"games": 4, "converged": 0.5, "avg_rounds": 3.0, "loops": 0.25}}
