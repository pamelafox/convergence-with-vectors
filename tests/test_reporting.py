"""Tests for report row construction and CSV/JSON output."""

from __future__ import annotations

import csv
import json

from operators import CandidateScore
from reporting import FIELDNAMES, candidate_rows, write_csv, write_json


def _sample_candidates() -> list[CandidateScore]:
    return [
        CandidateScore(rank=1, candidate="mountain", score=0.9, sim_a=0.8, sim_b=0.7),
        CandidateScore(rank=2, candidate="snow", score=0.5, sim_a=0.4, sim_b=0.3),
    ]


def test_candidate_rows_carries_through_metadata():
    rows = candidate_rows("fire", "whale", "minilm-l6", "centroid", _sample_candidates(), template=None, round_number=2, outcome="loop")
    assert len(rows) == 2
    first = rows[0]
    assert first.word_a == "fire"
    assert first.word_b == "whale"
    assert first.model == "minilm-l6"
    assert first.operator == "centroid"
    assert first.round == 2
    assert first.outcome == "loop"
    assert first.candidate == "mountain"
    assert first.score == 0.9


def test_write_csv_round_trips_all_fields(tmp_path):
    rows = candidate_rows("fire", "whale", "minilm-l6", "centroid", _sample_candidates())
    out_path = tmp_path / "results.csv"
    write_csv(rows, out_path)

    with out_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == FIELDNAMES
        records = list(reader)

    assert len(records) == 2
    assert records[0]["candidate"] == "mountain"
    assert records[0]["word_a"] == "fire"
    assert records[1]["rank"] == "2"


def test_write_json_round_trips_all_fields(tmp_path):
    rows = candidate_rows("fire", "whale", "minilm-l6", "centroid", _sample_candidates())
    out_path = tmp_path / "results.json"
    write_json(rows, out_path)

    records = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(records) == 2
    assert records[0]["candidate"] == "mountain"
    assert records[0]["score"] == 0.9
    assert set(records[0].keys()) == set(FIELDNAMES)
