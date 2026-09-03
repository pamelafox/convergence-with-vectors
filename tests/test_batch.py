"""Tests for batch CSV pair loading."""

from __future__ import annotations

import pytest

from compare_batch import load_pairs


def test_load_pairs_reads_word_a_word_b_columns(tmp_path):
    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text("word_a,word_b\nfire,whale\nsnow,mountain\n", encoding="utf-8")

    assert load_pairs(csv_path) == [("fire", "whale"), ("snow", "mountain")]


def test_load_pairs_skips_rows_with_blank_words(tmp_path):
    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text("word_a,word_b\nfire,whale\n,mountain\nsnow,\n", encoding="utf-8")

    assert load_pairs(csv_path) == [("fire", "whale")]


def test_load_pairs_requires_expected_columns(tmp_path):
    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text("a,b\nfire,whale\n", encoding="utf-8")

    with pytest.raises(ValueError, match="word_a"):
        load_pairs(csv_path)


def test_load_pairs_raises_when_no_pairs_found(tmp_path):
    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text("word_a,word_b\n", encoding="utf-8")

    with pytest.raises(ValueError, match="No usable word pairs"):
        load_pairs(csv_path)
