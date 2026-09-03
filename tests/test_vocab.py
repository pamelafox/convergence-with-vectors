"""Tests for vocabulary file parsing."""

from __future__ import annotations

import pytest

from vocab import load_vocab


def test_load_vocab_strips_blank_lines_and_comments(tmp_path):
    vocab_file = tmp_path / "words.txt"
    vocab_file.write_text("fire\n\n# a comment\nwhale\n   \nsnow\n", encoding="utf-8")

    assert load_vocab(vocab_file) == ["fire", "whale", "snow"]


def test_load_vocab_strips_surrounding_whitespace(tmp_path):
    vocab_file = tmp_path / "words.txt"
    vocab_file.write_text("  fire  \nwhale\t\n", encoding="utf-8")

    assert load_vocab(vocab_file) == ["fire", "whale"]


def test_load_vocab_deduplicates_preserving_first_seen_order(tmp_path):
    vocab_file = tmp_path / "words.txt"
    vocab_file.write_text("fire\nwhale\nfire\nsnow\nwhale\n", encoding="utf-8")

    assert load_vocab(vocab_file) == ["fire", "whale", "snow"]


def test_load_vocab_is_case_sensitive(tmp_path):
    vocab_file = tmp_path / "words.txt"
    vocab_file.write_text("Fire\nfire\n", encoding="utf-8")

    assert load_vocab(vocab_file) == ["Fire", "fire"]


def test_load_vocab_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_vocab(tmp_path / "does_not_exist.txt")


def test_load_vocab_empty_file_raises(tmp_path):
    vocab_file = tmp_path / "words.txt"
    vocab_file.write_text("\n\n# only comments\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no usable words"):
        load_vocab(vocab_file)


def test_load_vocab_supports_utf8(tmp_path):
    vocab_file = tmp_path / "words.txt"
    vocab_file.write_text("café\nnaïve\n", encoding="utf-8")

    assert load_vocab(vocab_file) == ["café", "naïve"]
