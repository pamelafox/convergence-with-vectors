"""Tests for exporting precomputed vocabulary embeddings to JSON for the web game."""

from __future__ import annotations

import json

import numpy as np

import export_web_embeddings as export_mod
from embeddings import VocabEmbeddings


def _fake_get_vocab_embeddings(model_name, words, cache_dir):
    # Deterministic per-model fake vectors so the test doesn't need to
    # download a real model, but still exercises the real JSON shape/rounding.
    vectors = np.array([[i + 0.123456789, -i - 0.987654321] for i in range(len(words))], dtype=np.float32)
    return VocabEmbeddings(words=words, vectors=vectors, index={w: i for i, w in enumerate(words)})


def test_export_model_writes_words_and_rounded_vectors(tmp_path, monkeypatch):
    monkeypatch.setattr(export_mod, "get_vocab_embeddings", _fake_get_vocab_embeddings)
    words = ["fire", "whale", "snow"]

    out_path = export_mod.export_model("fake-model", words, tmp_path / "cache", tmp_path / "out", precision=4)

    assert out_path == tmp_path / "out" / "fake-model.json"
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["model"] == "fake-model"
    assert payload["words"] == words
    assert len(payload["vectors"]) == len(words)
    assert payload["vectors"][1] == [round(1 + 0.123456789, 4), round(-1 - 0.987654321, 4)]


def test_main_writes_manifest_with_all_requested_models(tmp_path, monkeypatch):
    monkeypatch.setattr(export_mod, "get_vocab_embeddings", _fake_get_vocab_embeddings)
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text("fire\nwhale\n", encoding="utf-8")
    output_dir = tmp_path / "out"

    monkeypatch.setattr(
        "sys.argv",
        ["export_web_embeddings.py", "--vocab", str(vocab_path), "--models", "model-a", "model-b", "--output-dir", str(output_dir)],
    )
    export_mod.main()

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["words_count"] == 2
    assert manifest["models"] == ["model-a", "model-b"]
    assert (output_dir / "model-a.json").exists()
    assert (output_dir / "model-b.json").exists()


def test_export_model_vectors_match_get_vocab_embeddings(tmp_path, monkeypatch):
    """The exported JSON vectors should be a rounded copy of what get_vocab_embeddings returns."""
    monkeypatch.setattr(export_mod, "get_vocab_embeddings", _fake_get_vocab_embeddings)
    words = ["fire", "whale"]
    cache_dir = tmp_path / "cache"

    out_path = export_mod.export_model("fake-model", words, cache_dir, tmp_path / "out", precision=6)

    expected = export_mod.get_vocab_embeddings("fake-model", words, cache_dir)
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    for exported_row, expected_row in zip(payload["vectors"], expected.vectors):
        np.testing.assert_allclose(exported_row, expected_row, atol=1e-6)


def test_export_model_creates_output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(export_mod, "get_vocab_embeddings", _fake_get_vocab_embeddings)
    output_dir = tmp_path / "nested" / "out"

    export_mod.export_model("fake-model", ["fire"], tmp_path / "cache", output_dir, precision=6)

    assert (output_dir / "fake-model.json").exists()
