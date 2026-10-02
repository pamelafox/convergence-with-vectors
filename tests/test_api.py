"""Tests for the FastAPI backend and its SQLite embedding store.

Each test builds a tiny database of 2D unit vectors and passes a fake
embedding function, so no model is ever downloaded.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import db
from backend.app import create_app
from embeddings import ModelUnavailableError
from operators import normalize

VOCAB = {
    "a_like": np.array([1.0, 0.0]),
    "b_like": np.array([0.0, 1.0]),
    "mid": normalize(np.array([1.0, 1.0])),
    "far": np.array([-1.0, 0.0]),
}
# Vectors the fake model returns for inputs that aren't in the vocabulary
INPUTS = {"x": np.array([1.0, 0.0]), "y": np.array([0.0, 1.0]), "x and y": normalize(np.array([1.0, 1.0]))}


class FakeEmbedder:
    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, model_name: str, texts: list[str]) -> np.ndarray:
        self.calls.append(texts)
        return np.stack([INPUTS[text] for text in texts]).astype(np.float32)


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "test.db"
    conn = db.connect(path)
    for model in ["minilm-l6", "minilm-l12"]:
        db.store_vocab(conn, model, list(VOCAB), np.stack(list(VOCAB.values())))
    conn.close()
    return path


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def client(db_path, embedder):
    with TestClient(create_app(db_path, embed_fn=embedder)) as test_client:
        yield test_client


def test_models_are_listed_in_registry_order_with_labels(client):
    models = client.get("/api/models").json()["models"]
    assert [m["name"] for m in models] == ["minilm-l6", "minilm-l12"]
    assert models[0] == {"name": "minilm-l6", "label": "MiniLM L6", "provider": "sentence-transformers"}


def test_vocab_returns_candidate_words(client):
    assert sorted(client.get("/api/vocab", params={"model": "minilm-l6"}).json()["words"]) == sorted(VOCAB)


def test_unknown_model_is_404(client):
    assert client.get("/api/vocab", params={"model": "nope"}).status_code == 404


def test_combine_centroid_picks_the_midpoint_word(client):
    response = client.get("/api/combine", params={"model": "minilm-l6", "a": "x", "b": "y", "operator": "centroid", "top_k": 2})
    assert response.status_code == 200
    assert response.json()["candidates"][0]["candidate"] == "mid"


def test_combine_normalizes_input_words(client, embedder):
    client.get("/api/combine", params={"model": "minilm-l6", "a": "  X ", "b": "Y"})
    assert embedder.calls[0] == ["x", "y"]


def test_combine_excludes_inputs_unless_asked(client):
    params = {"model": "minilm-l6", "a": "a_like", "b": "b_like", "top_k": 10}
    assert "a_like" not in [c["candidate"] for c in client.get("/api/combine", params=params).json()["candidates"]]
    included = client.get("/api/combine", params={**params, "include_inputs": True}).json()["candidates"]
    assert "a_like" in [c["candidate"] for c in included]


def test_vocab_inputs_come_from_the_database_without_running_the_model(client, embedder):
    client.get("/api/combine", params={"model": "minilm-l6", "a": "a_like", "b": "b_like"})
    assert embedder.calls == []


def test_textual_embeds_the_phrase_once_then_uses_the_cache(client, embedder):
    params = {"model": "minilm-l6", "a": "x", "b": "y", "operator": "textual", "template": "{a} and {b}"}
    first = client.get("/api/combine", params=params).json()["candidates"]
    client.get("/api/combine", params=params)
    assert first[0]["candidate"] == "mid"
    assert embedder.calls == [["x", "y"], ["x and y"]]


def test_cached_inputs_never_become_candidates(client):
    client.get("/api/combine", params={"model": "minilm-l6", "a": "x", "b": "y"})
    candidates = client.get("/api/combine", params={"model": "minilm-l6", "a": "a_like", "b": "b_like", "top_k": 10}).json()["candidates"]
    assert {c["candidate"] for c in candidates} <= set(VOCAB)


@pytest.mark.parametrize("template", ["{a.__class__}", "{0}", "{a!r}", "{a:>10}", "{c}"])
def test_textual_rejects_templates_beyond_a_and_b(client, template):
    response = client.get("/api/combine", params={"model": "minilm-l6", "a": "x", "b": "y", "operator": "textual", "template": template})
    assert response.status_code == 422


def test_unknown_operator_is_422(client):
    assert client.get("/api/combine", params={"model": "minilm-l6", "a": "x", "b": "y", "operator": "mean"}).status_code == 422


def test_overlong_word_is_422(client):
    assert client.get("/api/combine", params={"model": "minilm-l6", "a": "x" * 61, "b": "y"}).status_code == 422


def test_similarities_cover_every_vocabulary_word(client):
    data = client.get("/api/similarities", params={"model": "minilm-l6", "a": "x", "b": "y"}).json()
    assert len(data["words"]) == len(data["sim_a"]) == len(data["sim_b"]) == len(VOCAB)
    assert data["sim_a"][data["words"].index("a_like")] == pytest.approx(1.0)
    assert data["cos_ab"] == pytest.approx(0.0)


def test_simulate_returns_the_path_and_outcome(client):
    response = client.post("/api/simulate", json={"model_a": "minilm-l6", "model_b": "minilm-l12", "word_a": "x", "word_b": "y", "max_rounds": 3})
    assert response.status_code == 200
    data = response.json()
    assert data["outcome"] == "converged"
    assert data["path"][1]["outputs"] == ["mid", "mid"]
    assert data["path"][1]["candidates"][0][0]["candidate"] == "mid"
    assert data["termination"] == {"round": 1}


def test_simulate_validates_round_limit(client):
    response = client.post("/api/simulate", json={"model_a": "minilm-l6", "model_b": "minilm-l12", "word_a": "x", "word_b": "y", "max_rounds": 500})
    assert response.status_code == 422


def test_simulate_records_each_rounds_models(client):
    data = client.post("/api/simulate", json={"model_a": "minilm-l12", "model_b": "minilm-l6", "word_a": "x", "word_b": "y"}).json()
    assert data["path"][0]["models"] is None
    assert data["path"][1]["models"] == ["minilm-l12", "minilm-l6"]


def test_simulate_needs_models_unless_random(client):
    response = client.post("/api/simulate", json={"word_a": "x", "word_b": "y"})
    assert response.status_code == 422


def test_simulate_with_random_models_picks_two_different_models(client):
    response = client.post("/api/simulate", json={"random_models": True, "seed": 1, "word_a": "x", "word_b": "y"})
    assert response.status_code == 200
    players = response.json()["path"][1]["models"]
    assert sorted(players) == ["minilm-l12", "minilm-l6"]


def test_simulate_unknown_model_is_404(client):
    response = client.post("/api/simulate", json={"model_a": "minilm-l6", "model_b": "nope", "word_a": "x", "word_b": "y"})
    assert response.status_code == 404


def test_unavailable_model_is_503(db_path):
    def unavailable(model_name, texts):
        raise ModelUnavailableError("Ollama isn't running")

    with TestClient(create_app(db_path, embed_fn=unavailable)) as test_client:
        response = test_client.get("/api/combine", params={"model": "minilm-l6", "a": "x", "b": "y"})
    assert response.status_code == 503
    assert response.json() == {"detail": "Ollama isn't running"}


def test_serves_pages_but_not_repository_files(client):
    assert client.get("/web/play.html").status_code == 200
    assert client.get("/pyproject.toml").status_code == 404
    assert client.get("/web/../pyproject.toml").status_code == 404


def test_root_redirects_to_the_game(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/web/play.html"


def test_missing_database_fails_at_startup(tmp_path):
    with pytest.raises(RuntimeError, match="build_db.py"):
        with TestClient(create_app(tmp_path / "missing.db", embed_fn=FakeEmbedder())):
            pass


def test_store_vocab_replaces_previous_vocabulary(db_path):
    conn = db.connect(db_path)
    db.store_vocab(conn, "minilm-l6", ["mid"], np.stack([VOCAB["mid"]]))
    assert db.load_vocab(conn, "minilm-l6").words == ["mid"]
    assert sorted(db.load_vocab(conn, "minilm-l12").words) == sorted(VOCAB)
    conn.close()
