"""End-to-end tests for the game page, driving a real browser with Playwright.

By default the backend runs with a tiny 2D vocabulary and a fake embedding function, so no model is downloaded.
Pass ``--real-models`` to run the same tests against data/convergence.db and the real embedding models.
"""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass

import numpy as np
import pytest
import uvicorn
from playwright.sync_api import Page, expect

from backend import db
from backend.app import DEFAULT_DB_PATH, create_app
from operators import normalize

FAKE_VOCAB = {
    "a_like": np.array([1.0, 0.0]),
    "b_like": np.array([0.0, 1.0]),
    "mid": normalize(np.array([1.0, 1.0])),
    "far": np.array([-1.0, 0.0]),
}
# "x" and "y" sit on the axes, so every model's centroid pick for x + y is "mid"
FAKE_INPUTS = {"x": np.array([1.0, 0.0]), "y": np.array([0.0, 1.0])}
# Real models load slowly the first time they embed a word
TIMEOUT = 120_000


def fake_embed(model_name: str, texts: list[str]) -> np.ndarray:
    return np.stack([FAKE_INPUTS.get(text, FAKE_VOCAB.get(text, np.array([0.0, -1.0]))) for text in texts]).astype(np.float32)


@dataclass
class Server:
    url: str
    words: tuple[str, str]


def build_fake_db(path):
    conn = db.connect(path)
    for model in ["minilm-l6", "harrier-270m"]:
        db.store_vocab(conn, model, list(FAKE_VOCAB), np.stack(list(FAKE_VOCAB.values())))
    conn.close()


@pytest.fixture(scope="module")
def server(request, tmp_path_factory):
    if request.config.getoption("--real-models"):
        if not DEFAULT_DB_PATH.exists():
            pytest.skip("--real-models needs data/convergence.db; build it with: uv run build_db.py")
        app = create_app(DEFAULT_DB_PATH)
        words = ("fire", "whale")
    else:
        path = tmp_path_factory.mktemp("e2e") / "test.db"
        build_fake_db(path)
        app = create_app(path, embed_fn=fake_embed)
        words = ("x", "y")

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    uv_server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=uv_server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not uv_server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("Test server didn't start")
        time.sleep(0.05)
    yield Server(f"http://127.0.0.1:{port}", words)
    uv_server.should_exit = True
    thread.join(timeout=5)


def model_pick(page: Page, server: Server, model: str) -> str:
    """Ask the backend which word a model picks for the starting pair, so assertions hold for fake and real models."""
    a, b = server.words
    response = page.request.get(f"{server.url}/api/combine", params={"model": model, "a": a, "b": b, "operator": "centroid", "top_k": 1}, timeout=TIMEOUT)
    return response.json()["candidates"][0]["candidate"]


def open_game(page: Page, server: Server):
    page.goto(f"{server.url}/web/play.html")
    expect(page.locator("#party-play")).to_be_enabled()
    page.get_by_text("Customize the game").click()
    page.get_by_label("Starting word A").fill(server.words[0])
    page.get_by_label("Starting word B").fill(server.words[1])


def test_models_play_and_converge(page: Page, server: Server):
    open_game(page, server)
    expect(page.get_by_label("Player A")).to_have_value("minilm-l6")
    expect(page.get_by_label("Player B")).to_have_value("harrier-270m")
    # A model playing itself always agrees, so the game converges in round 1 with fake or real models
    page.get_by_label("Player B").select_option("minilm-l6")
    word = model_pick(page, server, "minilm-l6")

    page.locator("#party-play").click()

    expect(page.locator("#party-outcome")).to_have_text("1 round to converge", timeout=TIMEOUT)
    expect(page.locator(".converged-round")).to_have_attribute("aria-label", f"Round 1: MiniLM L6 said {word}, MiniLM L6 said {word}, converged")
    expect(page.locator(".converge-mark")).to_be_visible()
    expect(page.locator("#party-play")).to_have_text("↻ Play again")


def test_human_plays_a_model_and_converges(page: Page, server: Server):
    open_game(page, server)
    page.get_by_label("You vs. a model").check()
    expect(page.get_by_label("Your opponent")).to_have_value("harrier-270m")
    word = model_pick(page, server, "harrier-270m")

    page.locator("#party-play").click()
    word_input = page.get_by_label("Your word")
    expect(word_input).to_be_visible(timeout=TIMEOUT)
    word_input.fill(word)
    word_input.press("Enter")

    expect(page.locator("#party-outcome")).to_have_text("1 round to converge", timeout=TIMEOUT)
    expect(page.locator(".converged-round")).to_have_attribute("aria-label", f"Round 1: You said {word}, Harrier 270M said {word}, converged")


def test_human_cannot_repeat_a_word_from_the_pair(page: Page, server: Server):
    open_game(page, server)
    page.get_by_label("You vs. a model").check()

    page.locator("#party-play").click()
    word_input = page.get_by_label("Your word")
    expect(word_input).to_be_visible(timeout=TIMEOUT)
    word_input.fill(server.words[0])
    word_input.press("Enter")

    expect(page.locator("#party-human-error")).to_have_text("Pick a new word, not one of the current pair.")
