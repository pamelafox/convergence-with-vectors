# AGENTS.md

## About this repository

This repository holds the slides and Python code for the PyBay talk **"Convergence in vector space"**
by Pamela Fox. The talk explains vector embeddings by recreating the improv word game *Convergence*
with Python, embedding models, and cosine similarity.

The repository previously held the code for an older talk, "Playing improv with Python",
and is now being rebuilt from that baseline for the new talk.

### The game

*Convergence* is a word-association game. Two players simultaneously say a word (usually a concrete
noun, like "fire" and "whale"). Everyone thinks about which single word sits at the intersection of
those two words. Two new players then simultaneously say their word (like "blubber" and "engine").
The game continues, round after round, until two players say the same word and the group converges.

The Python version replaces players with embedding models: each word becomes a vector, and a
"player" picks the candidate word whose embedding is most similar (by cosine similarity) to the
average of the two previous words' vectors. A round converges when both players pick the same word.
The talk pits different embedding models against each other to see which converge quickly, which get
stuck, and which make weird choices.

## Structure

| Path | Purpose |
|------|---------|
| `docs/index.html` | reveal.js slides, a single HTML file, published with GitHub Pages |
| `docs/slides_assets/` | CSS, images, and video for the slides |
| `backend/` | FastAPI app (`app.py`) and SQLite embedding store (`db.py`); serves `web/` and `/api` |
| `build_db.py` | Embeds the vocabulary into `data/convergence.db` (generated, gitignored) for the backend |
| `web/` | Pages for the web app; `web/api.js` is the only module that talks to the backend |
| `http/` | Sample embeddings requests for the VS Code REST Client extension |
| `pyproject.toml` | Runtime dependencies, `dev` dependency group, and ruff/black config |
| `uv.lock` | Locked dependency versions managed by uv |

Python game code lives in top-level modules named after what they demonstrate,
following the style of the previous talk's repository (for example `example.py`, `convergence.py`).
All scoring happens in Python: the web pages call the backend rather than reimplementing operators in JavaScript.

## Environment

* Python 3.11 (matches the dev container and the CI workflow).
* Manage dependencies with [uv](https://docs.astral.sh/uv/): `uv sync` installs runtime and dev dependencies into `.venv`.
* Add dependencies with `uv add <pkg>` (or `uv add --dev <pkg>`), which updates `pyproject.toml` and `uv.lock`.
* Run scripts and tools with `uv run`, e.g. `uv run compare_pair.py ...`.
* Run the web app with `uv run build_db.py` (once) and then `uv run fastapi dev backend/app.py`.
* Embedding models come from Microsoft Foundry (needs `AZURE_OPENAI_ENDPOINT` and Entra auth), Ollama (local), and
  `sentence-transformers` (downloaded from Hugging Face).

## Conventions

* Format and lint with `black` and `ruff`, both configured in `pyproject.toml` with `line-length = 200`.
* Run the same checks CI runs before committing:

    ```shell
    uv run ruff check .
    uv run black . --check
    ```

  or run `pre-commit run --all-files` if pre-commit is installed.
* Scripts are demo code for a live talk: keep them short, readable, and runnable top-to-bottom,
  and print results with `rich` where it helps the audience.
* Access models through the `openai` client library (both Foundry and Ollama expose
  OpenAI-compatible endpoints) or through `sentence-transformers` for local models.
* Never hardcode secrets; read tokens from environment variables, optionally loaded via `python-dotenv`.
* Run the tests with `uv run pytest tests/`; they use small synthetic vectors and fake embedders, so no model downloads are needed.
