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
| `index.html` | reveal.js slides, a single HTML file |
| `slides_assets/` | CSS and images for the slides |
| `http/` | Sample embeddings requests for the VS Code REST Client extension |
| `requirements.txt` | Runtime dependencies |
| `requirements-dev.txt` | Lint/format tooling, includes `requirements.txt` |

Python game code lives in top-level modules named after what they demonstrate,
following the style of the previous talk's repository (for example `example.py`, `convergence.py`).

## Environment

* Python 3.11 (matches the dev container and the CI workflow).
* Install runtime dependencies with `python -m pip install -r requirements.txt`.
* Install dev dependencies with `python -m pip install -r requirements-dev.txt`.
* Embedding models come from GitHub Models (needs `GITHUB_TOKEN`), Ollama (local), and
  `sentence-transformers` (downloaded from Hugging Face).

## Conventions

* Format and lint with `black` and `ruff`, both configured in `pyproject.toml` with `line-length = 200`.
* Run the same checks CI runs before committing:

    ```shell
    ruff check .
    black . --check
    ```

  or run `pre-commit run --all-files` if pre-commit is installed.
* Scripts are demo code for a live talk: keep them short, readable, and runnable top-to-bottom,
  and print results with `rich` where it helps the audience.
* Access models through the `openai` client library (both GitHub Models and Ollama expose
  OpenAI-compatible endpoints) or through `sentence-transformers` for local models.
* Never hardcode secrets; read tokens from environment variables, optionally loaded via `python-dotenv`.
* There is no test suite in this repository; verify changes by running the scripts.
