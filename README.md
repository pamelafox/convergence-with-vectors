# Convergence in vector space

This repository contains the slides and code for the PyBay talk "Convergence in vector space",
which explores vector embeddings by recreating the improv word game
[Convergence](#the-convergence-game) with Python, embedding models, and cosine similarity.

> Vector embeddings power modern search, but they aren't always so easy to understand.
> Instead of starting with math, let's build intuition through a word game!
> Convergence is a word-association game where players try to independently arrive at the same word.
> After we play it in person, I'll recreate it using Python, embeddings, and cosine similarity.
> We'll pit different embedding models against each other to see which converge quickly,
> which get stuck, and which make surprisingly weird choices.

* [The Convergence game](#the-convergence-game)
* [Viewing the slides](#viewing-the-slides)
* [Setting up the environment](#setting-up-the-environment)
* [Using embedding models](#using-embedding-models)
* [Repository structure](#repository-structure)

## The Convergence game

1. Everyone stands in a circle.
2. Two people stand in the center facing each other.
3. Everyone counts down, "3..2..1"
4. When the countdown finishes, the two people both yell a word at the same time.
   The word is typically a noun, an actual object in the world, like "fire" and "whale".
   They may have to repeat themselves after, since it can be hard to understand two words said at once.
5. The two people return to the circle. Everyone else starts thinking to themselves,
   "What single word is at the intersection of those two words?"
6. Now two new people enter the circle, whoever is ready first. Once again, everyone else counts down,
   and the two people yell their words at the same time, like "blubber" and "engine".
7. If those two people actually managed to say the same word together, everyone wins and celebration abounds!
   If not, the game continues until two people finally converge.
8. Play multiple rounds until you've had your fill.

The Python version of the game replaces the players with embedding models:
each word is turned into a vector, and the "word at the intersection" is found by
comparing candidate words to the average of the two vectors using cosine similarity.
A round converges when both players pick the same word.

## Viewing the slides

The slides are a single [reveal.js](https://revealjs.com/) page in [index.html](index.html).
View them [published on GitHub pages](https://pamelafox.github.io/convergence-with-vectors/),
or run a local Python server in the repo:

```shell
python -m http.server 8000
```

## Setting up the environment

If you open this up in a VS Code Dev Container or GitHub Codespaces, everything will be set up for you.
If not, follow these steps:

1. Set up a Python 3.11 (or higher) virtual environment and activate it.

2. Install the required packages:

    ```shell
    python -m pip install -r requirements.txt
    ```

3. For local development (linting and formatting), install the dev packages and pre-commit hooks:

    ```shell
    python -m pip install -r requirements-dev.txt
    pre-commit install
    ```

## Using embedding models

The talk compares embedding models from multiple sources:

| Source | Example models | Setup |
|--------|----------------|-------|
| [GitHub Models](https://github.com/marketplace/models) | `openai/text-embedding-3-small`, `cohere/cohere-embed-v3-english` | Requires a `GITHUB_TOKEN` environment variable |
| [Ollama](https://ollama.com/) | `nomic-embed-text`, `mxbai-embed-large` | Requires Ollama installed locally, then `ollama pull <model>` |
| [sentence-transformers](https://sbert.net/) | `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `microsoft/harrier-oss-v1-270m` | Downloads models from Hugging Face on first use |

To use GitHub Models, you need a `GITHUB_TOKEN` environment variable that stores a GitHub personal access token.
If you're running this inside a GitHub Codespace, the token is automatically available.
If not, generate a new [personal access token](https://github.com/settings/tokens) and run:

```shell
export GITHUB_TOKEN="your-github-token-goes-here"
```

To use Ollama models, pull the embedding model first:

```shell
ollama pull nomic-embed-text
```

Then compute normalized embeddings with the local model:

```shell
python ollama.py fire whale
```

Pass `--model` to use another locally installed Ollama embedding model.

To compute normalized embeddings locally with Microsoft's 270-million-parameter
[Harrier](https://huggingface.co/microsoft/harrier-oss-v1-270m) model, run:

```shell
python harrier.py fire whale
```

The model is downloaded from Hugging Face the first time the script runs.

To compare cosine similarities from the CPU-friendly
[`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
and [`all-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)
models, run:

```shell
python minilm.py fire whale
```

You can pass more than two texts to compare every pair. Both models are downloaded from
Hugging Face on first use and explicitly run on CPU.

The [http](http) folder contains sample REST Client requests for the GitHub Models and Ollama
embeddings endpoints, which are a quick way to check that your setup works.

## Repository structure

| Path | Purpose |
|------|---------|
| [index.html](index.html) | The reveal.js slides for the talk |
| [ollama.py](ollama.py) | Computes local embeddings with Ollama |
| [harrier.py](harrier.py) | Computes local embeddings with Microsoft Harrier |
| [minilm.py](minilm.py) | Compares cosine similarities from two MiniLM models on CPU |
| [slides_assets/](slides_assets) | CSS and images used by the slides |
| [http/](http) | Sample embeddings requests for the VS Code REST Client extension |
| [AGENTS.md](AGENTS.md) | Context and conventions for AI coding agents |
