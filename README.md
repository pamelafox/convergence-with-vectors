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
* [Comparing word-combination operators](#comparing-word-combination-operators)
* [Playing Convergence in the browser](#playing-convergence-in-the-browser)
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

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).

2. Create the virtual environment and install the packages (including dev tools) from `uv.lock`:

    ```shell
    uv sync
    ```

3. For local development (linting and formatting), install the pre-commit hooks:

    ```shell
    uv run pre-commit install
    ```

The commands below use `uv run`, which runs inside the project's virtual environment.
Alternatively, activate it with `source .venv/bin/activate` and drop the `uv run` prefix.

## Using embedding models

The talk compares embedding models from multiple sources:

| Source | Example models | Setup |
| -------- | ---------------- | ------- |
| [Microsoft Foundry](https://ai.azure.com/) | `text-embedding-3-small`, `text-embedding-3-large` | Requires a Foundry resource with a deployed embedding model |
| [Ollama](https://ollama.com/) | `nomic-embed-text`, `mxbai-embed-large` | Requires Ollama installed locally, then `ollama pull <model>` |
| [sentence-transformers](https://sbert.net/) | `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `microsoft/harrier-oss-v1-270m` | Downloads models from Hugging Face on first use |

To use Foundry models, deploy an embedding model (named after the model, like `text-embedding-3-small`)
in a Foundry resource, then set the endpoint and a Microsoft Entra access token:

```shell
az login
export AZURE_OPENAI_ENDPOINT="https://your-resource-name.openai.azure.com"
export AZURE_OPENAI_TOKEN=$(az account get-access-token --resource https://cognitiveservices.azure.com --query accessToken -o tsv)
```

Your account needs the "Cognitive Services OpenAI User" role on the Foundry resource.

To use Ollama models, pull the embedding model first:

```shell
ollama pull nomic-embed-text
```

Then compute normalized embeddings with the local model:

```shell
uv run ollama.py fire whale
```

Pass `--model` to use another locally installed Ollama embedding model.

To compute normalized embeddings locally with Microsoft's 270-million-parameter
[Harrier](https://huggingface.co/microsoft/harrier-oss-v1-270m) model, run:

```shell
uv run harrier.py fire whale
```

The model is downloaded from Hugging Face the first time the script runs.

To compare cosine similarities from the CPU-friendly
[`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
and [`all-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)
models, run:

```shell
uv run minilm.py fire whale
```

You can pass more than two texts to compare every pair. Both models are downloaded from
Hugging Face on first use and explicitly run on CPU.

The [http](http) folder contains sample REST Client requests for the Foundry and Ollama
embeddings endpoints, which are a quick way to check that your setup works.

## Comparing word-combination operators

This experiment compares several ways of combining two words' embeddings into a single
"intersection" word, side by side across [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
and [`sentence-transformers/all-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2),
both of which run on CPU. The vocabulary is a closed candidate list: every operator returns a word
from that file.

### Setup

Install the runtime dependencies as described above (`sentence-transformers`, `numpy`, `rich`).
The two MiniLM models are downloaded from Hugging Face the first time they're used.

### Operators

All embeddings are L2-normalized, so cosine similarity is a plain dot product.

| Operator | Definition |
| ---------- | ------------ |
| `centroid` | Nearest vocabulary word to `normalize(embed(a) + embed(b))`. |
| `balanced` | For each candidate `w`: `min(cos(w, a), cos(w, b))` — rewards being close to *both* inputs. |
| `geometric_mean` | Geometric mean of the two cosine similarities. **Negative similarities are clamped to 0 before multiplying** (a geometric mean is undefined for negatives, and multiplying two negatives would otherwise produce a misleading positive score), so a candidate dissimilar to either input scores exactly 0. |
| `textual` | Embed a phrase containing both words (e.g. `"{a} and {b}"`) and return the nearest vocabulary word to *that* embedding. The phrase template is configurable via `--template` (repeatable), so different templates can be compared side by side. |

### Commands

Compare all operators for one word pair:

```shell
uv run compare_pair.py --vocab data/sample_vocab.txt --word-a snow --word-b mountain --top-k 10
```

Run a batch of word pairs from a CSV file (columns `word_a`, `word_b`):

```shell
uv run compare_batch.py --vocab data/sample_vocab.txt --pairs data/sample_pairs.csv --output-csv batch_results.csv
```

Simulate Convergence: two models independently apply the same operator to a word pair, and their
outputs become the next round's pair, until both models agree, a pair repeats, no candidates remain after exclusion (stalled),
or a round limit is hit:

```shell
uv run simulate_convergence.py --vocab data/sample_vocab.txt --word-a fire --word-b whale --operator centroid --max-rounds 10
```

All three scripts accept `--models` (defaults to both MiniLM models), `--include-inputs` (to allow
the input words themselves as candidates, excluded by default), `--cache-dir` (defaults to
`.embedding_cache/`, which caches each model's vocabulary embeddings on disk so repeat runs skip
re-embedding), and `--output-csv`/`--output-json` for machine-readable results alongside the
`rich` terminal tables. Run `--help` on any script for the full option list.

### Interpretive cautions

* MiniLM sentence embeddings are **not** trained to support Word2Vec-style relational analogy
  arithmetic (e.g. `king - man + woman ≈ queen`). The `centroid` operator here is a *nearest-neighbor
  to an averaged vector*, which is a different (and much weaker) claim than relational analogy —
  don't read convergence, or the lack of it, as evidence for or against analogy arithmetic.
* Embedding the phrase `"a and b"` (the `textual` operator) is **not** mathematically equivalent to
  adding the independently generated embeddings of `a` and `b` — the transformer attends to the whole
  phrase, so results can differ meaningfully from `centroid`.
* Results are strongly shaped by the candidate vocabulary and, for `textual`, by the phrase template.
  A small or thematically narrow vocabulary will tend to converge faster and more predictably than a
  large, diverse one. Use this experiment to *expose* those differences across operators and models,
  not to declare a single operator or model universally "best".

### Tests

Focused tests cover vocabulary parsing, operator scoring math, tie-breaking, exclusion behavior, and
the convergence simulation's control flow (using small synthetic vectors so no model download is
required):

```shell
uv run pytest tests/
```

## Playing Convergence in the browser

The [web](web) folder has a static, GitHub-Pages-friendly version of the Convergence simulation:
[**web/play.html**](web/play.html) lets you pick two starting words, two models, and an operator,
then runs the same round-by-round simulation as `simulate_convergence.py` entirely client-side.
The page opens in **Game night** mode, an animated, model-filled game circle that counts down before
each simultaneous shout, starts from a random pair, and celebrates convergence with confetti.
Switch to **Vector lab** on the same page to keep using the full controls and detailed score,
candidate, contour, and operator-comparison views described below.
Type in either word box for native HTML vocabulary suggestions, or use **Random words** to choose
two distinct starting words. Operator radio buttons explain each scoring rule and expose its formula.
Expand **Why these words?** for any round to see each model's top five candidates, similarities to
both inputs, winning score margin (or tie-break), and a scatterplot of all eligible candidates with
equal-score contours. The round table highlights agreement and links repeated pairs back to their
earlier round; stalled games report the reason and the attempted round.
Within each model's round details, expand **What would another operator choose?** to compare all
three operators' winners and top five candidates for that same input pair. Different winning words
are highlighted; this comparison does not change the game or replay subsequent rounds.

[**web/operators.html**](web/operators.html) is a visual explainer for the operators' math, with the view that fits each one best.
Centroid and balanced use a rotatable 3D dome (built with three.js, loaded from a CDN) that places every word vector
relative to the plane of the two input words: centroid shows each word's angle to the midpoint vector, and balanced grows
circles around both input words. Geometric mean shows each word's two similarities as a rectangle next to the square with
the same area. A side-by-side table compares each operator's top 5.

There's no model inference in the browser: `export_web_embeddings.py` precomputes normalized
embeddings for a closed ~1000-word vocabulary ([data/vocab_1000.txt](data/vocab_1000.txt)) for each
model and writes them to JSON files under `web/data/`, which the page simply fetches. Because of
this, the web version only supports the vocabulary-only operators (`centroid`, `balanced`,
`geometric_mean`) and requires both starting words to come from that vocabulary — the `textual`
operator, which needs to embed an arbitrary phrase at request time, isn't available in the browser.

To regenerate the JSON files after changing the vocabulary or model list:

```shell
uv run export_web_embeddings.py --vocab data/vocab_1000.txt --output-dir web/data
```

To try it locally, run a static server from the repo root and open `web/play.html`:

```shell
python -m http.server 8000
```

## Repository structure

| Path | Purpose |
| ------ | --------- |
| [index.html](index.html) | The reveal.js slides for the talk |
| [ollama.py](ollama.py) | Computes local embeddings with Ollama |
| [harrier.py](harrier.py) | Computes local embeddings with Microsoft Harrier |
| [vocab.py](vocab.py) | Vocabulary file parsing for the word-combination experiments |
| [embeddings.py](embeddings.py) | Model registry and cached vocabulary embeddings |
| [operators.py](operators.py) | The four word-combination operators and their scoring math |
| [reporting.py](reporting.py) | Result rows, CSV/JSON writers, and terminal tables |
| [compare_pair.py](compare_pair.py) | CLI: compare all operators for one word pair |
| [compare_batch.py](compare_batch.py) | CLI: run a batch of word pairs from a CSV file |
| [simulate_convergence.py](simulate_convergence.py) | CLI: simulate the Convergence game between two models |
| [data/sample_vocab.txt](data/sample_vocab.txt) | Sample candidate vocabulary |
| [data/sample_pairs.csv](data/sample_pairs.csv) | Sample word pairs for `compare_batch.py` |
| [data/vocab_1000.txt](data/vocab_1000.txt) | ~1000-word vocabulary used by the browser game |
| [export_web_embeddings.py](export_web_embeddings.py) | CLI: precompute and export vocabulary embeddings as JSON for the browser game |
| [web/](web) | Static, GitHub-Pages-friendly browser version of the Convergence game |
| [tests/](tests) | Focused pytest tests for the modules above |
| [minilm.py](minilm.py) | Compares cosine similarities from two MiniLM models on CPU |
| [slides_assets/](slides_assets) | CSS and images used by the slides |
| [http/](http) | Sample embeddings requests for the VS Code REST Client extension |
| [AGENTS.md](AGENTS.md) | Context and conventions for AI coding agents |
