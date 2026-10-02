#!/usr/bin/env python
"""Embedding model battle: play many games of Convergence between every pair of models.

Each pairing plays the same random starting pairs, so the results are directly comparable.
Vectors come from the SQLite database built by build_db.py, so this runs in seconds
(the "textual" operator also embeds phrases, which loads the models).

Example:

    uv run model_battle.py --games 200 --operator centroid --output-json battle.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

from rich.console import Console
from rich.table import Table

from backend import db
from embeddings import MODELS, embed_texts
from operators import OPERATOR_NAMES
from simulate_convergence import run_simulation
from vocab import load_vocab


def play_battle(conn, models: list[str], start_pairs: list[tuple[str, str]], *, operator: str = "centroid", max_rounds: int = 10) -> list[dict]:
    """Play every starting pair for every pairing of models (including each model against itself)."""
    vocabs = {model: db.load_vocab(conn, model) for model in models}
    store = db.EmbeddingStore(conn, embed_texts)
    games = []
    for model_a, model_b in itertools.combinations_with_replacement(models, 2):
        for word_a, word_b in start_pairs:
            result = run_simulation([model_a, model_b], vocabs, word_a, word_b, operator=operator, max_rounds=max_rounds, top_k=1, embed=store.embed)
            games.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "start": [word_a, word_b],
                    "outcome": result["outcome"],
                    "rounds": result["termination"]["round"],
                    "path": [list(step["pair_out"]) for step in result["path"]],
                }
            )
    return games


def summarize(games: list[dict]) -> dict[tuple[str, str], dict]:
    """Per pairing: number of games, convergence rate, average rounds to converge, and loop rate."""
    by_pairing: dict[tuple[str, str], list[dict]] = {}
    for game in games:
        by_pairing.setdefault((game["model_a"], game["model_b"]), []).append(game)
    summary = {}
    for pairing, pairing_games in by_pairing.items():
        converged = [g for g in pairing_games if g["outcome"] == "converged"]
        summary[pairing] = {
            "games": len(pairing_games),
            "converged": len(converged) / len(pairing_games),
            "avg_rounds": sum(g["rounds"] for g in converged) / len(converged) if converged else None,
            "loops": sum(g["outcome"] == "loop" for g in pairing_games) / len(pairing_games),
        }
    return summary


def print_report(models: list[str], games: list[dict], summary: dict[tuple[str, str], dict], console: Console) -> None:
    label = {model: MODELS[model].label for model in models}

    matrix = Table(title="Convergence rate (avg rounds when converged)")
    matrix.add_column("")
    for model in models:
        matrix.add_column(label[model], justify="right")
    for row_model in models:
        cells = []
        for col_model in models:
            stats = summary.get((row_model, col_model)) or summary[(col_model, row_model)]
            rounds = f" ({stats['avg_rounds']:.1f})" if stats["avg_rounds"] else ""
            cells.append(f"{stats['converged']:.0%}{rounds}")
        matrix.add_row(label[row_model], *cells)
    console.print(matrix)

    ranking = Table(title="Each model against the others (excluding itself)")
    for column in ("Model", "Converged", "Avg rounds", "Loops"):
        ranking.add_column(column)
    rows = []
    for model in models:
        opponents = [g for g in games if model in (g["model_a"], g["model_b"]) and g["model_a"] != g["model_b"]]
        if not opponents:
            continue
        converged = [g for g in opponents if g["outcome"] == "converged"]
        avg_rounds = sum(g["rounds"] for g in converged) / len(converged) if converged else float("nan")
        loops = sum(g["outcome"] == "loop" for g in opponents) / len(opponents)
        rows.append((len(converged) / len(opponents), label[model], avg_rounds, loops))
    for rate, name, avg_rounds, loops in sorted(rows, reverse=True):
        ranking.add_row(name, f"{rate:.0%}", f"{avg_rounds:.1f}", f"{loops:.0%}")
    console.print(ranking)

    longest = sorted((g for g in games if g["outcome"] == "converged" and g["model_a"] != g["model_b"]), key=lambda g: -g["rounds"])[:5]
    if longest:
        console.print("\n[bold]Longest roads to convergence[/bold]")
        for game in longest:
            steps = " → ".join(" + ".join(pair) if pair[0] != pair[1] else f"[green]{pair[0]}[/green]" for pair in game["path"])
            console.print(f"{label[game['model_a']]} vs {label[game['model_b']]} ({game['rounds']} rounds): {steps}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=Path("data/convergence.db"), help="Embeddings database from build_db.py.")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), help="Models to battle (default: every model in the database).")
    parser.add_argument("--games", type=int, default=100, help="Random starting pairs per pairing (default: 100).")
    parser.add_argument("--operator", default="centroid", choices=OPERATOR_NAMES, help="Combining operator (default: centroid).")
    parser.add_argument("--max-rounds", type=int, default=10, help="Round limit per game (default: 10).")
    parser.add_argument("--seed", type=int, default=0, help="Seed for picking the starting pairs (default: 0).")
    parser.add_argument("--starting-words", type=Path, default=Path("data/vocab_1000.txt"), help="Words to pick random starting pairs from.")
    parser.add_argument("--output-json", type=Path, help="Optional path to write every game's path and the summary.")
    args = parser.parse_args()

    conn = db.connect(args.db)
    stored = set(db.vocab_models(conn))
    models = [model for model in (args.models or MODELS) if model in stored]
    words = load_vocab(args.starting_words)
    rng = random.Random(args.seed)
    start_pairs = [tuple(rng.sample(words, 2)) for _ in range(args.games)]

    games = play_battle(conn, models, start_pairs, operator=args.operator, max_rounds=args.max_rounds)
    summary = summarize(games)
    print_report(models, games, summary, Console())

    if args.output_json:
        report = {"operator": args.operator, "max_rounds": args.max_rounds, "summary": [{"model_a": a, "model_b": b, **stats} for (a, b), stats in summary.items()], "games": games}
        args.output_json.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
