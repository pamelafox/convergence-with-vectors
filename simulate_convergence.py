#!/usr/bin/env python
"""Simulate the Convergence game: two models repeatedly apply one operator.

Starting from a word pair, each model independently applies the same
combining operator to the current pair. Their two outputs become the next
round's pair. The simulation stops when both models produce the same word
(convergence), a pair repeats a previous round's state (loop), or a round
limit is reached.

Example:

    python simulate_convergence.py --vocab data/sample_vocab.txt \\
        --word-a fire --word-b whale --operator centroid --max-rounds 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from embeddings import MODELS, get_vocab_embeddings
from operators import OPERATOR_NAMES, apply_operator
from reporting import ResultRow, candidate_rows, write_csv, write_json
from vocab import load_vocab


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vocab", required=True, type=Path, help="Path to a newline-delimited vocabulary file.")
    parser.add_argument("--word-a", required=True, help="Starting word A.")
    parser.add_argument("--word-b", required=True, help="Starting word B.")
    parser.add_argument("--operator", default="centroid", choices=OPERATOR_NAMES, help="Combining operator to simulate (default: centroid).")
    parser.add_argument("--template", default="{a} and {b}", help="Phrase template, only used when --operator textual.")
    parser.add_argument(
        "--models",
        nargs=2,
        default=list(MODELS.keys())[:2],
        metavar=("MODEL_A", "MODEL_B"),
        help="Exactly two model short names or Hugging Face ids, one per 'player' (default: both MiniLM models).",
    )
    parser.add_argument("--max-rounds", type=int, default=10, help="Stop after this many rounds if no convergence/loop (default: 10).")
    parser.add_argument("--top-k", type=int, default=5, help="Ranked alternatives to record per round per model (default: 5).")
    parser.add_argument("--include-inputs", action="store_true", help="Allow a round's own input words as possible outputs.")
    parser.add_argument("--cache-dir", type=Path, default=Path(".embedding_cache"), help="Directory for cached vocabulary embeddings.")
    parser.add_argument("--output-csv", type=Path, help="Optional path to write machine-readable CSV results.")
    parser.add_argument("--output-json", type=Path, help="Optional path to write machine-readable JSON results.")
    return parser


def simulate_convergence(
    vocab_path: Path,
    word_a: str,
    word_b: str,
    *,
    operator: str = "centroid",
    template: str = "{a} and {b}",
    models: list[str] | None = None,
    max_rounds: int = 10,
    top_k: int = 5,
    include_inputs: bool = False,
    cache_dir: Path = Path(".embedding_cache"),
) -> tuple[list[dict], str, list[ResultRow]]:
    """Run the convergence simulation.

    Returns:
        A tuple of ``(path, outcome, report_rows)`` where ``path`` is a list of
        per-round dicts (``round``, ``pair_in``, ``outputs``, ``pair_out``),
        ``outcome`` is one of "converged", "loop", or "round_limit", and
        ``report_rows`` holds every ranked candidate seen along the way.
    """
    models = models or list(MODELS.keys())[:2]
    if len(models) != 2:
        raise ValueError("Convergence simulation requires exactly two models")

    words = load_vocab(vocab_path)
    vocab_by_model = {m: get_vocab_embeddings(m, words, cache_dir) for m in models}

    current_pair = (word_a, word_b)
    visited: dict[tuple[str, str], int] = {current_pair: 0}
    path: list[dict] = [{"round": 0, "pair_in": current_pair, "outputs": None, "pair_out": current_pair}]
    report_rows: list[ResultRow] = []
    outcome = "round_limit"

    for round_number in range(1, max_rounds + 1):
        exclude = set() if include_inputs else set(current_pair)
        outputs: dict[str, str] = {}
        for model_name in models:
            candidates = apply_operator(operator, model_name, vocab_by_model[model_name], current_pair[0], current_pair[1], top_k=top_k, exclude=exclude, template=template)
            if not candidates:
                outcome = "stalled"
                break
            outputs[model_name] = candidates[0].candidate
            report_rows.extend(
                candidate_rows(
                    current_pair[0],
                    current_pair[1],
                    model_name,
                    operator,
                    candidates,
                    template=template if operator == "textual" else None,
                    round_number=round_number,
                )
            )
        if outcome == "stalled":
            break

        new_pair = (outputs[models[0]], outputs[models[1]])
        path.append({"round": round_number, "pair_in": current_pair, "outputs": outputs, "pair_out": new_pair})

        if outputs[models[0]] == outputs[models[1]]:
            outcome = "converged"
            current_pair = new_pair
            break
        if new_pair in visited:
            outcome = "loop"
            current_pair = new_pair
            break

        visited[new_pair] = round_number
        current_pair = new_pair
    else:
        outcome = "round_limit"

    for row in report_rows:
        row.outcome = outcome

    return path, outcome, report_rows


def describe_outcome(outcome: str, path: list[dict], models: list[str]) -> str:
    last = path[-1]
    if outcome == "converged":
        return f"Converged on '{last['pair_out'][0]}' after {last['round']} round(s)."
    if outcome == "loop":
        return f"Loop detected: pair {last['pair_out']} at round {last['round']} repeats an earlier round's pair."
    if outcome == "stalled":
        return f"Stalled at round {last['round']}: no candidates remained after exclusion."
    return f"Round limit reached without convergence or a loop (pair ended at {last['pair_out']})."


def print_path(path: list[dict], models: list[str], console: Console) -> None:
    table = Table(title="Convergence path")
    table.add_column("Round")
    table.add_column("Pair in")
    table.add_column(f"{models[0]} ->")
    table.add_column(f"{models[1]} ->")
    table.add_column("Pair out")
    for step in path:
        outputs = step["outputs"] or {}
        table.add_row(
            str(step["round"]),
            " / ".join(step["pair_in"]),
            outputs.get(models[0], ""),
            outputs.get(models[1], ""),
            " / ".join(step["pair_out"]),
        )
    console.print(table)


def main() -> None:
    args = build_arg_parser().parse_args()
    console = Console()
    console.print(
        f"Simulating convergence from [bold]{args.word_a}[/bold] + [bold]{args.word_b}[/bold] " f"using operator [bold]{args.operator}[/bold] with models {args.models[0]} / {args.models[1]}"
    )

    path, outcome, rows = simulate_convergence(
        args.vocab,
        args.word_a,
        args.word_b,
        operator=args.operator,
        template=args.template,
        models=args.models,
        max_rounds=args.max_rounds,
        top_k=args.top_k,
        include_inputs=args.include_inputs,
        cache_dir=args.cache_dir,
    )

    print_path(path, args.models, console)
    console.print(f"[bold]Outcome:[/bold] {outcome} -- {describe_outcome(outcome, path, args.models)}")

    if args.output_csv:
        write_csv(rows, args.output_csv)
        console.print(f"Wrote CSV results to [bold]{args.output_csv}[/bold]")
    if args.output_json:
        write_json(rows, args.output_json)
        console.print(f"Wrote JSON results to [bold]{args.output_json}[/bold]")


if __name__ == "__main__":
    main()
