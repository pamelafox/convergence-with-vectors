#!/usr/bin/env python
"""Run word-combination operator comparisons for a batch of word pairs.

Example:

    python compare_batch.py --vocab data/sample_vocab.txt --pairs data/sample_pairs.csv \\
        --output-csv batch_results.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from rich.console import Console

from compare_pair import run_pair_comparison
from embeddings import MODELS
from operators import DEFAULT_TEMPLATES, OPERATOR_NAMES
from reporting import ResultRow, render_flat_table, write_csv, write_json


def load_pairs(path: str | Path) -> list[tuple[str, str]]:
    """Read word pairs from a CSV file with ``word_a``/``word_b`` columns."""
    pairs: list[tuple[str, str]] = []
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "word_a" not in reader.fieldnames or "word_b" not in reader.fieldnames:
            raise ValueError(f"{path} must have 'word_a' and 'word_b' columns")
        for record in reader:
            word_a, word_b = record["word_a"].strip(), record["word_b"].strip()
            if word_a and word_b:
                pairs.append((word_a, word_b))
    if not pairs:
        raise ValueError(f"No usable word pairs found in {path}")
    return pairs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vocab", required=True, type=Path, help="Path to a newline-delimited vocabulary file.")
    parser.add_argument("--pairs", required=True, type=Path, help="CSV file with 'word_a' and 'word_b' columns.")
    parser.add_argument("--top-k", type=int, default=5, help="Ranked candidates per operator per pair (default: 5).")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()), help="Model short names or Hugging Face ids to compare.")
    parser.add_argument("--operators", nargs="+", default=OPERATOR_NAMES, choices=OPERATOR_NAMES, help="Operators to run.")
    parser.add_argument("--template", dest="templates", action="append", help="Phrase template(s) for the 'textual' operator; repeatable.")
    parser.add_argument("--include-inputs", action="store_true", help="Include each pair's own words as possible candidates.")
    parser.add_argument("--cache-dir", type=Path, default=Path(".embedding_cache"), help="Directory for cached vocabulary embeddings.")
    parser.add_argument("--output-csv", type=Path, default=Path("batch_results.csv"), help="Path to write combined CSV results.")
    parser.add_argument("--output-json", type=Path, help="Optional path to write combined JSON results.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    console = Console()
    pairs = load_pairs(args.pairs)
    console.print(f"Running {len(pairs)} pair(s) x {len(args.models)} model(s) x {len(args.operators)} operator(s)...")

    all_rows: list[ResultRow] = []
    for word_a, word_b in pairs:
        all_rows.extend(
            run_pair_comparison(
                args.vocab,
                word_a,
                word_b,
                top_k=args.top_k,
                models=args.models,
                operators=args.operators,
                templates=args.templates or DEFAULT_TEMPLATES,
                include_inputs=args.include_inputs,
                cache_dir=args.cache_dir,
            )
        )

    # Concise terminal view: just the top-ranked candidate per pair/model/operator/template.
    winners = [row for row in all_rows if row.rank == 1]
    render_flat_table(winners, console=console, title="Top candidate per pair / model / operator")

    write_csv(all_rows, args.output_csv)
    console.print(f"Wrote {len(all_rows)} rows of CSV results to [bold]{args.output_csv}[/bold]")
    if args.output_json:
        write_json(all_rows, args.output_json)
        console.print(f"Wrote JSON results to [bold]{args.output_json}[/bold]")


if __name__ == "__main__":
    main()
