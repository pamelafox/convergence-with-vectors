#!/usr/bin/env python
"""Compare word-combination operators for a single pair of words, across models.

Example:

    python compare_pair.py --vocab data/sample_vocab.txt --word-a snow --word-b mountain --top-k 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from embeddings import MODELS, get_vocab_embeddings
from operators import DEFAULT_TEMPLATES, OPERATOR_NAMES, apply_operator
from reporting import ResultRow, candidate_rows, render_side_by_side, write_csv, write_json
from vocab import load_vocab


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vocab", required=True, type=Path, help="Path to a newline-delimited vocabulary file.")
    parser.add_argument("--word-a", required=True, help="First input word.")
    parser.add_argument("--word-b", required=True, help="Second input word.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of ranked candidates to show per operator (default: 10).")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS.keys()),
        help=f"Model short names or Hugging Face ids to compare (default: {' '.join(MODELS.keys())}).",
    )
    parser.add_argument(
        "--operators",
        nargs="+",
        default=OPERATOR_NAMES,
        choices=OPERATOR_NAMES,
        help=f"Operators to run (default: all of {OPERATOR_NAMES}).",
    )
    parser.add_argument(
        "--template",
        dest="templates",
        action="append",
        help="Phrase template for the 'textual' operator (use '{a}'/'{b}' placeholders); " f"repeatable. Default templates: {DEFAULT_TEMPLATES!r}.",
    )
    parser.add_argument(
        "--include-inputs",
        action="store_true",
        help="Include word-a/word-b themselves as possible candidates (excluded by default).",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(".embedding_cache"), help="Directory for cached vocabulary embeddings.")
    parser.add_argument("--output-csv", type=Path, help="Optional path to write machine-readable CSV results.")
    parser.add_argument("--output-json", type=Path, help="Optional path to write machine-readable JSON results.")
    return parser


def run_pair_comparison(
    vocab_path: Path,
    word_a: str,
    word_b: str,
    *,
    top_k: int = 10,
    models: list[str] | None = None,
    operators: list[str] | None = None,
    templates: list[str] | None = None,
    include_inputs: bool = False,
    cache_dir: Path = Path(".embedding_cache"),
) -> list[ResultRow]:
    """Run every requested operator (and template) for every model. Returns flat report rows."""
    models = models or list(MODELS.keys())
    operators = operators or OPERATOR_NAMES
    templates = templates or DEFAULT_TEMPLATES
    exclude = set() if include_inputs else {word_a, word_b}

    words = load_vocab(vocab_path)
    rows: list[ResultRow] = []
    for model_name in models:
        vocab_emb = get_vocab_embeddings(model_name, words, cache_dir)
        for operator_name in operators:
            if operator_name == "textual":
                for template in templates:
                    candidates = apply_operator(operator_name, model_name, vocab_emb, word_a, word_b, top_k=top_k, exclude=exclude, template=template)
                    rows.extend(candidate_rows(word_a, word_b, model_name, operator_name, candidates, template=template))
            else:
                candidates = apply_operator(operator_name, model_name, vocab_emb, word_a, word_b, top_k=top_k, exclude=exclude)
                rows.extend(candidate_rows(word_a, word_b, model_name, operator_name, candidates))
    return rows


def main() -> None:
    args = build_arg_parser().parse_args()
    console = Console()
    console.print(f"Comparing operators for [bold]{args.word_a}[/bold] + [bold]{args.word_b}[/bold] " f"across models: {', '.join(args.models)}")

    rows = run_pair_comparison(
        args.vocab,
        args.word_a,
        args.word_b,
        top_k=args.top_k,
        models=args.models,
        operators=args.operators,
        templates=args.templates,
        include_inputs=args.include_inputs,
        cache_dir=args.cache_dir,
    )

    # Group rows by (operator, template) so each group can be rendered as a
    # side-by-side table, one column per model.
    groups: dict[tuple[str, str | None], dict[str, list[ResultRow]]] = {}
    for row in rows:
        key = (row.operator, row.template)
        groups.setdefault(key, {m: [] for m in args.models})
        groups[key][row.model].append(row)

    for (operator_name, template), rows_by_model in groups.items():
        title = f"{operator_name}" + (f'  template="{template}"' if template else "")
        render_side_by_side(rows_by_model, console=console, title=title)

    if args.output_csv:
        write_csv(rows, args.output_csv)
        console.print(f"Wrote CSV results to [bold]{args.output_csv}[/bold]")
    if args.output_json:
        write_json(rows, args.output_json)
        console.print(f"Wrote JSON results to [bold]{args.output_json}[/bold]")


if __name__ == "__main__":
    main()
