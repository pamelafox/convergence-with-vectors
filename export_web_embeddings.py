#!/usr/bin/env python
"""Export precomputed vocabulary embeddings as static JSON for the browser game.

The browser can't run sentence-transformers, so this script does the one
expensive step (embedding the closed vocabulary) offline and writes the
resulting normalized vectors to JSON files that ``web/convergence.js`` can
simply ``fetch``. Because the vocabulary is closed and the web game only ever
scores vocabulary words against other vocabulary words, no model inference is
needed in the browser at all.

Example:

    python export_web_embeddings.py --vocab data/vocab_1000.txt \\
        --models minilm-l6 minilm-l12 --output-dir web/data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from embeddings import MODELS, get_vocab_embeddings
from vocab import load_vocab


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vocab", required=True, type=Path, help="Path to a newline-delimited vocabulary file.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS.keys()),
        help="Model short names to export (default: all models in embeddings.MODELS).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("web/data"), help="Directory to write per-model JSON files to (default: web/data).")
    parser.add_argument("--cache-dir", type=Path, default=Path(".embedding_cache"), help="Directory for cached vocabulary embeddings.")
    parser.add_argument("--precision", type=int, default=6, help="Decimal places to round exported vector components to (default: 6).")
    return parser


def export_model(model_name: str, words: list[str], cache_dir: Path, output_dir: Path, precision: int) -> Path:
    """Compute and write one model's vocabulary embeddings as JSON.

    Returns the path the JSON file was written to.
    """
    vocab = get_vocab_embeddings(model_name, words, cache_dir)
    payload = {
        "model": model_name,
        "words": vocab.words,
        "vectors": [[round(float(x), precision) for x in row] for row in vocab.vectors],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{model_name}.json"
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    return out_path


def main() -> None:
    args = build_arg_parser().parse_args()
    words = load_vocab(args.vocab)

    manifest = {"words_count": len(words), "models": []}
    for model_name in args.models:
        out_path = export_model(model_name, words, args.cache_dir, args.output_dir, args.precision)
        print(f"Wrote {out_path} ({len(words)} words)")
        manifest["models"].append(model_name)

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
