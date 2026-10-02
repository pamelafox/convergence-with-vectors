#!/usr/bin/env python
"""Build the SQLite database of vocabulary embeddings used by the FastAPI backend.

Embeds every vocabulary word with each model and stores the vectors in a
single ``embeddings`` table. Re-running replaces each model's vocabulary
but keeps any embeddings the backend has cached on demand.

Example:

    uv run build_db.py --vocab data/vocab_5000.txt --db data/convergence.db
"""

from __future__ import annotations

import argparse
from pathlib import Path

from backend import db
from embeddings import MODELS, get_vocab_embeddings, is_available
from vocab import load_vocab


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vocab", type=Path, default=Path("data/vocab_5000.txt"), help="Path to a newline-delimited vocabulary file (default: data/vocab_5000.txt).")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS), help="Models to embed the vocabulary with (default: every model that's available).")
    parser.add_argument("--db", type=Path, default=Path("data/convergence.db"), help="SQLite database to write (default: data/convergence.db).")
    parser.add_argument("--cache-dir", type=Path, default=Path(".embedding_cache"), help="Directory for cached vocabulary embeddings.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    words = load_vocab(args.vocab)
    args.db.parent.mkdir(parents=True, exist_ok=True)
    conn = db.connect(args.db)
    try:
        for model_name in args.models:
            # Ollama models are skipped where no Ollama server is running, like the container image build
            if not is_available(model_name):
                print(f"Skipping {model_name}: not available (is Ollama running, with the model pulled?)")
                continue
            vocab = get_vocab_embeddings(model_name, words, args.cache_dir)
            db.store_vocab(conn, model_name, vocab.words, vocab.vectors)
            print(f"Stored {len(words)} words for {model_name}")
    finally:
        conn.close()
    print(f"Wrote {args.db}")


if __name__ == "__main__":
    main()
