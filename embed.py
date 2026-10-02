"""Embed words with one model, then compare every pair with cosine similarity.

uv run embed.py fire whale flame
uv run embed.py fire whale flame --model qwen3-embedding-0.6b
"""

import argparse
from itertools import combinations

from rich.console import Console
from rich.table import Table
from sentence_transformers import SentenceTransformer

from embeddings import MODELS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("texts", nargs="*", default=["fire", "whale", "flame"], help="Words to embed (default: fire whale flame)")
    parser.add_argument("--model", default="minilm-l6", choices=list(MODELS), help="Model to use (default: minilm-l6)")
    args = parser.parse_args()

    console = Console()
    model = SentenceTransformer(MODELS[args.model].model_id, device="cpu")
    vectors = model.encode(args.texts, normalize_embeddings=True)

    console.print(f"[bold]{MODELS[args.model].label}[/bold]: {vectors.shape[1]} dimensions")
    for text, vector in zip(args.texts, vectors, strict=True):
        console.print(f"{text}: {vector[:6].round(3)} ...")

    table = Table(title="Cosine similarity")
    table.add_column("Word A")
    table.add_column("Word B")
    table.add_column("Similarity", justify="right")
    for i, j in combinations(range(len(args.texts)), 2):
        # Normalized vectors, so the dot product is the cosine similarity
        table.add_row(args.texts[i], args.texts[j], f"{vectors[i] @ vectors[j]:.3f}")
    console.print(table)


if __name__ == "__main__":
    main()
