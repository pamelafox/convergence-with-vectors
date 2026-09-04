import argparse
from itertools import combinations

import numpy as np
from rich.console import Console
from rich.table import Table
from sentence_transformers import SentenceTransformer

MODEL_NAMES = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
]
DEFAULT_TEXTS = ["fire", "whale"]


def cosine_similarities(texts: list[str], model_name: str) -> tuple[int, list[tuple[str, str, float]]]:
    model = SentenceTransformer(model_name, device="cpu")
    embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    pairs = [(texts[first], texts[second], float(np.dot(embeddings[first], embeddings[second]))) for first, second in combinations(range(len(texts)), 2)]
    return embeddings.shape[1], pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare cosine similarities from two CPU-friendly MiniLM models.")
    parser.add_argument("texts", nargs="*", help="Text to compare (defaults to: fire whale)")
    args = parser.parse_args()
    texts = args.texts or DEFAULT_TEXTS

    if len(texts) < 2:
        parser.error("provide at least two texts to compare")

    console = Console()
    table = Table(title="MiniLM cosine similarities")
    table.add_column("Model")
    table.add_column("Dimensions", justify="right")
    table.add_column("Text A")
    table.add_column("Text B")
    table.add_column("Cosine similarity", justify="right")

    for model_name in MODEL_NAMES:
        console.print(f"Loading [bold]{model_name}[/bold] on CPU...")
        dimensions, similarities = cosine_similarities(texts, model_name)
        short_name = model_name.rsplit("/", maxsplit=1)[-1]
        for index, (text_a, text_b, similarity) in enumerate(similarities):
            table.add_row(short_name if index == 0 else "", str(dimensions) if index == 0 else "", text_a, text_b, f"{similarity:.4f}")

    console.print(table)


if __name__ == "__main__":
    main()
