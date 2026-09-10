import argparse
from itertools import combinations

import numpy as np
from rich.console import Console
from rich.table import Table

from embeddings import MODELS, embed_texts

MODEL_NAMES = list(MODELS)
DEFAULT_TEXTS = ["fire", "whale"]


def compute_similarities(texts: list[str], model_name: str) -> tuple[int, list[tuple[str, str, float]]]:
    embeddings = embed_texts(model_name, texts)

    pairs: list[tuple[str, str, float]] = []
    for first, second in combinations(range(len(texts)), 2):
        first_vector, second_vector = embeddings[first], embeddings[second]
        norms_product = np.linalg.norm(first_vector) * np.linalg.norm(second_vector)
        similarity = float(np.dot(first_vector, second_vector) / norms_product) if norms_product else 0.0
        pairs.append((texts[first], texts[second], similarity))

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
        dimensions, similarities = compute_similarities(texts, model_name)
        short_name = model_name.rsplit("/", maxsplit=1)[-1]
        for index, (text_a, text_b, similarity) in enumerate(similarities):
            table.add_row(short_name if index == 0 else "", str(dimensions) if index == 0 else "", text_a, text_b, f"{similarity:.4f}")

    console.print(table)


if __name__ == "__main__":
    main()
