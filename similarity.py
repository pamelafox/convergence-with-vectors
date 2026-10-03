"""Cosine similarity between one word and every word in the vocabulary.

Prints the most similar words and optionally draws a histogram of all the similarities as an SVG.

    uv run similarity.py
    uv run similarity.py fire --svg docs/slides_assets/fire_similarity_histogram.svg
"""

import argparse
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

from embeddings import MODELS, embed_texts, get_vocab_embeddings
from vocab import load_vocab


def histogram_svg(word: str, similarities: dict[str, float], marked: str) -> str:
    """Draw a histogram of similarities, with a dashed line at the ``marked`` word."""
    counts, edges = np.histogram(list(similarities.values()), bins=np.arange(0, 1.0001, 0.025))
    left, right, top, bottom = 100, 785, 60, 470
    ymax = int(np.ceil(counts.max() / 200) * 200)

    def sx(value: float) -> float:
        return left + value * (right - left)

    def sy(count: float) -> float:
        return bottom - count * (bottom - top) / ymax

    parts = []
    bar_width = sx(0.025) - left - 2
    for count, edge in zip(counts, edges[:-1]):
        if count:
            parts.append(f'<rect x="{sx(edge) + 1:.1f}" y="{sy(count):.1f}" width="{bar_width:.1f}" height="{bottom - sy(count):.1f}" fill="#3776ab"/>')
    for tick in range(0, ymax + 1, 200):
        parts.append(f'<line x1="{left - 6}" y1="{sy(tick):.1f}" x2="{left}" y2="{sy(tick):.1f}" stroke="#4b5563" stroke-width="1.5"/>')
        parts.append(f'<text x="{left - 10}" y="{sy(tick) + 9:.1f}" text-anchor="end">{tick}</text>')
    for tick in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
        parts.append(f'<line x1="{sx(tick):.1f}" y1="{bottom}" x2="{sx(tick):.1f}" y2="{bottom + 6}" stroke="#4b5563" stroke-width="1.5"/>')
        parts.append(f'<text x="{sx(tick):.1f}" y="{bottom + 34}" text-anchor="middle">{tick:.1f}</text>')
    x = sx(similarities[marked])
    parts.append(f'<line x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{bottom}" stroke="#c2410c" stroke-width="2" stroke-dasharray="6 5"/>')
    parts.append(f'<text x="{x:.1f}" y="{top - 16}" fill="#c2410c" text-anchor="middle" font-weight="bold">{marked}</text>')
    return (
        "\n".join(
            [
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 815 560" font-family="sans-serif" font-size="30" fill="#4b5563">',
                *parts,
                f'<path d="M{left},{top} V{bottom} H{right}" fill="none" stroke="#4b5563" stroke-width="1.5"/>',
                f'<text x="{(left + right) / 2}" y="552" text-anchor="middle" font-weight="bold">Cosine similarity to "{word}"</text>',
                "</svg>",
            ]
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("word", nargs="?", default="fire", help="Word to compare to the vocabulary (default: fire)")
    parser.add_argument("--model", default="minilm-l6", choices=list(MODELS), help="Model to use (default: minilm-l6)")
    parser.add_argument("--vocab", type=Path, default=Path("data/vocab_5000.txt"), help="Vocabulary file (default: data/vocab_5000.txt)")
    parser.add_argument("--top", type=int, default=5, help="How many of the most similar words to show (default: 5)")
    parser.add_argument("--svg", type=Path, help="Optional path to write a histogram of all the similarities")
    args = parser.parse_args()

    vocab = get_vocab_embeddings(args.model, load_vocab(args.vocab))
    query = embed_texts(args.model, [args.word])[0]
    similarities = {word: float(sim) for word, sim in zip(vocab.words, vocab.similarities_to(query)) if word != args.word}
    ranked = sorted(similarities, key=similarities.get, reverse=True)

    table = Table(title=f"Most similar to '{args.word}' ({MODELS[args.model].label}, {len(similarities)} words)")
    table.add_column("Word")
    table.add_column("Similarity", justify="right")
    for word in ranked[: args.top]:
        table.add_row(word, f"{similarities[word]:.2f}")
    Console().print(table)
    values = np.array(list(similarities.values()))
    Console().print(f"Median similarity: {np.median(values):.2f}; words above 0.5: {(values > 0.5).sum()}")

    if args.svg:
        args.svg.write_text(histogram_svg(args.word, similarities, ranked[0]))
        Console().print(f"Wrote {args.svg}")


if __name__ == "__main__":
    main()
