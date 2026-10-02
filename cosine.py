"""Cosine similarity between words, computed with the full formula and with np.dot.

The first word is compared to each of the others. Optionally draws their angles as an SVG.

    uv run cosine.py
    uv run cosine.py fire flame --svg docs/slides_assets/cosine_angles.svg
"""

import argparse
import math
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from sentence_transformers import SentenceTransformer

from embeddings import MODELS

COLORS = ["#4b5563", "#c2410c", "#1d4ed8", "#047857", "#7c3aed"]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def angles_svg(base: str, similarities: dict[str, float]) -> str:
    """Draw each word as an arrow at its true angle from the base word (only angles to the base are exact)."""
    length, ox, oy = 280, 0, 0
    parts = []
    arrows = [(base, 0.0, COLORS[0])] + [(word, math.acos(max(-1.0, min(1.0, sim))), COLORS[(i + 1) % len(COLORS)]) for i, (word, sim) in enumerate(similarities.items())]
    for i, (word, angle, color) in enumerate(arrows[1:]):
        radius = 70 + 50 * i
        ex, ey = ox + radius * math.cos(angle), oy - radius * math.sin(angle)
        parts.append(f'<path d="M{ox + radius},{oy} A{radius},{radius} 0 0 0 {ex:.1f},{ey:.1f}" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="5 4"/>')
    tips = []
    for word, angle, color in arrows:
        x, y = ox + length * math.cos(angle), oy - length * math.sin(angle)
        tips.append((x, y))
        parts.append(f'<line x1="{ox}" y1="{oy}" x2="{x:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="4" marker-end="url(#tip)"/>')
        label = f'<tspan font-weight="bold">{word}</tspan>'
        if angle:
            label += f'<tspan x="{x + 14:.1f}" dy="34" font-size="28">cos {math.cos(angle):.2f} · {math.degrees(angle):.0f}°</tspan>'
        parts.append(f'<text x="{x + 14:.1f}" y="{y - (4 if angle else -8):.1f}" fill="{color}">{label}</text>')
    # Fit the canvas to the arrows plus room for their labels (about 210 units wide, 40 tall)
    left, top = min(0, *(x for x, _ in tips)) - 10, min(y for _, y in tips) - 40
    right, bottom = max(x for x, _ in tips) + 220, 20
    header = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{left:.0f} {top:.0f} {right - left:.0f} {bottom - top:.0f}" font-family="sans-serif" font-size="34">',
        '<defs><marker id="tip" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker></defs>',
    ]
    return "\n".join(header + parts + ["</svg>"]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("words", nargs="*", default=["fire", "flame"], help="Words to compare to the first word (default: fire flame)")
    parser.add_argument("--model", default="minilm-l6", choices=list(MODELS), help="Model to use (default: minilm-l6)")
    parser.add_argument("--svg", type=Path, help="Optional path to write a diagram of the angles")
    args = parser.parse_args()

    model = SentenceTransformer(MODELS[args.model].model_id, device="cpu")
    vectors = model.encode(args.words, normalize_embeddings=True)
    base, base_vector = args.words[0], vectors[0]

    table = Table(title=f"Cosine similarity to '{base}' ({MODELS[args.model].label})")
    for column in ("Word", "Full formula", "np.dot (normalized)", "Angle"):
        table.add_column(column, justify="right")
    similarities = {}
    for word, vector in zip(args.words[1:], vectors[1:], strict=True):
        similarities[word] = cosine_similarity(base_vector, vector)
        table.add_row(word, f"{similarities[word]:.3f}", f"{np.dot(base_vector, vector):.3f}", f"{math.degrees(math.acos(similarities[word])):.0f}°")
    Console().print(table)

    if args.svg:
        args.svg.write_text(angles_svg(base, similarities))
        Console().print(f"Wrote {args.svg}")


if __name__ == "__main__":
    main()
