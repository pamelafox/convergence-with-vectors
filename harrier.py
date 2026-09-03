import argparse

from rich.console import Console
from sentence_transformers import SentenceTransformer

MODEL_NAME = "microsoft/harrier-oss-v1-270m"
DEFAULT_TEXTS = ["fire", "whale"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute text embeddings with Microsoft Harrier.")
    parser.add_argument("texts", nargs="*", help="Text to embed (defaults to: fire whale)")
    args = parser.parse_args()
    texts = args.texts or DEFAULT_TEXTS

    console = Console()
    console.print(f"Loading [bold]{MODEL_NAME}[/bold]...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, normalize_embeddings=True)

    console.print(f"Computed {len(embeddings)} embeddings with {embeddings.shape[1]} dimensions:")
    for text, embedding in zip(texts, embeddings, strict=True):
        console.print(f"[bold]{text}[/bold]: {embedding[:8].round(4)} ...")


if __name__ == "__main__":
    main()
