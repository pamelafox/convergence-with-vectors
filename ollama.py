import argparse

import numpy as np
from openai import OpenAI
from rich.console import Console

MODEL_NAME = "nomic-embed-text"
DEFAULT_TEXTS = ["fire", "whale"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute text embeddings with a local Ollama model.")
    parser.add_argument("texts", nargs="*", help="Text to embed (defaults to: fire whale)")
    parser.add_argument("--model", default=MODEL_NAME, help=f"Ollama model to use (default: {MODEL_NAME})")
    args = parser.parse_args()
    texts = args.texts or DEFAULT_TEXTS

    console = Console()
    console.print(f"Computing embeddings with [bold]{args.model}[/bold]...")
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    response = client.embeddings.create(model=args.model, input=texts)
    embeddings = np.array([item.embedding for item in response.data])
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    console.print(f"Computed {len(embeddings)} embeddings with {embeddings.shape[1]} dimensions:")
    for text, embedding in zip(texts, embeddings, strict=True):
        console.print(f"[bold]{text}[/bold]: {embedding[:8].round(4)} ...")


if __name__ == "__main__":
    main()
