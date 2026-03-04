#!/usr/bin/env python3
"""
Script to get embedding for a given text using OpenRouter/OpenAI API.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / ".env")

DEFAULT_MODEL = "qwen/qwen3-embedding-4b"
DEFAULT_API_BASE = "https://openrouter.ai/api/v1"

def get_embedding(client: OpenAI, text: str, model: str) -> np.ndarray:
    """Get embedding for a single text."""
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding)

def main():
    parser = argparse.ArgumentParser(description="Get embedding for a text string.")
    parser.add_argument("text", nargs="?", help="The text to embed. Can also be provided via stdin.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Embedding model (default: {DEFAULT_MODEL})")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help=f"API base URL (default: {DEFAULT_API_BASE})")
    parser.add_argument("--api-key", help="API key (overrides OPENROUTER_API_KEY env var)")

    args = parser.parse_args()

    # Get text from argument or stdin
    text = args.text
    if not text:
        if not sys.stdin.isatty():
            text = sys.stdin.read().strip()
    
    if not text:
        parser.error("No text provided. Please provide text as an argument or via stdin.")

    # Setup client
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: API key not found. Please set OPENROUTER_API_KEY env var or use --api-key.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(
        api_key=api_key,
        base_url=args.api_base,
    )

    try:
        embedding = get_embedding(client, text, args.model)
        print(embedding)
        print(embedding.shape)
    except Exception as e:
        print(f"Error getting embedding: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
