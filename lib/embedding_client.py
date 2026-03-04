"""Context embedding client for dialogue history."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI
from sotopia.generation_utils import get_llm_call_log_path

# Default configuration (must match the model used for prompt space embeddings)
DEFAULT_EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"
DEFAULT_API_BASE = "https://openrouter.ai/api/v1"


class EmbeddingClient:
    """Client for generating embeddings from dialogue history."""

    def __init__(
        self,
        model: str = DEFAULT_EMBEDDING_MODEL,
        api_base: str = DEFAULT_API_BASE,
        api_key: str | None = None,
    ) -> None:
        """
        Initialize the embedding client.

        Args:
            model: Embedding model name
            api_base: API base URL
            api_key: API key (defaults to OPENROUTER_API_KEY env var)
        """
        self.model = model
        api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "API key not found. Please set OPENROUTER_API_KEY env var."
            )
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.gen_ids: list[str] = []  # Track gen_ids for cost calculation

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a single text.

        Args:
            text: The text to embed

        Returns:
            Embedding vector as numpy array
        """
        text = text.replace("\n", " ")
        response = self.client.embeddings.create(input=[text], model=self.model)

        # Extract gen_id from response for cost tracking
        if hasattr(response, "id") and response.id:
            self.gen_ids.append(response.id)

            # Log to LLM call log for unified cost calculation
            self._log_embedding_call(response)

        return np.array(response.data[0].embedding)

    def _log_embedding_call(self, response: Any) -> None:
        """Log embedding call to LLM call log for cost tracking."""
        log_path = get_llm_call_log_path()
        if not log_path:
            return

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "id": getattr(response, "id", None),
            "model": self.model,
            "caller": "embedding",
            "usage": {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) if hasattr(response, "usage") and response.usage else 0,
                "completion_tokens": 0,  # Embeddings don't have completion tokens
                "total_tokens": getattr(response.usage, "total_tokens", 0) if hasattr(response, "usage") and response.usage else 0,
            },
        }

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def extract_agent_history(messages: list[tuple[str, object]]) -> str:
    """
    Extract agent-only messages for context embedding.

    This excludes Environment messages and "did nothing" messages,
    returning only actual agent speech.

    Args:
        messages: List of (sender, message) tuples from the dialogue

    Returns:
        Formatted string of agent-only messages
    """
    agent_messages = []
    for sender, msg in messages:
        # Skip Environment messages
        if sender == "Environment":
            continue
        # Get natural language representation
        if hasattr(msg, "to_natural_language"):
            msg_text = msg.to_natural_language()
        else:
            msg_text = str(msg)
        # Skip "did nothing" messages
        if "did nothing" in msg_text:
            continue
        agent_messages.append(f"{sender}: {msg_text}")

    return "\n".join(agent_messages)
