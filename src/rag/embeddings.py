"""Embedding utilities for document vectorization."""

from langchain_openai import OpenAIEmbeddings
from src.config import settings


def get_embeddings() -> OpenAIEmbeddings:
    """Get the configured embedding model.

    Uses OpenAI's text-embedding-3-small by default —
    good balance of quality, speed, and cost.
    """
    return OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY,
    )
