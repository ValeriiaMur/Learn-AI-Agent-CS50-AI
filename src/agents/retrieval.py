"""Retrieval Agent — searches the knowledge base for relevant educational content."""

from src.rag.retriever import HybridRetriever


# Singleton retriever instance
_retriever = None


def get_retriever() -> HybridRetriever:
    """Get or create the hybrid retriever singleton."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def retrieve_context(query: str, k: int = 5) -> tuple[str, list]:
    """Retrieve relevant educational content for a query.

    Args:
        query: The user's question or topic
        k: Number of documents to retrieve

    Returns:
        Tuple of (formatted context string, raw document list)
    """
    retriever = get_retriever()
    documents = retriever.retrieve(query, k=k)
    context = retriever.format_context(documents)
    return context, documents
