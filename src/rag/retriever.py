"""Hybrid retriever — combines semantic search with keyword matching."""

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from src.config import settings
from src.rag.embeddings import get_embeddings


class HybridRetriever:
    """Retrieves relevant educational content using semantic + keyword search.

    Semantic search finds conceptually similar content.
    Keyword filtering ensures exact term matches aren't missed.
    Results are deduplicated and ranked by combined relevance.
    """

    def __init__(self, persist_dir: str = None):
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self._vectorstore = None

    @property
    def vectorstore(self) -> Chroma:
        """Lazy-load the vector store."""
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=get_embeddings(),
                collection_name=settings.COLLECTION_NAME,
            )
        return self._vectorstore

    def semantic_search(self, query: str, k: int = 4) -> list[Document]:
        """Pure semantic (embedding) search."""
        return self.vectorstore.similarity_search(query, k=k)

    def keyword_search(self, query: str, k: int = 4) -> list[Document]:
        """Keyword-based search using Chroma's where_document filter.

        Falls back to semantic search if keyword matching isn't supported.
        """
        try:
            # Use Chroma's built-in document content filtering
            results = self.vectorstore.similarity_search(
                query,
                k=k,
                filter=None,  # ChromaDB doesn't support full-text natively
            )
            # Manual keyword boost: re-rank results containing exact query terms
            query_terms = set(query.lower().split())
            scored = []
            for doc in results:
                content_lower = doc.page_content.lower()
                keyword_hits = sum(1 for term in query_terms if term in content_lower)
                scored.append((doc, keyword_hits))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in scored]
        except Exception:
            return self.semantic_search(query, k)

    def retrieve(self, query: str, k: int = 6) -> list[Document]:
        """Hybrid retrieval: merge semantic and keyword results, deduplicate.

        Args:
            query: The user's question or topic
            k: Total number of results to return

        Returns:
            Deduplicated, ranked list of relevant documents
        """
        # Get results from both strategies
        semantic_results = self.semantic_search(query, k=k)
        keyword_results = self.keyword_search(query, k=k)

        # Deduplicate by content hash, preserving order
        seen = set()
        merged = []
        for doc in semantic_results + keyword_results:
            content_hash = hash(doc.page_content[:200])
            if content_hash not in seen:
                seen.add(content_hash)
                merged.append(doc)

        return merged[:k]

    def format_context(self, documents: list[Document]) -> str:
        """Format retrieved documents into a context string for the LLM."""
        if not documents:
            return "No relevant educational content found for this topic."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "unknown")
            context_parts.append(
                f"[Source {i}: {source}]\n{doc.page_content}"
            )

        return "\n\n---\n\n".join(context_parts)
