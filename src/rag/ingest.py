"""Document ingestion pipeline — chunks, embeds, and stores educational content."""

import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from src.config import settings
from src.rag.embeddings import get_embeddings


def load_documents(data_dir: str = None):
    """Load all supported documents from the data directory.

    Supports: .md, .txt files. Extend with PDF loaders as needed.
    """
    data_dir = data_dir or settings.DATA_DIR
    documents = []

    # Markdown files (using TextLoader — lightweight, no NLTK dependency)
    for filepath in glob.glob(os.path.join(data_dir, "**/*.md"), recursive=True):
        try:
            loader = TextLoader(filepath, encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = os.path.basename(filepath)
                doc.metadata["file_type"] = "markdown"
            documents.extend(docs)
            print(f"  Loaded: {filepath} ({len(docs)} sections)")
        except Exception as e:
            print(f"  Error loading {filepath}: {e}")

    # Text files
    for filepath in glob.glob(os.path.join(data_dir, "**/*.txt"), recursive=True):
        try:
            loader = TextLoader(filepath)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = os.path.basename(filepath)
                doc.metadata["file_type"] = "text"
            documents.extend(docs)
            print(f"  Loaded: {filepath} ({len(docs)} sections)")
        except Exception as e:
            print(f"  Error loading {filepath}: {e}")

    return documents


def chunk_documents(documents, chunk_size: int = None, chunk_overlap: int = None):
    """Split documents into chunks optimized for retrieval.

    Uses recursive character splitting with markdown-aware separators.
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)
    print(f"  Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks


def create_vector_store(chunks):
    """Create or update ChromaDB vector store with document chunks."""
    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=settings.CHROMA_PERSIST_DIR,
        collection_name=settings.COLLECTION_NAME,
    )

    print(f"  Stored {len(chunks)} chunks in ChromaDB at {settings.CHROMA_PERSIST_DIR}")
    return vectorstore


def run_ingestion(data_dir: str = None):
    """Full ingestion pipeline: load → chunk → embed → store."""
    print("\n=== LearnAI Agent — Document Ingestion ===\n")

    print("1. Loading documents...")
    documents = load_documents(data_dir)
    if not documents:
        print("  No documents found. Add .md or .txt files to data/sample_docs/")
        return None

    print(f"\n2. Chunking {len(documents)} documents...")
    chunks = chunk_documents(documents)

    print("\n3. Creating vector store...")
    vectorstore = create_vector_store(chunks)

    print("\n=== Ingestion complete! ===\n")
    return vectorstore


# Allow running as: python -m src.rag.ingest
if __name__ == "__main__":
    run_ingestion()
