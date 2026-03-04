"""Configuration and settings for LearnAI Agent."""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings loaded from environment."""

    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Model routing tiers
    PRIMARY_MODEL: str = os.getenv("PRIMARY_MODEL", "gpt-4o-mini")
    PREMIUM_MODEL: str = os.getenv("PREMIUM_MODEL", "gpt-4o")
    LOCAL_MODEL: str = os.getenv("LOCAL_MODEL", "llama3")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Vector DB
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    COLLECTION_NAME: str = "learnai_docs"

    # Embeddings
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Chunking
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Data
    DATA_DIR: str = os.getenv("DATA_DIR", "./data/sample_docs")

    # LangSmith
    LANGCHAIN_TRACING: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"


settings = Settings()
