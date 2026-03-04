"""Tiered LLM routing — deterministic → mid-tier → premium fallback.

This implements cost-optimized model selection:
- Tier 1 (Local/Free): Simple definitions, glossary lookups → Llama via Ollama
- Tier 2 (Mid-tier): Standard explanations, quizzes → GPT-4o-mini
- Tier 3 (Premium): Complex reasoning, analogies, multi-step explanations → GPT-4o
"""

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from src.config import settings


class ModelRouter:
    """Routes requests to the appropriate LLM tier based on task complexity."""

    def __init__(self):
        self._models = {}

    def _get_primary(self) -> ChatOpenAI:
        """Mid-tier model for standard tasks."""
        if "primary" not in self._models:
            self._models["primary"] = ChatOpenAI(
                model=settings.PRIMARY_MODEL,
                temperature=0.3,
                api_key=settings.OPENAI_API_KEY,
            )
        return self._models["primary"]

    def _get_premium(self) -> ChatOpenAI:
        """Premium model for complex reasoning."""
        if "premium" not in self._models:
            self._models["premium"] = ChatOpenAI(
                model=settings.PREMIUM_MODEL,
                temperature=0.4,
                api_key=settings.OPENAI_API_KEY,
            )
        return self._models["premium"]

    def _get_local(self) -> ChatOllama | None:
        """Local model for simple lookups. Returns None if unavailable."""
        if "local" not in self._models:
            try:
                model = ChatOllama(
                    model=settings.LOCAL_MODEL,
                    base_url=settings.OLLAMA_BASE_URL,
                    temperature=0.1,
                )
                # Quick health check
                model.invoke("test")
                self._models["local"] = model
            except Exception:
                self._models["local"] = None
        return self._models["local"]

    def get_model(self, complexity: str = "standard"):
        """Get the appropriate model for the task complexity.

        Args:
            complexity: One of "simple", "standard", "complex"

        Returns:
            A LangChain chat model instance.
        """
        if complexity == "simple":
            local = self._get_local()
            if local is not None:
                return local
            # Fall through to mid-tier if local unavailable
            return self._get_primary()

        if complexity == "complex":
            return self._get_premium()

        # Default: mid-tier
        return self._get_primary()

    def classify_complexity(self, intent: str, message: str) -> str:
        """Determine task complexity from intent and message.

        Simple heuristic — can be replaced with a classifier.

        Args:
            intent: The classified intent (CONCEPT, QUIZ, DEEPER, etc.)
            message: The user's raw message

        Returns:
            Complexity level: "simple", "standard", or "complex"
        """
        # Simple: short definition-style questions
        simple_signals = ["what is", "define", "what does", "meaning of"]
        if intent == "CONCEPT" and any(s in message.lower() for s in simple_signals):
            if len(message.split()) < 15:
                return "simple"

        # Complex: deep explanations, analogies, multi-part questions
        complex_signals = [
            "explain in detail", "how does it work internally",
            "compare and contrast", "what's the intuition",
            "walk me through", "why does", "prove that",
            "analogy", "step by step",
        ]
        if intent == "DEEPER" or any(s in message.lower() for s in complex_signals):
            return "complex"

        return "standard"


# Singleton
model_router = ModelRouter()
