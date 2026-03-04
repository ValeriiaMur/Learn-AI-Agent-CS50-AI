"""Tests for the tiered model router."""

import pytest
from src.utils.model_router import ModelRouter


@pytest.fixture
def router():
    return ModelRouter()


class TestComplexityClassification:
    """Test the heuristic complexity classifier."""

    def test_simple_definition(self, router):
        assert router.classify_complexity("CONCEPT", "What is a neural network?") == "simple"

    def test_simple_define(self, router):
        assert router.classify_complexity("CONCEPT", "Define backpropagation") == "simple"

    def test_standard_concept(self, router):
        assert router.classify_complexity("CONCEPT", "How do transformers handle long-range dependencies in sequences?") == "standard"

    def test_complex_deeper(self, router):
        assert router.classify_complexity("DEEPER", "Explain attention in detail") == "complex"

    def test_complex_analogy(self, router):
        assert router.classify_complexity("CONCEPT", "Give me an analogy for how gradient descent works") == "complex"

    def test_complex_compare(self, router):
        assert router.classify_complexity("CONCEPT", "Compare and contrast CNNs and transformers for image tasks") == "complex"

    def test_quiz_is_standard(self, router):
        assert router.classify_complexity("QUIZ", "Quiz me on neural networks") == "standard"

    def test_progress_is_standard(self, router):
        assert router.classify_complexity("PROGRESS", "What have I learned?") == "standard"
