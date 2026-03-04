"""Tests for the progress tracker."""

import pytest
from src.agents.progress import ProgressTracker


@pytest.fixture
def tracker():
    return ProgressTracker()


class TestProgressTracker:
    """Test progress recording and summarization."""

    def test_empty_progress(self, tracker):
        summary = tracker.get_progress_summary()
        assert "No topics studied yet" in summary

    def test_record_interaction(self, tracker):
        tracker.record_interaction("neural networks", "CONCEPT")
        assert "neural networks" in tracker.topics
        assert tracker.topics["neural networks"].interactions == 1

    def test_confidence_explored(self, tracker):
        tracker.record_interaction("CNNs", "CONCEPT")
        assert tracker.topics["CNNs"].confidence == "explored"

    def test_confidence_understood(self, tracker):
        for _ in range(3):
            tracker.record_interaction("backpropagation", "CONCEPT")
        assert tracker.topics["backpropagation"].confidence == "understood"

    def test_confidence_mastered(self, tracker):
        tracker.record_interaction("gradient descent", "CONCEPT")
        for _ in range(5):
            tracker.record_interaction("gradient descent", "QUIZ", quiz_correct=True)
        assert tracker.topics["gradient descent"].confidence == "mastered"

    def test_suggestions_for_new_learner(self, tracker):
        suggestions = tracker.get_suggested_topics()
        assert len(suggestions) > 0

    def test_suggestions_based_on_studied(self, tracker):
        tracker.record_interaction("neural networks", "CONCEPT")
        suggestions = tracker.get_suggested_topics()
        # Should suggest related topics
        assert any(
            t in suggestions
            for t in ["backpropagation", "activation functions", "loss functions"]
        )

    def test_interaction_log(self, tracker):
        tracker.record_interaction("RNNs", "CONCEPT")
        tracker.record_interaction("RNNs", "QUIZ", quiz_correct=True)
        assert len(tracker.interaction_log) == 2
        assert tracker.interaction_log[1]["quiz_correct"] is True
