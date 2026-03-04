"""Progress Tracker — maintains a simple knowledge model of the learner's session."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TopicRecord:
    """A single topic the learner has interacted with."""
    topic: str
    interactions: int = 0
    quiz_attempts: int = 0
    quiz_correct: int = 0
    first_seen: str = field(default_factory=lambda: datetime.now().isoformat())
    last_seen: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def confidence(self) -> str:
        """Estimate learner confidence level."""
        if self.quiz_attempts > 0 and self.quiz_correct / self.quiz_attempts >= 0.8:
            return "mastered"
        if self.interactions >= 3 or (self.quiz_attempts > 0 and self.quiz_correct > 0):
            return "understood"
        return "explored"


class ProgressTracker:
    """Tracks what the learner has studied and how well they understand it.

    Stores session-level progress in memory. Could be extended with
    persistent storage (SQLite, Redis) for multi-session tracking.
    """

    def __init__(self):
        self.topics: dict[str, TopicRecord] = {}
        self.interaction_log: list[dict] = []

    def record_interaction(self, topic: str, intent: str, quiz_correct: bool = None):
        """Record a learner interaction with a topic.

        Args:
            topic: The AI/ML topic discussed
            intent: The intent type (CONCEPT, QUIZ, DEEPER, etc.)
            quiz_correct: If this was a quiz, whether they got it right
        """
        if topic not in self.topics:
            self.topics[topic] = TopicRecord(topic=topic)

        record = self.topics[topic]
        record.interactions += 1
        record.last_seen = datetime.now().isoformat()

        if intent == "QUIZ" and quiz_correct is not None:
            record.quiz_attempts += 1
            if quiz_correct:
                record.quiz_correct += 1

        self.interaction_log.append({
            "topic": topic,
            "intent": intent,
            "quiz_correct": quiz_correct,
            "timestamp": datetime.now().isoformat(),
        })

    def get_progress_summary(self) -> str:
        """Generate a human-readable progress summary."""
        if not self.topics:
            return "No topics studied yet. Ask me about any AI/ML concept to get started!"

        lines = []
        mastered = []
        understood = []
        explored = []
        needs_review = []

        for topic, record in self.topics.items():
            confidence = record.confidence
            if confidence == "mastered":
                mastered.append(topic)
            elif confidence == "understood":
                understood.append(topic)
            else:
                explored.append(topic)

            # Flag topics with failed quizzes
            if record.quiz_attempts > 0 and record.quiz_correct / record.quiz_attempts < 0.5:
                needs_review.append(topic)

        if mastered:
            lines.append(f"Mastered: {', '.join(mastered)}")
        if understood:
            lines.append(f"Good understanding: {', '.join(understood)}")
        if explored:
            lines.append(f"Explored (keep going!): {', '.join(explored)}")
        if needs_review:
            lines.append(f"Needs review: {', '.join(needs_review)}")

        lines.append(f"\nTotal interactions: {len(self.interaction_log)}")
        lines.append(f"Topics covered: {len(self.topics)}")

        return "\n".join(lines)

    def get_suggested_topics(self) -> list[str]:
        """Suggest related topics based on CS50 AI curriculum.

        Uses a topic adjacency graph aligned to the 7 course modules:
        Search, Knowledge, Uncertainty, Optimization, Learning,
        Neural Networks, and Language.
        """
        topic_graph = {
            # Search module
            "bfs": ["DFS", "A* search", "heuristic functions"],
            "dfs": ["BFS", "depth-limited search", "minimax"],
            "a* search": ["heuristic functions", "greedy best-first search"],
            "minimax": ["alpha-beta pruning", "evaluation functions"],
            "alpha-beta pruning": ["minimax", "game trees"],
            # Knowledge module
            "propositional logic": ["inference rules", "entailment", "resolution"],
            "inference": ["modus ponens", "resolution", "model checking"],
            "first-order logic": ["universal quantifier", "existential quantifier"],
            "resolution": ["conjunctive normal form", "proof by contradiction"],
            # Uncertainty module
            "probability": ["Bayes' Rule", "conditional probability", "Bayesian networks"],
            "bayesian networks": ["Markov models", "inference", "sampling"],
            "markov models": ["Hidden Markov Models", "Markov chains"],
            "hidden markov models": ["Viterbi algorithm", "filtering"],
            "sampling": ["rejection sampling", "likelihood weighting"],
            # Optimization module
            "hill climbing": ["simulated annealing", "random restart"],
            "simulated annealing": ["hill climbing", "optimization"],
            "constraint satisfaction": ["arc consistency", "backtracking", "AC-3"],
            "linear programming": ["optimization", "simplex algorithm"],
            # Learning module
            "k-nearest neighbors": ["classification", "SVMs"],
            "svms": ["kernel trick", "maximum-margin hyperplane"],
            "regression": ["gradient descent", "regularization", "loss functions"],
            "reinforcement learning": ["Q-learning", "Markov decision processes"],
            "q-learning": ["reinforcement learning", "epsilon-greedy"],
            # Neural Networks module
            "neural networks": ["backpropagation", "activation functions", "gradient descent"],
            "backpropagation": ["gradient descent", "chain rule", "loss functions"],
            "cnns": ["convolutional layers", "pooling", "image classification"],
            "rnns": ["LSTMs", "sequence modeling", "vanishing gradients"],
            "dropout": ["regularization", "overfitting", "early stopping"],
            # Language module
            "word embeddings": ["Word2Vec", "GloVe", "TF-IDF"],
            "attention mechanism": ["self-attention", "transformers"],
            "transformers": ["multi-head attention", "positional encoding", "BERT vs GPT"],
            "large language models": ["fine-tuning", "transfer learning", "tokenization"],
        }

        studied = set(t.lower() for t in self.topics.keys())
        suggestions = set()

        for topic in studied:
            for key, related in topic_graph.items():
                if key in topic or topic in key:
                    for r in related:
                        if r.lower() not in studied:
                            suggestions.add(r)

        if not suggestions:
            return [
                "BFS & DFS",
                "propositional logic",
                "Bayes' Rule",
                "neural networks",
            ]

        return list(suggestions)[:5]
