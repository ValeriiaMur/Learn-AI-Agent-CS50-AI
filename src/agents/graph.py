"""LangGraph state machine — orchestrates the multi-agent workflow.

Flow:
  User message → Router → Retrieval → Model Router → Tutor → Response
                                                  ↕
                                          Progress Tracker
"""

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END

from src.agents.router import classify_intent
from src.agents.retrieval import retrieve_context
from src.agents.tutor import generate_teaching_response
from src.agents.progress import ProgressTracker
from src.utils.model_router import model_router


class AgentState(TypedDict):
    """State that flows through the agent graph."""
    message: str
    intent: str
    context: str
    documents: list
    complexity: str
    progress_summary: str
    response: str
    chat_history: list[dict]
    topic: str


def route_intent(state: AgentState) -> AgentState:
    """Step 1: Classify the user's intent."""
    intent = classify_intent(state["message"])
    state["intent"] = intent
    return state


def retrieve_knowledge(state: AgentState) -> AgentState:
    """Step 2: Retrieve relevant educational content."""
    if state["intent"] == "PROGRESS":
        # No retrieval needed for progress queries
        state["context"] = ""
        state["documents"] = []
        return state

    context, documents = retrieve_context(state["message"])
    state["context"] = context
    state["documents"] = documents
    return state


def classify_complexity(state: AgentState) -> AgentState:
    """Step 3: Determine model tier based on intent and message."""
    complexity = model_router.classify_complexity(
        state["intent"], state["message"]
    )
    state["complexity"] = complexity
    return state


def generate_response(state: AgentState) -> AgentState:
    """Step 4: Generate the teaching response."""
    response = generate_teaching_response(
        message=state["message"],
        intent=state["intent"],
        context=state["context"],
        progress=state["progress_summary"],
        chat_history=state.get("chat_history", []),
        complexity=state["complexity"],
    )
    state["response"] = response
    return state


def extract_topic(message: str) -> str:
    """Simple topic extraction from the user's message.

    In a production system, you'd use NER or a classifier.
    """
    # Strip common question prefixes
    prefixes = [
        "what is", "what are", "explain", "how does", "how do",
        "tell me about", "teach me about", "quiz me on",
        "what's", "define", "describe",
    ]
    lower = message.lower().strip("?!. ")
    for prefix in prefixes:
        if lower.startswith(prefix):
            return lower[len(prefix):].strip()
    return lower[:50]  # Fallback: first 50 chars


def build_graph() -> StateGraph:
    """Build the LangGraph agent orchestration graph."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("route", route_intent)
    workflow.add_node("retrieve", retrieve_knowledge)
    workflow.add_node("classify", classify_complexity)
    workflow.add_node("respond", generate_response)

    # Define edges (linear pipeline)
    workflow.set_entry_point("route")
    workflow.add_edge("route", "retrieve")
    workflow.add_edge("retrieve", "classify")
    workflow.add_edge("classify", "respond")
    workflow.add_edge("respond", END)

    return workflow.compile()


class LearnAIAgent:
    """Main agent class — wraps the LangGraph workflow with progress tracking."""

    def __init__(self):
        self.graph = build_graph()
        self.progress = ProgressTracker()
        self.chat_history: list[dict] = []

    def chat(self, message: str) -> str:
        """Process a user message and return the tutor's response.

        Args:
            message: The user's input

        Returns:
            The tutor's educational response
        """
        # Build initial state
        state: AgentState = {
            "message": message,
            "intent": "",
            "context": "",
            "documents": [],
            "complexity": "standard",
            "progress_summary": self.progress.get_progress_summary(),
            "response": "",
            "chat_history": self.chat_history,
            "topic": "",
        }

        # Run the graph
        result = self.graph.invoke(state)

        # Extract topic and record progress
        topic = extract_topic(message)
        self.progress.record_interaction(topic, result["intent"])

        # Update chat history
        self.chat_history.append({"role": "user", "content": message})
        self.chat_history.append({"role": "assistant", "content": result["response"]})

        return result["response"]

    def get_progress(self) -> str:
        """Get the learner's progress summary."""
        return self.progress.get_progress_summary()

    def get_suggestions(self) -> list[str]:
        """Get suggested next topics."""
        return self.progress.get_suggested_topics()
