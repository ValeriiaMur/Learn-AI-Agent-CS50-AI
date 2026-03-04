"""Tutor Agent — the core teaching agent that generates educational responses."""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.utils.model_router import model_router
from src.utils.prompts import (
    TUTOR_SYSTEM_PROMPT,
    QUIZ_SYSTEM_PROMPT,
    EXPLAIN_DEEPER_PROMPT,
    PROGRESS_SUMMARY_PROMPT,
)


def generate_teaching_response(
    message: str,
    intent: str,
    context: str,
    progress: str,
    chat_history: list[dict],
    complexity: str = "standard",
) -> str:
    """Generate an educational response based on intent and context.

    Args:
        message: The user's message
        intent: Classified intent (CONCEPT, QUIZ, DEEPER, PROGRESS, OFF_TOPIC)
        context: Retrieved educational content from the knowledge base
        progress: Learner's progress summary
        chat_history: Previous messages in the conversation
        complexity: Task complexity for model routing

    Returns:
        The tutor's response string
    """
    model = model_router.get_model(complexity)

    # Build conversation history for context
    messages = []

    # Select system prompt based on intent
    if intent == "QUIZ":
        system_prompt = QUIZ_SYSTEM_PROMPT.format(context=context, progress=progress)
    elif intent == "DEEPER":
        history_str = _format_chat_history(chat_history[-6:])  # Last 3 exchanges
        system_prompt = EXPLAIN_DEEPER_PROMPT.format(
            context=context, chat_history=history_str
        )
    elif intent == "PROGRESS":
        system_prompt = PROGRESS_SUMMARY_PROMPT.format(progress=progress)
    elif intent == "OFF_TOPIC":
        system_prompt = (
            "You are LearnAI, an AI/ML tutor. The user's message doesn't seem "
            "related to AI/ML. Gently redirect them back to learning, and suggest "
            "an interesting AI/ML topic they might enjoy based on their progress.\n\n"
            f"Their progress: {progress}"
        )
    else:
        # CONCEPT — default teaching mode
        system_prompt = TUTOR_SYSTEM_PROMPT.format(context=context, progress=progress)

    messages.append(SystemMessage(content=system_prompt))

    # Add recent chat history for continuity
    for msg in chat_history[-6:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    # Add current message
    messages.append(HumanMessage(content=message))

    # Generate response
    response = model.invoke(messages)
    return response.content


def _format_chat_history(history: list[dict]) -> str:
    """Format chat history into a readable string."""
    if not history:
        return "No previous conversation."

    lines = []
    for msg in history:
        role = "Learner" if msg["role"] == "user" else "Tutor"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)
