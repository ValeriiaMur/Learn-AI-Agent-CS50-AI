"""Router Agent — classifies user intent to direct the conversation flow."""

from langchain_core.messages import SystemMessage, HumanMessage
from src.utils.model_router import model_router
from src.utils.prompts import ROUTER_SYSTEM_PROMPT


VALID_INTENTS = {"CONCEPT", "QUIZ", "DEEPER", "PROGRESS", "OFF_TOPIC"}


def classify_intent(message: str) -> str:
    """Classify the user's message into an intent category.

    Uses the mid-tier model for reliable classification.

    Args:
        message: The user's raw input

    Returns:
        One of: CONCEPT, QUIZ, DEEPER, PROGRESS, OFF_TOPIC
    """
    model = model_router.get_model("standard")

    response = model.invoke([
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=message),
    ])

    intent = response.content.strip().upper()

    # Validate — fall back to CONCEPT if model returns unexpected value
    if intent not in VALID_INTENTS:
        intent = "CONCEPT"

    return intent
