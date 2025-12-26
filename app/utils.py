from typing import List
from langchain_core.messages import BaseMessage, HumanMessage

def format_conversation_history(messages: List[BaseMessage]) -> str:
    """
    Formats the conversation history for LLM prompts.
    Converts a list of BaseMessages into a string format:
    Host: [message]
    Guest: [message]
    """
    formatted_lines = []
    for m in messages:
        role = "Host" if isinstance(m, HumanMessage) else "Guest"
        formatted_lines.append(f"{role}: {m.content}")
    
    return "\n".join(formatted_lines)
