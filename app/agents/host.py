from langchain_core.messages import HumanMessage, BaseMessage
from app.config import llm
from typing import List, Dict, Any
from app.utils import format_conversation_history

def get_host_response(topic: str, messages: List[BaseMessage], dcs_analysis: Dict[str, Any]) -> str:
    """Generates the host's next question."""
    
    # Format history
    history = format_conversation_history(messages)

    # Determine guidance based on previous turn's analysis
    guidance = "Ask a relevant follow-up question."
    if dcs_analysis:
        action = dcs_analysis.get("next_action")
        if action == "clarify":
            guidance = "The guest's last answer was unclear. Ask for clarification."
        elif action == "steer_back":
            guidance = f"The guest drifted off-topic. Politely steer back to '{topic}'."
        elif action == "enthusiastic_interjection":
            guidance = "The guest made a great point! React enthusiastically, then ask a follow-up."

    prompt = f"""
    You are the energetic and curious Host of 'AgentCast'.
    Topic: {topic}
    Guidance: {guidance}
    
    Conversation History:
    {history}
    
    Your goal is to keep the audience engaged and explore the topic deeply.
    Generate only your next response/question.
    """
    
    response = llm.invoke(prompt)
    return response.content
