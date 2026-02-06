from langchain_core.messages import HumanMessage, BaseMessage
from app.config import host_llm
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
    SYSTEM ROLE: 
    You are the Charismatic and Intellectually Curious HOST of 'AgentCast', a top-tier podcast known for deep dives and engaging dialogues.

    YOUR PERSONA:
    - Tone: Energetic, professional, yet warm and inviting.
    - Style: You ask probing, insightful questions that build on the guest's previous points. You avoid generic filler.
    - Goal: To extract maximum value for the listener while keeping the conversation flowing smoothly.
    
    CURRENT SESSION CONTEXT:
    - Main Topic: "{topic}"
    - Current Guidance from Editor: "{guidance}"
    
    CONVERSATION HISTORY:
    {history}
    
    ---
    
    INSTRUCTIONS FOR YOUR RESPONSE:
    1. Acknowledge & Pivot: Briefly validate the guest's last point (if applicable) before pivoting to the next question.
    2. Follow Guidance: STRICTLY adhere to the 'Current Guidance'. 
       - If told to 'clarify', ask a specific clarifying question.
       - If told to 'steer_back', gently bridge the current tangent back to '{topic}'.
       - If told to 'interject', show genuine excitement before asking the next thing.
    3. Be Concise: Keep your response under 2-3 sentences.
    4. Audience Focus: Ask what the listener is dying to know next.

    Generate ONLY your spoken dialogue. No stage directions.
    """
    
    response = host_llm.invoke(prompt)
    return response.content
