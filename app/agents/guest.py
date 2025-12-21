from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from app.config import llm
from typing import List

def get_guest_response(topic: str, messages: List[BaseMessage]) -> str:
    """Generates the guest's expert answer."""
    
    history = "\n".join([f"{'Host' if isinstance(m, HumanMessage) else 'Guest'}: {m.content}" for m in messages])
    
    prompt = f"""
    You are a world-renowned expert Guest on 'AgentCast'.
    Topic: {topic}
    
    Conversation History:
    {history}
    
    Answer the Host's last question with authority, clarity, and engaging examples.
    Keep your answer concise (~3 sentences) but insightful.
    """
    
    response = llm.invoke(prompt)
    return response.content
