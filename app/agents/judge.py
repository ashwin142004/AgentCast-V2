from app.config import judge_llm
from app.schemas import DCSAnalysis

from typing import List, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from app.utils import format_conversation_history

def run_dcs_analysis(topic: str, messages: List[BaseMessage]) -> DCSAnalysis:
    """
    Analyzes the ENTIRE conversation history to provide:
    1. Immediate feedback on the last turn (Coherence, Drift, Next Action)
    2. Aggregate metrics for the whole session (Avg Coherence, Topic Retention, MOS, Stability)
    """
    
    # 1. Format history for context
    history_str = format_conversation_history(messages)
    
    # 2. Extract last turn for specific focus
    last_turn_text = "No interaction yet."
    if len(messages) >= 2:
        last_turn_text = f"Host: {messages[-2].content}\nGuest: {messages[-1].content}"

    prompt = f"""
    You are the Lead Editor and Quality Assurance Judge for 'AgentCast', a high-tier podcast.
    
    Your Role: 
    Evaluate the podcast conversation for quality, flow, and adherence to the topic.
    You must provide BOTH immediate feedback on the latest exchange AND cumulative metrics for the entire session.

    Session Context:
    Topic: {topic}
    
    FULL Conversation History:
    {history_str}
    
    ---
    
    EVALUATION INSTRUCTIONS:
    
    1. IMMEDIATE TURN ANALYSIS (Focus on the very last Host-Guest exchange):
       - Coherence Score (1-10): How logically does the guest's answer follow the host's question?
       - Topic Drift: Is the last answer 'on_topic', 'slight_drift', or 'off_topic'?
       - Next Action: Recommend the host's next move ('standard_follow_up', 'clarify', 'steer_back', 'enthusiastic_interjection').
    
    2. CUMULATIVE SESSION METRICS (Analyze the entire history above):
       - Avg Coherence (0-10): The average coherence quality across all turns so far.
       - Topic Retention % (0-100): What percentage of the conversation has stayed strictly on the main topic?
       - Mean Opinion Score (1-5): Estimate the listener's overall satisfaction and engagement level (1=Poor, 5=Excellent).
       - Conversational Stability Index (0-1): A calculated metric reflecting the smoothness of flow. 
         (Subtract points for interruptions, abrupt topic shifts, or confusion. 1.0 is a perfect flow.)

    Output your analysis strictly in the required JSON schema.
    """
    
    structured_llm = judge_llm.with_structured_output(DCSAnalysis)
    return structured_llm.invoke(prompt)
