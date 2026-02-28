from app.config import judge_llm
from app.schemas import DCSAnalysis

def run_dcs_analysis(topic: str, host_question: str, guest_answer: str) -> DCSAnalysis:
    """Analyzes the conversation turn for coherence and topic drift."""
    
    prompt = f"""
    Analyze this podcast exchange.
    Topic: {topic}
    Host: {host_question}
    Guest: {guest_answer}
    
    Provide your analysis in JSON format matching the schema:
    - coherence_score (1-10)
    - coherence_reason (string)
    - topic_drift ('on_topic', 'slight_drift', 'off_topic')
    - next_action ('standard_follow_up', 'clarify', 'steer_back', 'enthusiastic_interjection')
    """
    
    structured_llm = judge_llm.with_structured_output(DCSAnalysis)
    return structured_llm.invoke(prompt)
