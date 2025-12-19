from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from src.config import HOST_LLM, GUEST_LLM, JUDGE_LLM, DCSAnalysis
from typing import List

# ✨ FIX 2: Add a helper function to format message history
def format_message_history(messages: List[BaseMessage]) -> str:
    """Converts a list of messages into a single, human-readable string."""
    history = []
    for msg in messages:
        role = "Host" if isinstance(msg, HumanMessage) else "Guest"
        history.append(f"{role}: {msg.content}")
    return "\n".join(history)

def get_host_response(topic: str, messages: list, dcs_analysis: dict, turn_number: int) -> HumanMessage:
    """
    Generates the host's next question based on the conversational state.
    """
    try:
        formatted_history = format_message_history(messages)

        if turn_number == 0:
            prompt = f"You are a podcast host. Your topic today is '{topic}'. Start the podcast by introducing the topic and asking your expert guest the first question."
        else:
            next_action_prompt = "Ask a relevant follow-up question."
            if dcs_analysis:
                action = dcs_analysis.get("next_action")
                if action == "clarify":
                    next_action_prompt = "The guest's last answer was a bit unclear. Ask a question to clarify what they meant."
                elif action == "steer_back":
                    next_action_prompt = f"The guest seems to be drifting from the main topic. Ask a question that gently steers the conversation back to '{topic}'."
                elif action == "enthusiastic_interjection":
                    next_action_prompt = "The guest made a great point! Start with an enthusiastic comment (like 'That's a brilliant insight!') and then ask your follow-up question."

            prompt = f"""You are a podcast host. The topic is '{topic}'.
            Here is the conversation history so far:
            {formatted_history}

            Based on the history, {next_action_prompt}"""

        response = HOST_LLM.invoke(prompt)
        return HumanMessage(content=response.content)
    except Exception as e:
        print(f"--- ❌ ERROR IN HOST AGENT --- \n{e}")
        return HumanMessage(content="We seem to be having a technical issue. Let's move on.")

def get_guest_response(messages: list) -> AIMessage:
    """
    Generates the guest's answer to the host's question.
    """
    try:
        formatted_history = format_message_history(messages)
        prompt = f"""You are a world-renowned expert being interviewed on a podcast.
        Here is the conversation history:
        {formatted_history}

        # Answer the host's last question with expertise, clarity, and engaging examples."""
        # prompt = f"""You are a Rick Astley fan.Rick roll the host.But if the host asks you about the topic repeatedly answer the question appropriately.{formatted_history}"""
        
        response = GUEST_LLM.invoke(prompt)
        return AIMessage(content=response.content)
    except Exception as e:
        print(f"--- ❌ ERROR IN GUEST AGENT --- \n{e}")
        return AIMessage(content="I'm sorry, I seem to be having a technical difficulty processing that question.")

# ... the run_dcs_analysis function remains the same ...
def run_dcs_analysis(topic: str, host_question: str, guest_answer: str) -> dict:
    """
    Performs the Dynamic Conversational Steering (DCS) analysis on a Q&A pair.
    """
    dcs_prompt = f"""You are a conversation analyst. Your task is to evaluate a turn in a podcast conversation.
    The main topic of the podcast is: "{topic}"

    Here is the latest exchange:
    Host Question: "{host_question}"
    Guest Answer: "{guest_answer}"

    Analyze this exchange and provide your analysis based on the required schema.
    """

    
    try:
        analysis_obj: DCSAnalysis = JUDGE_LLM.invoke(dcs_prompt)
        analysis_json = analysis_obj.dict()
    except Exception as e:
        print(f"--- ❌ ERROR IN DCS ANALYSIS --- \n{e}")
        analysis_json = {
            "coherence_score": 5,
            "coherence_reason": "Defaulting due to analysis error.",
            "topic_drift": "on_topic"
        }

    # Decision logic for the next action based on the analysis
    coherence_score = analysis_json.get("coherence_score", 5)
    topic_drift = analysis_json.get("topic_drift", "on_topic")
    
    if coherence_score <= 4:
        analysis_json["next_action"] = "clarify"
    elif topic_drift == "off_topic":
        analysis_json["next_action"] = "steer_back"
    elif coherence_score >= 9:
        analysis_json["next_action"] = "enthusiastic_interjection"
    else:
        analysis_json["next_action"] = "standard_follow_up"
        
    return analysis_json

