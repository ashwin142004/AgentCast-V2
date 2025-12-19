import json
from typing import List, TypedDict, Dict, Any
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from src.agents import get_host_response, get_guest_response, run_dcs_analysis

# --- 1. DEFINE THE STATE ---
class GraphState(TypedDict):
    topic: str
    messages: List[BaseMessage]
    turn_number: int
    dcs_analysis: Dict[str, Any]
    evaluation_log: List[Dict[str, Any]]

# --- 2. DEFINE THE GRAPH NODES (WITH DETAILED LOGGING) ---

def host_agent_node(state: GraphState):
    """Node that generates the host's response."""
    print("\n" + "="*50)
    print("--- 🎙️ EXECUTING HOST NODE ---")
    
    # ✨ LOGGING ADDED HERE
    turn_number = state['turn_number']
    if turn_number > 0:
        dcs_analysis = state['dcs_analysis']
        next_action = dcs_analysis.get('next_action', 'standard_follow_up')
        print(f"[DCS Command Received]: Host will perform action: '{next_action}'")
    else:
        print("[DCS Command Received]: This is the first turn, no command needed.")
    
    response_message = get_host_response(
        state["topic"],
        state["messages"],
        state["dcs_analysis"],
        state["turn_number"]
    )
    return {"messages": state["messages"] + [response_message]}

def guest_agent_node(state: GraphState):
    """Node that generates the guest's response."""
    print("\n" + "="*50)
    print("--- 👨‍🔬 EXECUTING GUEST NODE ---")
    response_message = get_guest_response(state["messages"])
    return {"messages": state["messages"] + [response_message]}

def dcs_analyzer_node(state: GraphState):
    """Node that runs the DCS analysis and logs the results."""
    print("\n" + "="*50)
    print("--- 🧠 EXECUTING DCS ANALYSIS NODE ---")
    
    messages = state["messages"]
    turn_number = state["turn_number"]
    
    host_question = messages[-2].content
    guest_answer = messages[-1].content
    
    # ✨ LOGGING ADDED HERE
    print(f"\n[DCS Input] Topic: '{state['topic']}'")
    print(f"[DCS Input] Host Question: '{host_question}'")
    print(f"[DCS Input] Guest Answer (truncated): '{guest_answer[:150].strip()}...'")

    analysis_result = run_dcs_analysis(state["topic"], host_question, guest_answer)

    # ✨ LOGGING ADDED HERE
    print("\n[DCS Judge Output]:")
    print(json.dumps(analysis_result, indent=2))
    print("="*50)

    # --- LOGGING FOR EVALUATION ---
    current_log = {
        "turn": turn_number + 1,
        "host_question": host_question,
        "guest_answer": guest_answer,
        "coherence_score": analysis_result.get("coherence_score"),
        "coherence_reason": analysis_result.get("coherence_reason"),
        "topic_drift": analysis_result.get("topic_drift"),
        "next_host_action": analysis_result.get("next_action"),
    }

    # Pass the messages along to preserve the state.
    return {
        "messages": state["messages"],
        "turn_number": turn_number + 1,
        "dcs_analysis": analysis_result,
        "evaluation_log": state["evaluation_log"] + [current_log]
    }

# --- 3. DEFINE CONDITIONAL EDGES ---
def should_continue(state: GraphState):
    """Determines whether to continue the conversation or end."""
    if state["turn_number"] >= 3:
        return "end"
    return "continue"

# --- 4. COMPILE THE GRAPH ---
def compile_graph():
    """Compiles the LangGraph workflow."""
    workflow = StateGraph(GraphState)

    workflow.add_node("host", host_agent_node)
    workflow.add_node("guest", guest_agent_node)
    workflow.add_node("dcs_analyzer", dcs_analyzer_node)

    workflow.set_entry_point("host")

    workflow.add_edge("host", "guest")
    workflow.add_edge("guest", "dcs_analyzer")

    workflow.add_conditional_edges(
        "dcs_analyzer",
        should_continue,
        {"continue": "host", "end": END}
    )
    return workflow.compile()

