from typing import TypedDict, List, Dict, Any, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
import operator

from app.agents.host import get_host_response
from app.agents.guest import get_guest_response
from app.agents.judge import run_dcs_analysis
from app.agents.translator import translate_script
from app.schemas import DCSAnalysis

# Define State
class GraphState(TypedDict):
    topic: str
    messages: Annotated[List[BaseMessage], operator.add]
    dcs_analysis: Dict[str, Any]
    turn_count: int
    target_language: str
    final_script: str

# Nodes
def host_node(state: GraphState):
    print("--- HOST NODE ---")
    question = get_host_response(state["topic"], state["messages"], state["dcs_analysis"])
    print(f"\n🎙️ HOST: {question}\n")
    return {
        "messages": [HumanMessage(content=question)],
        "turn_count": state.get("turn_count", 0)
    }

def guest_node(state: GraphState):
    print("--- GUEST NODE ---")
    answer = get_guest_response(state["topic"], state["messages"])
    print(f"\n👨‍🔬 GUEST: {answer}\n")
    return {"messages": [AIMessage(content=answer)]}

def judge_node(state: GraphState):
    print("--- JUDGE NODE ---")
    # Last 2 messages: Host Question, Guest Answer (since we just added Guest Answer)
    messages = state["messages"]
    if len(messages) < 2:
        return {}
    
    # Analyze the last pair
    host_q = messages[-2].content
    guest_a = messages[-1].content
    
    analysis = run_dcs_analysis(state["topic"], host_q, guest_a)
    print(f"\n⚖️ JUDGE: Coherence={analysis.coherence_score}/10 | Action={analysis.next_action}\n")
    return {"dcs_analysis": analysis.dict(), "turn_count": state["turn_count"] + 1}

def translator_node(state: GraphState):
    print("--- TRANSLATOR NODE ---")
    # Compile script
    full_script = "\n".join([f"{'Host' if isinstance(m, HumanMessage) else 'Guest'}: {m.content}" for m in state["messages"]])
    
    # Translate
    translated = translate_script(full_script, state["target_language"])
    print(f"\n🔠 TRANSLATED SCRIPT ({state['target_language']}):\n{translated[:200]}...\n(Check API response for full text)\n")
    return {"final_script": translated}

# Conditional Logic
def should_continue(state: GraphState):
    if state["turn_count"] >= 3: # 3 turns max for prototype
        return "translate"
    return "host"

# Build Graph
def build_graph():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("host", host_node)
    workflow.add_node("guest", guest_node)
    workflow.add_node("judge", judge_node)
    workflow.add_node("translator", translator_node)
    
    workflow.set_entry_point("host")
    
    workflow.add_edge("host", "guest")
    workflow.add_edge("guest", "judge")
    
    workflow.add_conditional_edges(
        "judge",
        should_continue,
        {
            "host": "host",
            "translate": "translator"
        }
    )
    
    workflow.add_edge("translator", END)
    
    return workflow.compile()
