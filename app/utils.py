from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

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

def format_conversation_as_list(messages: List[BaseMessage]) -> List[dict]:
    """
    Formats the conversation history as a list of dictionaries.
    Pairs Host and Guest messages.
    """
    conversation = []
    
    # Iterate through messages in pairs (assuming Host starts and strict alternation for now, 
    # but handling potential odd length or mis-ordering gracefully is better)
    # Since the graph enforces Host -> Guest -> Judge -> Host..., the messages should be alternating pairs.
    
    # simple pairing
    current_turn = {}
    
    for m in messages:
        if isinstance(m, HumanMessage):
            # Start of a new turn or overwrite if double host (shouldn't happen in this graph)
            content = m.content
            if isinstance(content, list):
                content = " ".join([block.get("text", "") for block in content if block.get("type") == "text"])
            current_turn = {"Host": content}
        elif isinstance(m, AIMessage):
            if "Host" in current_turn:
                content = m.content
                if isinstance(content, list):
                    content = " ".join([block.get("text", "") for block in content if block.get("type") == "text"])
                current_turn["Guest"] = content
                conversation.append(current_turn)
                current_turn = {} # Reset
            else:
                # Guest spoke without Host? Should not happen in strict flow but handle safely
                pass 
                
    return conversation
