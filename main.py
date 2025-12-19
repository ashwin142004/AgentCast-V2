#Main file for DCS
import pandas as pd
from datetime import datetime
from src.graph import compile_graph

def main():
    """Main function to run the AgentCast podcast generation."""
    app = compile_graph()

    podcast_topic = "The rise of artificial general intelligence and its potential impact on society"
    
    initial_state = {
        "topic": podcast_topic,
        "messages": [],
        "turn_number": 0,
        "dcs_analysis": {},
        "evaluation_log": []
    }

    print(f"🚀 Starting AgentCast podcast on: {podcast_topic}\n")

    final_state = None
    # ✨ FIX 1: Only print messages from the host and guest nodes
    generating_nodes = ["host", "guest"]

    for event in app.stream(initial_state):
        if "__end__" not in event:
            node_name = next(iter(event.keys()))
            if node_name in generating_nodes:
                latest_message = event[node_name]['messages'][-1]
                role = "Host" if latest_message.type == "human" else "Guest"
                print(f"\n>> {role}:\n{latest_message.content}\n")
        else:
            final_state = event["__end__"]

    print("\n--- ✅ PODCAST FINISHED ---")

    if final_state and final_state.get("evaluation_log"):
        print("\n--- 📈 EVALUATION INDEX ---")
        evaluation_df = pd.DataFrame(final_state["evaluation_log"])
        print(evaluation_df.to_string())
        
        # Save the evaluation log to a CSV file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"outputs/evaluation_log_{timestamp}.csv"
        evaluation_df.to_csv(output_path, index=False)
        print(f"\nEvaluation log saved to {output_path}")
    else:
        print("No final state or evaluation log to process.")

if __name__ == "__main__":
    main()

