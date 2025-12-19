AgentCast: A Multi-Agent Podcasting System
This project is an implementation of AgentCast, a system that uses a multi-agent architecture to autonomously generate a podcast conversation between an AI Host and an AI Guest.

Project Structure
agentcast/
├── .env                  # Stores API keys and other secrets
├── main.py               # The main script to run the podcast generation
├── requirements.txt      # Lists all Python dependencies for the project
├── README.md             # This file
├── data/                 # Directory for training data (e.g., podcast transcripts)
└── outputs/              # Directory where evaluation logs (CSV) are saved
└── src/
    ├── __init__.py       # Makes 'src' a Python package
    ├── agents.py         # Defines the logic and prompts for the Host, Guest, and Judge agents
    ├── config.py         # Handles configuration, like loading API keys and model names
    └── graph.py          # Defines the LangGraph state, nodes, and conversational flow

Setup Instructions
Clone the Repository (or create the structure manually):
Create the folders and files as shown in the structure above.

Create a Virtual Environment (Recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install Dependencies:
Install all the required Python libraries from the requirements.txt file.

pip install -r requirements.txt

Set Up Your API Key:

Create a file named .env in the root directory of the project (agentcast/).

Inside the .env file, add your Google API key in the following format:

GOOGLE_API_KEY="your_actual_google_api_key_here"

How to Run the Project
Once the setup is complete, you can run the podcast generation from your terminal:

python main.py

The script will:

Initialize the Host, Guest, and Judge LLMs.

Start the podcast conversation on the topic defined in main.py.

Print the turn-by-turn dialogue to the console in real-time.

After the conversation ends, it will print a summary table of the evaluation log.

It will also save this detailed evaluation log as a CSV file in the outputs/ directory, named with the current timestamp.