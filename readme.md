# **Aura AI Agent Engine**

Welcome to the Aura AI project. This is a proof-of-concept for a sophisticated, hybrid AI agent designed to act as the "brain" for a proactive, intelligent mobile operating system.

The agent uses a **Hybrid "Thinker/Doer" Architecture** to ensure both flexibility and reliability:

1. **The Thinker (LLM):** A powerful, general-purpose Large Language Model (like Llama 3\) that analyzes raw user context and understands the high-level _intent_.
2. **The Doer (Custom Model):** A specialized, fine-tuned transformer model (T5) that takes the simple intent from the Thinker and generates a precise, machine-executable workflow.

## **Project Structure**

aura_ai_project/  
│  
├── contexts/ \# Folder for user context scenarios (JSON files)  
│ ├── cricket_match_scenario.json  
│ └── ... (add your other .json files here)  
│  
├── outputs/ \# Where the trained "Doer" model is saved  
│ └── doer_model/  
│  
├── main.py \# Main script to run the full AI agent demo  
├── train_doer_model.py \# Script to train your custom "Doer" model  
├── doer_dataset.jsonl \# The dataset for training the "Doer" model  
├── requirements.txt \# All the Python dependencies for the project  
└── README.md \# This file

## **1\. Setup and Installation**

Follow these steps to set up your project environment.

### **Prerequisites**

- Python 3.8 or newer.
- pip (Python's package installer).
- (Optional but Recommended) [Ollama](https://ollama.com/) installed for running the local "Thinker" LLM.

### **Installation Steps**

1. Clone or Download the Project:  
   Make sure all the project files are in a single folder on your computer.
2. Open Your Terminal:  
   Navigate to the project's root directory (aura_ai_project/) in your terminal or command prompt.
3. (Recommended) Create a Virtual Environment:  
   This keeps your project's dependencies isolated.  
   python \-m venv venv

   Activate the environment:

   - On Windows: .\\venv\\Scripts\\activate
   - On macOS/Linux: source venv/bin/activate

4. Install All Dependencies:  
   This single command will read the requirements.txt file and install everything you need, including simpletransformers, torch, requests, and pandas.  
   pip install \-r requirements.txt

5. Download a Local LLM (The "Thinker"):  
   If you have Ollama installed, pull a model to use as the reasoning engine. llama3 is a great, fast starting point.  
   ollama pull llama3

## **2\. How to Use**

The process is a two-step cycle: **Train** your custom model, then **Run** the agent to see it in action.

### **Step A: Train Your Custom "Doer" Model**

Before you can run the main agent, you must train your specialized "Doer" model.

1. Prepare Your Dataset:  
   Add or modify the training examples in the doer_dataset.jsonl file. Ensure each line is a valid JSON object.
2. Run the Training Script:  
   Execute the training script from your terminal. This will read the dataset, fine-tune the T5 model, and save the result to the outputs/doer_model directory.  
   python train_doer_model.py

   _(This might take a while, especially on a CPU. For faster training, use Google Colab with a free GPU.)_

### **Step B: Run the Aura AI Agent**

Once your "Doer" model is trained and saved, you can run the main agent.

1. Prepare Your Scenarios:  
   Add or edit the .json context files in the contexts/ folder. These are the test situations you want the AI to analyze.
2. Configure the Agent:  
   Open main.py and ensure the LOCAL_MODEL_NAME is set to the Ollama model you want to use (e.g., "phi3").
3. Run the Main Script:  
   Execute the main script from your terminal. It will loop through each scenario in the contexts folder and print the final AI-generated workflow.  
   python main.py

You have now successfully run the full hybrid AI pipeline\!
