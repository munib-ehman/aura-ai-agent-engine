import os
import json
import requests
import pandas as pd
from simpletransformers.t5 import T5Model

# --- Part 1: The "Thinker" - LLM for Reasoning ---

def get_intermediate_representation(context_dict, model_name="llama3"):
    """
    Step 1 of the hybrid model. The LLM analyzes the context and extracts a clear,
    structured 'Intermediate Representation' (IR) of the user's intent.
    """
    prompt = f"""
    You are a highly intelligent context analysis AI. Your only job is to analyze a user's raw context
    and distill it into a simple, structured JSON object representing their core intent and key entities.
    This is the "Intermediate Representation" (IR) that a specialized model will use.

    Respond ONLY with a single, clean JSON object.

    EXAMPLE 1:
    CONTEXT: {{ "pending_task": "what's the score of the Pak v India match? Saad is asking on WhatsApp", "user_profile": {{...}} }}
    YOUR PERFECT OUTPUT:
    {{
      "intent": "get_live_score_and_reply",
      "entities": {{
        "topic": "Pakistan vs India Cricket Match",
        "recipient_name": "Saad",
        "recipient_contact": "+923001234567",
        "score_source_app": "Cricinfo",
        "reply_app": "WhatsApp"
      }}
    }}
    
    EXAMPLE 2:
    CONTEXT: {{ "pending_task": "Start a group call with the project team on WhatsApp.", "user_profile": {{...}} }}
    YOUR PERFECT OUTPUT:
    {{
      "intent": "make_whatsapp_call",
      "entities": {{
        "recipient_name": "Project Team",
        "recipient_contact": "project_alpha_group"
      }}
    }}
    ---
    
    Now, analyze the following context and generate the IR.
    CONTEXT: {json.dumps(context_dict)}
    """
    
    url = "http://localhost:11434/api/generate"
    data = { "model": model_name, "prompt": prompt, "format": "json", "stream": False }
    
    try:
        response = requests.post(url, json=data, timeout=120)
        response.raise_for_status()
        result = response.json()
        return json.loads(result['response']), True
    except requests.exceptions.RequestException as e:
        return {"error": f"Connection to Ollama failed. Is it running? Error: {e}"}, False
    except json.JSONDecodeError:
        return {"error": "The LLM returned an invalid JSON. Could not decode.", "raw_response": result.get('response', '')}, False
    except Exception as e:
        return {"error": str(e)}, False

# --- Part 2: The "Doer" - Your Custom Model ---

def generate_workflow_from_ir(intermediate_rep):
    """
    Step 2 of the hybrid model. This function loads YOUR custom-trained model
    and uses it to generate the final, precise workflow from the IR.
    """
    try:
        # We explicitly tell the model not to use the GPU when loading it.
        model = T5Model("t5", "outputs/doer_flan_model/", use_cuda=False)
    except Exception as e:
        return {"error": f"Failed to load the trained 'Doer' model from 'outputs/doer_model/'. Have you trained it yet? Error: {e}"}, False

    # Prepare the input for the T5 model
    ir_json_string = json.dumps(intermediate_rep)
    input_text = f"workflow: {ir_json_string}"
    
    # --- THIS IS THE FIX ---
    # In simpletransformers, you modify the model.args object for prediction settings.
    model.args.max_length = 512
    
    # Use the model to predict (generate) the workflow.
    # The predict function will now use the max_length we just set.
    predictions = model.predict([input_text])
    
    try:
        # The model's output is a JSON string, so we parse it
        final_workflow = json.loads(predictions[0])
        return final_workflow, True
    except (json.JSONDecodeError, IndexError):
        return {"error": "The custom 'Doer' model returned an invalid workflow JSON.", "raw_output": predictions[0] if predictions else "No output"}, False


def generate_workflow_from_plan(context_dict, simple_plan, model_name="llama3", max_retries=2):
    """
    STEP 3: The 'Formatter' takes the simple plan and converts it into a perfect JSON workflow.
    *** UPGRADED WITH A SELF-CORRECTION LOOP ***
    """
    base_prompt = f"""
    You are a precise AI workflow generator. Your only job is to convert a simple plan into a structured JSON workflow.
    You MUST use the provided command schema.
    Your response MUST be a single, clean JSON object containing a "title", "body", and "workflow" array.

    --- COMMAND SCHEMA ---
    - OPEN_APP: {{ "app_name": "string" }}
    - SEND_MESSAGE: {{ "app": "string", "to": "string", "message": "string", "attachments": ["string"] }}
    - APP_ACTION: {{ "app_name": "string", "action_description": "string", "output_variable": "string" }}
    - CREATE_EMAIL: {{ "app": "string", "to": "string", "subject": "string", "body": "string", "attachments": ["string"] }}
    - SET_REMINDER: {{ "app": "string", "content": "string", "trigger": {{...}} }}
    - MAKE_CALL: {{ "to": "string" }}
    - GET_DEVICE_STATE: {{ "state_key": "string", "output_variable": "string" }}

    --- ORIGINAL CONTEXT ---
    {json.dumps(context_dict, indent=2)}

    --- SIMPLE PLAN TO EXECUTE ---
    {simple_plan}

    --- YOUR TASK ---
    Generate the final, machine-readable JSON workflow based on the plan and the context.
    """
    
    url = "http://localhost:11434/api/generate"
    
    for attempt in range(max_retries):
        data = { "model": model_name, "prompt": base_prompt, "format": "json", "stream": False }
        try:
            response = requests.post(url, json=data, timeout=120)
            response.raise_for_status()
            result = response.json()
            raw_response = result['response']
            
            # --- VALIDATION STEP ---
            # Try to parse the JSON. If it works, we succeed.
            parsed_json = json.loads(raw_response)
            return parsed_json, True

        except json.JSONDecodeError:
            # --- CORRECTION STEP ---
            print("  > Formatter returned invalid JSON. Attempting self-correction...")
            # If parsing fails, create a new, simpler prompt to fix the broken JSON.
            base_prompt = f"""
            The following text is broken JSON. Your only job is to fix all syntax errors (like missing quotes, brackets, or commas) and return a single, valid JSON object.
            Do not add any explanation or extra text.
            
            BROKEN JSON:
            {raw_response}

            CORRECTED JSON:
            """
        except Exception as e:
            return {"error": f"LLM 'Formatter' (Step 3) failed: {e}"}, False

    # If all retries fail, return the final error.
    return {"error": "The LLM failed to produce valid JSON after multiple correction attempts.", "raw_output": raw_response}, False


# --- Main Execution ---
def run_hybrid_agent(context_dict, model_name="llama3"):
    """Executes the full Thinker -> Doer pipeline."""
    
    print("  AGENT STEP 1: Reasoning with LLM (The Thinker)...")
    intermediate_rep, success = get_intermediate_representation(context_dict, model_name)
    if not success:
        return intermediate_rep, False
    
    print("  > Intermediate Representation Generated:")
    print(json.dumps(intermediate_rep, indent=2))
    
    print("\n  AGENT STEP 2: Generating Workflow with Custom Model (The Doer)...")
    final_workflow, success = generate_workflow_from_ir(intermediate_rep)
    if success:
        print("  > Final Workflow Generated Successfully.")
    
    return final_workflow, success

if __name__ == '__main__':
    # --- Configuration ---
    # The name of the local model you downloaded with Ollama (e.g., "phi3", "llama3")
    LOCAL_MODEL_NAME = "llama3" 
    CONTEXTS_DIR = "contexts"

    print("="*40)
    print(f"  Aura AI Hybrid Agent Engine")
    print("="*40 + "\n")

    if not os.path.exists(CONTEXTS_DIR):
        os.makedirs(CONTEXTS_DIR)
        print(f"Directory '{CONTEXTS_DIR}' created. Please add context files to test.")
        exit()
    
    if not os.path.exists("outputs/doer_model"):
        print("Error: The trained 'Doer' model was not found in the 'outputs/doer_model' directory.")
        print("Please run 'python train_doer_model.py' first to train and save the model.")
        exit()

    context_files = [f for f in os.listdir(CONTEXTS_DIR) if f.endswith('.json')]
    if not context_files:
        print(f"No JSON context files found in '{CONTEXTS_DIR}'. Please add a scenario file to test.")
        exit()

    for file_name in context_files:
        # The variable name has been corrected from 'file_.name' to 'file_name'
        file_path = os.path.join(CONTEXTS_DIR, file_name)
        
        try:
            with open(file_path, 'r') as f:
                context_data = json.load(f)
            
            print(f"--- Analyzing Scenario from: {file_name} ---")
            print(f"Pending Task: {context_data.get('pending_task', 'N/A')}\n")

            suggestion, success = run_hybrid_agent(context_data, LOCAL_MODEL_NAME)
            
            print("\n--- Final AI-Generated Workflow ---")
            if success:
                print(json.dumps(suggestion, indent=2))
            else:
                print(f"Error generating workflow: {suggestion}")

            print("\n" + "="*40 + "\n")

        except Exception as e:
            print(f"\nAn unexpected error occurred while processing '{file_name}': {e}")

