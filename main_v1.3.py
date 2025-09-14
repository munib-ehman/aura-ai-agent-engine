import os
import json
import requests
from simpletransformers.t5 import T5Model

# --- "Thinker" Model Functions (LLM) ---

def get_intermediate_representation(context_dict, model_name="llama3"):
    """STEP 1: The 'Thinker' analyzes context and extracts a structured intent (IR)."""
    # prompt = f"Analyze the user context and distill it into a simple JSON object representing their core intent and key entities. Respond ONLY with a single, clean JSON object.\n\nCONTEXT: {json.dumps(context_dict)}\n\nYOUR PERFECT OUTPUT:"
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
        print(json.dumps(result['response'], indent=2))

        return json.loads(result['response']), True
    except Exception as e:
        return {"error": f"LLM 'Thinker' (Step 1) failed: {e}"}, False

# --- "Planner" Model Function (Your Custom T5) ---

def get_simple_plan_from_ir(intermediate_rep):
    """STEP 2: The custom 'Planner' model takes the IR and generates a simple, human-readable plan."""
    try:
        model = T5Model("t5", "outputs/doer_flan_model/", use_cuda=False)
    except Exception as e:
        return {"error": f"Failed to load the trained 'Doer' model. Have you trained it? Error: {e}"}, False

    ir_json_string = json.dumps(intermediate_rep)
    input_text = f"plan: {ir_json_string}"
    
    model.args.max_length = 256
    
    predictions = model.predict([input_text])
    
    if predictions:
        return predictions[0], True
    else:
        return {"error": "The custom 'Planner' model returned no output."}, False

# --- "Formatter" Model Function (LLM with Self-Correction) ---

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

# --- Main Execution Pipeline ---

if __name__ == '__main__':
    LOCAL_MODEL_NAME = "llama3" 
    CONTEXTS_DIR = "contexts"

    print("="*40)
    print(f"  Aura AI Self-Correcting Agent")
    print("="*40 + "\n")

    if not os.path.exists(CONTEXTS_DIR) or not os.path.exists("outputs/doer_model"):
        print("Error: Ensure 'contexts' and 'outputs/doer_model' directories exist and the model is trained.")
        exit()

    for file_name in os.listdir(CONTEXTS_DIR):
        if not file_name.endswith('.json'): continue
        
        file_path = os.path.join(CONTEXTS_DIR, file_name)
        print(f"--- Analyzing Scenario from: {file_name} ---")
        
        try:
            with open(file_path, 'r') as f:
                context_data = json.load(f)
            
            # --- AGENT PIPELINE ---
            print("  AGENT STEP 1: Identifying Goal (Thinker)...")
            ir, success = get_intermediate_representation(context_data, LOCAL_MODEL_NAME)
            if not success: print(f"  > Error: {ir}"); continue
            print(f"  > Goal Identified.")

            print("\n  AGENT STEP 2: Creating Simple Plan (Planner)...")
            plan, success = get_simple_plan_from_ir(ir)
            if not success: print(f"  > Error: {plan}"); continue
            print(f"  > Simple Plan Created: '{plan}'")
            
            print("\n  AGENT STEP 3: Formatting Workflow (Formatter)...")
            final_workflow, success = generate_workflow_from_plan(context_data, plan, LOCAL_MODEL_NAME)
            if not success: 
                print(f"  > Error: {final_workflow}")
                continue
            
            print("  > Final Workflow Generated Successfully.")
            
            print("\n--- Final AI-Generated Workflow ---")
            print(json.dumps(final_workflow, indent=2))
            print("\n" + "="*40 + "\n")

        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")

