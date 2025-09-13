import os
import json
import requests

# --- Part 1: The New "Agentic" Core Logic ---

def get_llm_response(prompt, model_name="llama3"):
    """A generic function to call the local Ollama LLM."""
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model_name,
        "prompt": prompt,
        "format": "json",
        "stream": False
    }
    try:
        response = requests.post(url, json=data, timeout=120)
        response.raise_for_status()
        result = response.json()
        return json.loads(result['response']), True
    except Exception as e:
        # Simplified error handling for the main function
        return {"error": str(e), "raw_response": result.get('response', '') if 'result' in locals() else 'N/A'}, False

def run_aura_agent(context_dict, model_name="llama3"):
    """
    Executes the multi-step AI reasoning process to generate a reliable workflow.
    """
    print("  AGENT STEP 1: Identifying Goal...")
    # --- STEP 1: GOAL IDENTIFICATION ---
    goal_prompt = f"""
    You are a goal identification AI. Analyze the user's context and determine their primary, high-level goal.
    Respond ONLY with a JSON object with a single key: "goal".
    CONTEXT: {json.dumps(context_dict)}
    """
    goal_result, success = get_llm_response(goal_prompt, model_name)
    if not success: return goal_result, False
    
    user_goal = goal_result.get('goal', 'No goal identified.')
    print(f"  > Goal Identified: {user_goal}")

    # --- STEP 2: PLANNING ---
    print("\n  AGENT STEP 2: Creating a Plan...")
    plan_prompt = f"""
    You are a logical planner. Given a user's goal and their profile, create a simple, step-by-step plan.
    Respond ONLY with a JSON object with a single key: "plan", which is an array of strings.
    Strictly follow the user's app preferences and only use installed apps.

    USER GOAL: "{user_goal}"
    USER PROFILE: {json.dumps(context_dict.get('user_profile', {}))}
    """
    plan_result, success = get_llm_response(plan_prompt, model_name)
    if not success: return plan_result, False
    
    step_by_step_plan = plan_result.get('plan', [])
    print("  > Plan Created:")
    for i, step in enumerate(step_by_step_plan, 1):
        print(f"    {i}. {step}")

    # --- STEP 3: WORKFLOW GENERATION ---
    print("\n  AGENT STEP 3: Generating Workflow from Plan...")
    workflow_prompt = f"""
    You are a workflow generator. Your only job is to convert a step-by-step plan into a structured JSON workflow.
    Use the provided 'PERFECT OUTPUT' example as a strict template for your response format and logic, including stateful variables.
    Your response MUST be a single, clean JSON object.

    --- WORKFLOW COMMAND SCHEMA ---
    Valid Commands: ['OPEN_APP', 'CREATE_EMAIL', 'SEND_MESSAGE', 'APP_ACTION', 'SET_REMINDER', 'BOOK_RIDE']
    The 'APP_ACTION' command can save its result to an 'output_variable'.
    Subsequent commands can use variables with curly braces, e.g., {{variable_name}}.

    --- EXAMPLE OF A PERFECT STATEFUL WORKFLOW ---
    PLAN: ["Find the latest report file using File Manager.", "Create an email in Outlook and attach the found file."]
    PERFECT OUTPUT:
    {{
      "title": "Find and Send Report",
      "body": "Finds the latest report file and emails it to the team using Outlook.",
      "workflow": [
        {{
          "command": "APP_ACTION",
          "parameters": {{
            "app_name": "File Manager",
            "action_description": "Search for the most recent file named 'weekly_report*.pdf'",
            "output_variable": "report_file_path"
          }}
        }},
        {{
          "command": "CREATE_EMAIL",
          "parameters": {{
            "app": "Outlook",
            "to": "team@example.com",
            "subject": "Latest Weekly Report",
            "attachment": "{{report_file_path}}"
          }}
        }}
      ]
    }}
    --- END OF EXAMPLE ---

    --- CURRENT TASK ---
    PLAN TO EXECUTE: {json.dumps(step_by_step_plan)}
    USER CONTEXT (for details like contact info): {json.dumps(context_dict)}

    Generate the final JSON workflow now.
    """
    final_workflow, success = get_llm_response(workflow_prompt, model_name)
    if success:
        print("  > Workflow Generated Successfully.")
    return final_workflow, success

# --- Main Execution ---
if __name__ == '__main__':
    LOCAL_MODEL_NAME = "llama3"

    print("="*40)
    print(f"  Aura AI Agent Engine (Local Brain)")
    print("="*40 + "\n")

    CONTEXTS_DIR = "contexts"

    if not os.path.exists(CONTEXTS_DIR):
        print(f"Error: Directory '{CONTEXTS_DIR}' not found. Creating it.")
        os.makedirs(CONTEXTS_DIR)
        exit()

    context_files = [f for f in os.listdir(CONTEXTS_DIR) if f.endswith('.json')]
    if not context_files:
        print(f"No JSON context files found in '{CONTEXTS_DIR}'. Please add scenario files.")
        exit()

    for file_name in context_files:
        file_path = os.path.join(CONTEXTS_DIR, file_name)
        print(f"--- Analyzing Scenario from: {file_name} ---")
        
        try:
            with open(file_path, 'r') as f:
                context_data = json.load(f)
            
            print(f"Pending Task: {context_data.get('pending_task', 'N/A')}\n")

            suggestion, success = run_aura_agent(context_data, LOCAL_MODEL_NAME)
            
            print("\n--- Final AI-Generated Workflow ---")
            if success:
                print(json.dumps(suggestion, indent=2))
            else:
                print(f"Error generating workflow: {suggestion}")

            print("\n" + "="*40 + "\n")

        except Exception as e:
            print(f"\nAn unexpected error occurred while processing '{file_name}': {e}")

