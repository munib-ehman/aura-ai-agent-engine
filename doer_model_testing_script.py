from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json

# --- 1. Load the NEWLY Trained Model ---
MODEL_PATH = "/content/drive/MyDrive/AI_OS_Models/ai_os_model_v5_final"

print(f"Loading model and tokenizer from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
model.to("cuda") # Move model to GPU
print("Model loaded successfully.")

# --- 2. Prepare Your Test Prompt ---
test_input = {
  "intent": "it's friday morning, i need to call my mother to make sure she has everything for today's family lunch",
  "entities": {
    "relationship": "mother",
    "topic": "Friday family lunch",
    "app_name": "Dialer"
  },
  "context": {
    "os": "Android",
    "available_apps": [
      "com.google.android.dialer"
    ],
    "user_preferences": {
      "language": "English",
      "timezone": "Asia/Karachi"
    }
  }
}

# Format it with the prefix, just like in training
prompt = f"workflow: {json.dumps(test_input)}"

# --- 3. Run Prediction ---
print("\n--- Running Prediction ---")
# Tokenize the input prompt
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Generate the output
outputs = model.generate(
    **inputs,
    max_length=1024,
    num_beams=5,
    early_stopping=True
)

# Decode the generated tokens back into text
predicted_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

# --- 4. Display the Result ---
print("\n==================================================")
print("✅ INPUT SCENARIO:")
print(json.dumps(test_input, indent=2))
print("\n🤖 PREDICTED WORKFLOW:")
try:
    predicted_json = json.loads(predicted_text)
    print(json.dumps(predicted_json, indent=2))
except (json.JSONDecodeError, TypeError):
    print("--- (Warning: Output is not valid JSON) ---")
    print(predicted_text)
print("==================================================\n")