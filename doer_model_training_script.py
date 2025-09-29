import json
import pandas as pd
from datasets import Dataset
# --- CHANGE 1: IMPORT THE DATA COLLATOR ---
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

# --- Configuration ---
BASE_MODEL = "google/flan-t5-small"
DATASET_FILE = "doer_dataset_new_template.jsonl"
# Let's save to a new, clean folder to be safe
OUTPUT_MODEL_PATH = "/content/drive/MyDrive/AI_OS_Models/ai_os_model_v6" 

# --- Your data loading function (this is perfect, no changes needed) ---
def load_jsonl_to_dataframe(file_path):
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line.strip()))
    
    data_for_df = []
    for record in records:
        input_data = record.get('input', {})
        output_object = record.get('output', {})
        prompt = f"workflow: {json.dumps(input_data)}"
        completion = json.dumps(output_object)
        data_for_df.append({"prompt": prompt, "completion": completion})
    return pd.DataFrame(data_for_df)

print("Loading dataset...")
train_df = load_jsonl_to_dataframe(DATASET_FILE)
hg_dataset = Dataset.from_pandas(train_df)
print("Dataset loaded and prepared.")

# --- Your tokenization function (this is perfect, no changes needed) ---
print("\nTokenizing data...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
def tokenize_function(examples):
    model_inputs = tokenizer(examples["prompt"], max_length=1024, truncation=True)
    labels = tokenizer(text_target=examples["completion"], max_length=1024, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
tokenized_dataset = hg_dataset.map(tokenize_function, batched=True)
print("Tokenization complete.")

# --- Train the Model ---
print("\nSetting up trainer...")
model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_MODEL_PATH,
    report_to="none",
    num_train_epochs=35,
    per_device_train_batch_size=2,
    learning_rate=5e-5,
    weight_decay=0.01,
    save_total_limit=3,
    predict_with_generate=True,
    push_to_hub=False
)

# --- CHANGE 2: CREATE THE DATA COLLATOR ---
# This will handle the padding for us automatically
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# --- CHANGE 3: GIVE THE COLLATOR TO THE TRAINER ---
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator  # Add the data collator here
)

print("\n--- Starting Model Training ---")
trainer.train()
print("\n--- Training Complete ---")

print(f"\nSaving model to {OUTPUT_MODEL_PATH}...")
trainer.save_model()
print("Model saved successfully!")