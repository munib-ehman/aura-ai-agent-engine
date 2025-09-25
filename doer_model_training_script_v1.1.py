import json
import pandas as pd
from simpletransformers.t5 import T5Model, T5Args
import os

# --- Configuration ---
MODEL_TYPE = 't5'
MODEL_NAME = 't5-small'
DATASET_FILE = 'doer_dataset.jsonl'
OUTPUT_DIR = 'outputs/doer_model'
TRAIN_EPOCHS = 12

def create_training_dataframe(file_path):
    """
    Loads the .jsonl file and prepares it for training.
    
    *** THIS VERSION INCLUDES A SMART CONVERTER ***
    It automatically handles both the OLD and NEW dataset formats.
    """
    records = []
    print(f"Loading and preparing dataset from '{file_path}'...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                clean_line = line.strip()
                if clean_line:
                    records.append(json.loads(clean_line))
        
        print(f"Dataset loaded. Found {len(records)} training examples.")
        
        data_for_df = []
        for i, record in enumerate(records):
            input_data = record.get('input', {})
            output_data = record.get('output', {})
            
            # --- AUTOMATIC FORMAT CONVERSION LOGIC ---
            # Check if the 'output' is a dictionary (the old format)
            if isinstance(output_data, dict) and 'workflow' in output_data:
                # If so, extract just the workflow array for the target
                output_workflow = output_data.get('workflow', [])
                if i == 0: # Print a message only once
                    print("Note: Old dataset format detected. Auto-converting to new format for training.")
            else:
                # Otherwise, assume it's already in the new format (or handle errors)
                output_workflow = output_data if isinstance(output_data, list) else []

            data_for_df.append({
                "prefix": "workflow",
                "input_text": json.dumps(input_data),
                "target_text": json.dumps(output_workflow)
            })
            
        return pd.DataFrame(data_for_df)

    except FileNotFoundError:
        print(f"Error: Dataset file not found at '{file_path}'")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from the dataset file: {e}")
        return None

def train_doer_model():
    """
    Trains the T5 model on our custom dataset and saves it.
    """
    print("--- Starting Training for Custom 'Doer' Model ---")
    train_df = create_training_dataframe(DATASET_FILE)
    
    if train_df is None or train_df.empty:
        print("Training cannot proceed without a valid dataset.")
        return

    model_args = T5Args()
    model_args.max_seq_length = 512
    model_args.train_batch_size = 2
    model_args.eval_batch_size = 2
    model_args.num_train_epochs = TRAIN_EPOCHS
    model_args.overwrite_output_dir = True
    model_args.output_dir = OUTPUT_DIR
    model_args.save_steps = -1
    model_args.use_cuda = False
    model_args.n_gpu = 0

    print(f"Initializing '{MODEL_NAME}' model...")
    model = T5Model(MODEL_TYPE, MODEL_NAME, args=model_args, use_cuda=False)

    print("\n--- Starting Model Fine-Tuning ---")
    model.train_model(train_df)
    
    print("\n--- Training Complete ---")
    print(f"Your custom 'Doer' model has been saved to the '{OUTPUT_DIR}' directory.")

if __name__ == '__main__':
    train_doer_model()

