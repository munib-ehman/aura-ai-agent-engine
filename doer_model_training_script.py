import json
import pandas as pd
from simpletransformers.t5 import T5Model, T5Args
import os

# --- Configuration ---
MODEL_TYPE = 't5'
MODEL_NAME = 't5-small'
DATASET_FILE = 'doer_dataset.jsonl'
OUTPUT_DIR = 'outputs/doer_model'
TRAIN_EPOCHS = 4

def create_training_dataframe(file_path):
    """
    Loads the .jsonl file and converts it into a pandas DataFrame
    with the required 'prefix', 'input_text', and 'target_text' columns.
    """
    records = []
    print(f"Loading and preparing dataset from '{file_path}'...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Sanitize line before parsing
                clean_line = line.strip()
                if clean_line:
                    records.append(json.loads(clean_line))
        
        print(f"Dataset loaded. Found {len(records)} training examples.")
        
        # Convert the records into the format required by Simple Transformers
        data_for_df = []
        for record in records:
            input_data = record.get('input', {})
            output_data = record.get('output', {})
            
            # The model learns to translate from input_text to target_text
            data_for_df.append({
                "prefix": "workflow", # A task prefix helps the model learn
                "input_text": json.dumps(input_data),
                "target_text": json.dumps(output_data)
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

    # Configure the model's training arguments
    model_args = T5Args()
    model_args.max_seq_length = 512
    model_args.train_batch_size = 2
    model_args.eval_batch_size = 2
    model_args.num_train_epochs = TRAIN_EPOCHS
    model_args.overwrite_output_dir = True
    model_args.use_cuda = False
    model_args.output_dir = OUTPUT_DIR
    model_args.save_steps = -1 # Save at the end of each epoch

    # --- DEFINITIVE CUDA FIX ---
    # We are being extremely explicit to force CPU usage.
    model_args.use_cuda = False
    model_args.n_gpu = 0

    # Initialize the T5 model
    print(f"Initializing '{MODEL_NAME}' model...")
    # The 'cuda_device=-1' is a final, forceful override.
    model = T5Model(MODEL_TYPE, MODEL_NAME, args=model_args,use_cuda=False, cuda_device=-1)

    # Train the model
    print("\n--- Starting Model Fine-Tuning ---")
    model.train_model(train_df)
    
    print("\n--- Training Complete ---")
    print(f"Your custom 'Doer' model has been saved to the '{OUTPUT_DIR}' directory.")

if __name__ == '__main__':
    train_doer_model()

