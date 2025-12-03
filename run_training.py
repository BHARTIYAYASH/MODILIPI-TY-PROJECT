# run_training.py
# This single file lets you TEST your data pipeline first, then TRAIN.

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import load_dataset
import evaluate
import numpy as np
import random
import sys # We use this to stop the script after testing

# --- 1. PREPROCESSING FUNCTIONS (From your Colab) ---

def create_continuous_string(text):
    """Remove all spaces from text to create a continuous string"""
    if not isinstance(text, str):
        return "" # Handle potential bad data
    return text.replace(" ", "")

def preprocess_dataset(examples):
    """Creates the 'continuous_string' (input) and 'source_context' (answer key)"""
    examples["continuous_string"] = [create_continuous_string(text) for text in examples["correct"]]
    examples["source_context"] = examples["correct"]
    return examples

def tokenize_and_align_labels(examples, tokenizer, label2id):
    """
    The most important function.
    Takes the 'continuous_string' and 'source_context' and creates the 'labels' column.
    """
    # Tokenize the 'continuous_string' (the "Question")
    tokenized_inputs = tokenizer(
        examples["continuous_string"],
        truncation=True,
        is_split_into_words=False,
        padding="max_length",
        max_length=256,
        return_offsets_mapping=True # This gives us (start_char, end_char) for each token
    )

    labels = []

    # Loop over each sentence in the batch
    for i, original_sentence in enumerate(examples["source_context"]):
        if not isinstance(original_sentence, str):
            labels.append([-100] * len(tokenized_inputs["input_ids"][i]))
            continue

        # Get the original words from the 'source_context' (the "Answer Key")
        original_words = original_sentence.split()

        # Create word boundaries from the 'Answer Key' in the continuous string
        word_boundaries = []
        current_pos = 0
        for word in original_words:
            word_boundaries.append((current_pos, current_pos + len(word)))
            current_pos += len(word) # Index in the *continuous* string

        offset_mapping = tokenized_inputs["offset_mapping"][i]
        label_ids = []

        # Go token by token and assign a label
        for offset in offset_mapping:
            start_char, end_char = offset
            
            # If it's a [CLS], [SEP], or [PAD] token, label it -100 to ignore
            if start_char == 0 and end_char == 0:
                label_ids.append(-100)
                continue

            # Find which 'Answer Key' word this token belongs to
            word_idx = None
            for idx, (word_start, word_end) in enumerate(word_boundaries):
                if word_start <= start_char and end_char <= word_end:
                    word_idx = idx
                    break

            if word_idx is None:
                label_ids.append(label2id["O"]) # Not part of any word
            else:
                # Is this token the *start* of the word?
                is_first_token = start_char == word_boundaries[word_idx][0]
                label_ids.append(label2id["B"] if is_first_token else label2id["I"])

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    if "offset_mapping" in tokenized_inputs:
        del tokenized_inputs["offset_mapping"]
    return tokenized_inputs

# --- 2. METRICS FUNCTION ---
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    # Define label_list inside or pass it in
    label_list = ["O", "B", "I"]
    
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    seqeval_metric = evaluate.load("seqeval")
    results = seqeval_metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# --- 3. MAIN SCRIPT ---
def main():
    # --- Basic Setup ---
    print("Starting script...")
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = "l3cube-pune/marathi-roberta"
    label2id = {"O": 0, "B": 1, "I": 2}
    id2label = {v: k for k, v in label2id.items()}

    # --- Load Data ---
    print("Loading dataset from 'train.csv'...")
    try:
        full_dataset = load_dataset('csv', data_files='train.csv')['train']
    except FileNotFoundError:
        print("ERROR: 'train.csv' not found. Make sure it's in the same folder.")
        return

    # --- Preprocess Data ---
    print("Running initial preprocessing (creating continuous_string)...")
    processed_dataset = full_dataset.map(preprocess_dataset, batched=True)

    # --- Load Tokenizer ---
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # --- Tokenize and Align Labels ---
    print("Running tokenization and label alignment...")
    tokenized_dataset = processed_dataset.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer, label2id), 
        batched=True
    )
    
    # =================================================================
    # --- 🚀 STEP 1: TEST THE DATA PIPELINE (DEBUGGING BLOCK) 🚀 ---
    # =================================================================
    print("\n" + "="*80)
    print("STARTING DATA PREPARATION TEST...")
    print("We will check the first 3 examples to see if labels are correct.")
    print("="*80)
    
    for i in range(3):
        print(f"\n--- EXAMPLE {i+1} ---")
        
        # Get data from the *original* dataset
        original_text = processed_dataset[i]['correct']
        continuous_text = processed_dataset[i]['continuous_string']
        
        # Get data from the *final tokenized* dataset
        token_ids = tokenized_dataset[i]['input_ids']
        label_ids = tokenized_dataset[i]['labels']
        
        # Convert IDs back to human-readable text
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        labels = [id2label.get(l_id, "IGN") for l_id in label_ids] # IGN for -100

        print(f"Original Text:   {original_text}")
        print(f"Continuous Text: {continuous_text}\n")
        print("Tokens and Labels (Head-to-Head):")
        print("-" * 40)
        print(f"{'TOKEN':<15} | {'LABEL':<5}")
        print("-" * 40)
        
        for token, label in zip(tokens, labels):
            if token == tokenizer.pad_token: # Stop when we hit padding
                break
            if token not in (tokenizer.cls_token, tokenizer.sep_token):
                print(f"{token:<15} | {label:<5}")
        
    print("\n" + "="*80)
    print("TESTING COMPLETE. Check the labels above.")
    print("If they look correct (B at start of words, I inside words), you are ready to train.")
    
    # 🛑 We stop the script here so you can check the output.
    # 🛑 TO TRAIN: COMMENT OUT OR DELETE THE LINE 'return' BELOW.
    # return 
    # =================================================================
    # --- 🛑 END OF TESTING BLOCK 🛑 ---
    # =================================================================


    # =================================================================
    # --- 🚀 STEP 2: FULL MODEL TRAINING 🚀 ---
    # =================================================================
    # This code will only run if you remove the 'return' line above.
    
    print("\nStarting full model training...")
    
    # --- Split and Clean Data ---
    dataset_split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_split['train']
    eval_dataset = dataset_split['test']
    
    # Remove old columns
    train_dataset = train_dataset.remove_columns(full_dataset.column_names)
    eval_dataset = eval_dataset.remove_columns(full_dataset.column_names)

    # --- Load Model ---
    print("Loading model for training...")
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    model.to(device)

    # --- Training Arguments ---
    # (Using the settings from your Colab notebook)
    training_args = TrainingArguments(
        output_dir="./marathi-segmentation-final",
        num_train_epochs=3, # Using 3 epochs from your notebook
        per_device_train_batch_size=8, # 8 is safer for 6GB VRAM
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2, # Effective batch size = 16
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=200,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # --- Start Training ---
    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    trainer.train()

    # --- Save Final Model ---
    print("Training complete. Saving model...")
    final_model_path = "./marathi-segmentation-model-final"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Model saved successfully to '{final_model_path}'")

# --- Run the main function ---
if __name__ == "__main__":
    main()