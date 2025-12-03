# test_interactive.py
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# --- 1. CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ⚠️ IMPORTANT: Path to your saved model
# This must match the folder name of the model you want to test
# model_path = "./marathi-segmentation-final"
model_path = "./marathi-segmentation-final/checkpoint-266862"

# The labels MUST match what you used for training
id2label = {0: "O", 1: "B", 2: "I"}

# --- 2. LOAD YOUR FINE-TUNED MODEL & TOKENIZER ---
print(f"Loading model from {model_path}...")
try:
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to(device)
    model.eval() # Set model to evaluation mode
except Exception as e:
    print(f"--- ERROR ---")
    print(f"Could not load model from '{model_path}'.")
    print(f"Did you run the CORRECTED training script yet?")
    print(f"Error details: {e}")
    exit()

print("Model loaded successfully. Type 'quit' to exit.")
print("="*50)

# --- 3. THE PREDICTION FUNCTION ---
def predict_segmentation(continuous_text):
    inputs = tokenizer(
        continuous_text,
        return_tensors="pt",
        truncation=True,
        return_offsets_mapping=True
    )
    
    offset_mapping = inputs.pop('offset_mapping')
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    predictions = torch.argmax(logits, dim=2)[0].cpu().numpy()
    offsets = offset_mapping[0].cpu().numpy()

    segmented_text = ""
    
    for (pred, offset) in zip(predictions, offsets):
        label = id2label.get(pred, "O") # Get label, default to "O"
        token_text = continuous_text[offset[0]:offset[1]]
        
        if not token_text:
            continue
        
        if label == "B":
            if segmented_text: # Add a space if it's not the first word
                segmented_text += " "
            segmented_text += token_text
        elif label == "I":
            segmented_text += token_text
        elif label == "O": # Treat "O" like "I" (append without space)
            segmented_text += token_text

    return segmented_text

# --- 4. INTERACTIVE LOOP ---
while True:
    # Get input from you
    text = input("Enter continuous Marathi text: ")
    
    if text.lower() == 'quit':
        break
        
    # Get the model's prediction
    segmented_result = predict_segmentation(text)
    
    # Print the result
    print(f"Segmented:  {segmented_result}")
    print("-" * 20)

print("Exiting test script. Goodbye!")