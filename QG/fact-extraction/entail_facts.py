import json
import torch
import argparse
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model(model_name):
    """
    Loads the model and tokenizer to the appropriate device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Entailment model: {model_name} on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()
    
    print("Model loaded successfully.")
    return tokenizer, model, device

def check_entailment(tokenizer, model, device, premise, hypothesis):
    """
    Scores a single premise-hypothesis pair.
    """
    inputs = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=[(hypothesis, premise)],
        add_special_tokens=True, 
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        # Softmax over the classes (dim=-1) to get probabilities
        probs = torch.softmax(logits, dim=-1)[0]
        
        # Taking index 0 as per your original logic
        entail_score = probs[0].item()
        
    return entail_score

def process_dataset(input_file, tokenizer, model, device, threshold):
    """
    Reads input file, filters atomic facts based on entailment, and saves to output.
    Output is saved to QG/entailed_facts[-mini].jsonl
    """
    print(f"Starting Entailment Filtering with threshold {threshold}.")
    
    # Get the workspace root (2 levels up from this script in QG/fact-extraction/)
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent.parent
    
    # Convert input_file to absolute path if it's relative
    input_path = Path(input_file)
    if not input_path.is_absolute():
        input_path = workspace_root / input_path
    
    # Determine if input is mini dataset
    is_mini = "-mini" in input_path.stem
    
    # Generate output path
    output_dir = workspace_root / "QG"
    
    if is_mini:
        output_filename = "entailed_facts-mini.jsonl"
    else:
        output_filename = "entailed_facts.jsonl"
    
    output_path = output_dir / output_filename
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

    with open(input_path, 'r', encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
        for line_num, line in enumerate(f_in):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            premise = data.get('en', "")
            raw_facts_str = data.get('atomic_facts', "")

            # --- 1. Parsing Logic ---
            atomic_facts = []
            if isinstance(raw_facts_str, str) and raw_facts_str.strip():
                try:
                    atomic_facts = json.loads(raw_facts_str)
                except Exception:
                    # Fallback: sometimes the string might be just a sentence, but usually it's a JSON list string
                    atomic_facts = []
            elif isinstance(raw_facts_str, list):
                atomic_facts = raw_facts_str
            
            # --- 2. Filtering Logic ---
            print(f"[{line_num}] Processing ID: {data.get('id', 'Unknown')}")
            entailed_facts = []

            for hypothesis in atomic_facts:
                if not hypothesis:
                    continue
                
                score = check_entailment(tokenizer, model, device, premise, hypothesis)
                is_entailed = score > threshold
                
                status = "✅ PASS" if is_entailed else "❌ FAIL"
                print(f"  {status} {score:.4f} | {hypothesis[:50]}...")

                if is_entailed:
                    entailed_facts.append(hypothesis)

            # --- 3. Saving ---
            # Update the record with only the valid facts (stored as JSON string)
            data['atomic_facts'] = json.dumps(entailed_facts, ensure_ascii=False)
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
            f_out.flush()

    print(f"\nProcessing complete. Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Entailment Filtering locally.")
    
    parser.add_argument(
        "--input_file", 
        type=str, 
        required=True, 
        help="Path to the input .jsonl file (e.g., 'QG/atomic_facts.jsonl' or 'QG/atomic_facts-mini.jsonl')"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="potsawee/deberta-v3-large-mnli", 
        help="HuggingFace model ID for entailment (default: potsawee/deberta-v3-large-mnli)"
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.9, 
        help="Probability threshold for entailment (default: 0.9)"
    )

    args = parser.parse_args()

    # Load model once
    tokenizer, model, device = load_model(args.model_name)

    # Run processing
    process_dataset(
        input_file=args.input_file,
        tokenizer=tokenizer,
        model=model,
        device=device,
        threshold=args.threshold
    )