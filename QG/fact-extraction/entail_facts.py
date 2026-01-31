import json
import torch
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class FactEntailmentFilter:
    def __init__(self, model_name="potsawee/deberta-v3-large-mnli", threshold=0.5):
        """
        Initializes the NLI model and tokenizer once.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        
        print(f"Loading Entailment model: {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("Model loaded successfully.")

    def _check_entailment(self, premise, hypothesis):
        """
        Internal method to score a single premise-hypothesis pair.
        """
        inputs = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=[(hypothesis, premise)],
            add_special_tokens=True, 
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            entail_score = probs[0].item()
            
        return entail_score

    def process_dataset(self, input_file):
        """
        Reads input file, filters atomic facts based on entailment, and saves to output.
        Output is saved in the same directory as input with name entailed_facts[-mini].jsonl
        """
        print(f"Starting Entailment Filtering.")
        
        # Get the workspace root (2 levels up from this script in QG/fact-extraction/)
        script_dir = Path(__file__).resolve().parent
        workspace_root = script_dir.parent.parent
        
        # Convert input_file to absolute path if it's relative
        input_path = Path(input_file)
        if not input_path.is_absolute():
            input_path = workspace_root / input_path
        
        # Determine if input is mini dataset
        is_mini = "-mini" in input_path.stem
        
        # Generate output path in the same directory as input
        if is_mini:
            output_filename = "entailed_facts-mini.jsonl"
        else:
            output_filename = "entailed_facts.jsonl"
        
        output_path = input_path.parent / output_filename
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")

        with open(input_path, 'r', encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                premise = data.get('en', "")
                raw_facts_str = data.get('atomic_facts', "")

                # --- 1. Parsing Logic (Preserved from snippet) ---
                atomic_facts = []
                if isinstance(raw_facts_str, str) and raw_facts_str.strip():
                    try:
                        atomic_facts = json.loads(raw_facts_str)
                    except Exception:
                        atomic_facts = []
                elif isinstance(raw_facts_str, list):
                    atomic_facts = raw_facts_str
                
                # --- 2. Filtering Logic ---
                print(f"Processing ID: {data.get('id', 'Unknown')}")
                entailed_facts = []

                for hypothesis in atomic_facts:
                    if not hypothesis:
                        continue
                    
                    score = self._check_entailment(premise, hypothesis)
                    is_entailed = score > self.threshold
                    
                    status = "✅ PASS" if is_entailed else "❌ FAIL"
                    print(f"[{status}] {score:.4f} | {hypothesis[:50]}...")

                    if is_entailed:
                        entailed_facts.append(hypothesis)

                # --- 3. Saving ---
                # Update the record with only the valid facts
                data['atomic_facts'] = entailed_facts
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                f_out.flush()

        print(f"\nProcessing complete. Results saved to {output_path}")