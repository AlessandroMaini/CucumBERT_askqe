import json
import os
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from fact_prompt import atomic_fact_prompt 

class AtomicFactsExtractor:
    def __init__(self, model_id="Qwen/Qwen3-4B-Instruct-2507"):
        """
        Initializes the model and tokenizer once.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Atomic Facts model: {model_id} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            cache_dir="",
            device_map="auto",
        )
        
        # Pre-calculate terminators
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            self.tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        ]
        print("Model loaded successfully.")

    def _call_model(self, prompt):
        """
        Internal method to handle the generation logic.
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=1024,
                eos_token_id=self.terminators,
                do_sample=False,
            )

        response = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True)

    def extract_facts(self, input_file, model_name="qwen3-4b"):
        """
        Reads the input file, extracts atomic facts, and saves to the output file.
        Output is saved to QG/{model_name}/atomic_facts[-mini].jsonl
        """
        print(f"Starting Atomic Fact Extraction.")
        
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
        output_dir = workspace_root / "QG" / model_name
        if is_mini:
            output_filename = "atomic_facts-mini.jsonl"
        else:
            output_filename = "atomic_facts.jsonl"
        
        output_path = output_dir / output_filename
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")

        with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
            for line in f_in:
                data = json.loads(line)
                
                if "en" in data:
                    sentence = data["en"]
                    
                    # Construct prompt
                    prompt = atomic_fact_prompt.replace("{{sentence}}", sentence)
                    
                    print(f"Processing: {sentence[:50]}...")
                    
                    # Generate response
                    raw_response = self._call_model(prompt)
                    
                    # Clean response
                    clean_response = raw_response.strip('"\n \'')

                    print(f"> Facts: {clean_response[:50]}...")
                    print("=" * 40)

                    # Save data
                    data["atomic_facts"] = clean_response
                    f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                    f_out.flush() # Ensure data is written incrementally