import json
import os
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from perturb_prompt import prompts 

class PerturbationEngine:
    def __init__(self, model_id="Qwen/Qwen3-4B-Instruct-2507"):
        """
        Initializes the model and tokenizer once.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.language_map = {
            "es": "Spanish",
            "fr": "French",
            "hi": "Hindi",
            "tl": "Tagalog",
            "zh": "Chinese"
        }
        
        print(f"Loading model: {model_id} on {self.device}...")
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

    def perturb_dataset(self, input_file, language, perturbation_type):
        """
        Performs a specific perturbation over a specified dataset.
        Output is saved to contratico/en-{language}[-mini]/{perturbation_type}.jsonl
        """
        print(f"Starting perturbation: {perturbation_type} for language: {language}")
        
        # Get the workspace root (1 level up from this script in contratico/)
        script_dir = Path(__file__).resolve().parent
        workspace_root = script_dir.parent
        
        # Convert input_file to absolute path if it's relative
        input_path = Path(input_file)
        if not input_path.is_absolute():
            input_path = workspace_root / input_path
        
        # Determine if input is mini dataset
        is_mini = "-mini" in input_path.stem
        
        # Generate output path
        if is_mini:
            output_dir = script_dir / f"en-{language}-mini"
        else:
            output_dir = script_dir / f"en-{language}"
        
        output_path = output_dir / f"{perturbation_type}.jsonl"
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Output will be saved to: {output_path}")
        
        target_lang = self.language_map.get(language, language)
        prompt_key = f"{perturbation_type}_{language}"

        # Check if the specific prompt exists
        if prompt_key not in prompts:
            print(f"Warning: Prompt key '{prompt_key}' not found. Skipping.")
            return

        with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
            for line in f_in:
                data = json.loads(line)
                
                # Check if the source language exists in the record
                if language in data:
                    sentence = data[language]
                    
                    # Format prompt
                    prompt_template = prompts[prompt_key]
                    prompt = prompt_template.replace("{{original}}", sentence).replace("{{target_lang}}", target_lang)
                    
                    print(f"Prompt: {prompt[:100]}...") # Print first 100 chars for verification
                    
                    # Generate response
                    response = self._call_model(prompt)
                    clean_response = response.strip('"\n ')
                    
                    print(f"> {clean_response[:100]}...")
                    print("=" * 40)

                    # Update data record
                    data["perturbation"] = perturbation_type
                    data[f"pert_{language}"] = clean_response
                    
                    # Write to output
                    f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                    f_out.flush() # Ensure data is written incrementally