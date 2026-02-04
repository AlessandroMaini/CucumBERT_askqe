import torch
import json
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from qg_prompt import prompts
from answerability_check import AnswerabilityChecker 

class QuestionGenerator:
    def __init__(self, model_id="Qwen/Qwen3-4B-Instruct-2507"):
        """
        Initializes the model and tokenizer once.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading QG model: {model_id} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            cache_dir="",
            device_map="auto",
        )
        
        # Define terminators
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        ]
        print("Model loaded successfully.")

    def _call_model(self, prompt):
        """
        Internal method to handle the specific Qwen generation logic.
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
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=1024,
                eos_token_id=self.terminators,
                do_sample=False, 
            )

        # Slice to get only new tokens
        response = outputs[0][input_ids.shape[-1]:]
        decoded = self.tokenizer.decode(response, skip_special_tokens=True)
        
        # Clean quotes if present
        if decoded:
            decoded = decoded.strip('"\'')
            
        return decoded

    def generate_questions(self, input_file, prompt_variant, model_name="qwen3-4b", check_variant=None):
        """
        Main function to process the dataset and generate questions.
        Expected input: QG/entailed_facts.jsonl or QG/entailed_facts-mini.jsonl
        Output is saved to QG/{model_name}/questions-{prompt_variant}[-mini].jsonl
        
        Args:
            check_variant: Optional. When prompt_variant is "anscheck", specifies which
                          answerability checker to use.
        """
        print(f"Starting QG ({prompt_variant}).")
        
        # Initialize answerability checker if needed
        answerability_checker = None
        if prompt_variant == "anscheck" and check_variant:
            answerability_checker = AnswerabilityChecker(check_variant=check_variant)
        
        # Get the workspace root (2 levels up from this script in QG/code/)
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
        
        if prompt_variant == "anscheck" and check_variant:
            # Special naming for anscheck variant
            if is_mini:
                output_filename = f"questions-anscheck-{check_variant}-mini.jsonl"
            else:
                output_filename = f"questions-anscheck-{check_variant}.jsonl"
        else:
            if is_mini:
                output_filename = f"questions-{prompt_variant}-mini.jsonl"
            else:
                output_filename = f"questions-{prompt_variant}.jsonl"
        
        output_path = output_dir / output_filename
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")

        # --- 1. Resume Logic ---
        processed_ids = set()
        if output_path.exists():
            print("Found existing output file. Resuming...")
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        processed_ids.add(data.get("id"))
                    except: pass

        # --- 2. Processing Loop ---
        with open(input_path, 'r', encoding='utf-8') as f_in, open(output_path, 'a', encoding='utf-8') as f_out:
            for line in f_in:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Skip if already processed
                if data.get("id") in processed_ids:
                    continue
                
                sentence = data.get('en', None)

                if sentence:
                    # --- Prompt Construction Logic ---
                    prompt_template = prompts.get(prompt_variant, prompts.get('vanilla', ''))
                    
                    if prompt_variant == "semantic":
                        semantic = data.get('semantic_roles', None)
                        if semantic:
                            prompt = prompt_template.replace("{{sentence}}", sentence).replace("{{semantic_roles}}", str(semantic))
                        else:
                            # Fallback if semantic roles missing
                            prompt = prompt_template.replace("{{sentence}}", sentence) # Might need vanilla template fallback depending on your prompts file

                    elif prompt_variant == "atomic":
                        atomics = data.get('atomic_facts', None)
                        if atomics:
                            prompt = prompt_template.replace("{{sentence}}", sentence).replace("{{atomic_facts}}", str(atomics))
                        else:
                            # Fallback if atomic facts missing
                            prompt = prompt_template.replace("{{sentence}}", sentence)

                    else:  # Vanilla/AnsCheck case
                        prompt = prompt_template.replace("{{sentence}}", sentence)

                    # --- Generation ---
                    print(f"Prompting for ID {data.get('id')}...")
                    generated_questions = self._call_model(prompt)

                    print(f"Generated: {generated_questions[:60]}...") # Print preview
                    print("-" * 40)

                    # --- Answerability Check (if applicable) ---
                    if answerability_checker:
                        print(f"Running answerability check ({check_variant})...")
                        generated_questions = answerability_checker.check_answerability(
                            context=sentence,
                            questions=generated_questions
                        )
                        print(f"Check completed.")

                    # --- Saving ---
                    data['questions'] = generated_questions
                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                    f_out.flush() # Ensure write