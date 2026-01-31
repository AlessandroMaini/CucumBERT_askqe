import torch
import json
import os
import re
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from qa_prompt import qa_prompt

class QuestionAnswerer:
    def __init__(self, model_id="Qwen/Qwen3-4B-Instruct-2507"):
        """
        Initializes the model and tokenizer once.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading QA model: {model_id} on {self.device}...")
        
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
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=1024,
                eos_token_id=self.terminators,
                do_sample=False, 
            )

        response = outputs[0][input_ids.shape[-1]:]
        decoded = self.tokenizer.decode(response, skip_special_tokens=True)
        
        if decoded:
            decoded = decoded.strip('"\'')
            
        return decoded

    def answer_questions(self, input_file, pipeline_type, sentence_key):
        """
        Main function to generate answers for the questions in the dataset.
        Output: QA/qwen3-4b/{language}-{pipeline_type}[-{perturbation}][-mini].jsonl
        """
        print(f"Starting QA Process.")
        
        # Get the workspace root (2 levels up from this script in QA/code/)
        script_dir = Path(__file__).resolve().parent
        workspace_root = script_dir.parent.parent
        
        # Convert input_file to absolute path if it's relative
        input_path = Path(input_file)
        if not input_path.is_absolute():
            input_path = workspace_root / input_file
        
        if not input_path.exists():
            print(f"Error: Input file {input_path} not found.")
            return
        
        # Determine if input is mini dataset
        is_mini = "-mini" in input_path.stem
        
        # Parse input path to determine language and perturbation
        input_str = str(input_path)
        
        # Check if from backtranslation
        if "backtranslation" in input_str:
            # Pattern: backtranslation/en-{language}/bt-{perturbation}.jsonl
            lang_match = re.search(r'en-([a-z]{2})', input_str)
            pert_match = re.search(r'bt-([a-z_]+)', input_path.name)
            
            language = lang_match.group(1) if lang_match else "unknown"
            perturbation = pert_match.group(1) if pert_match else None
        else:
            # From data/processed - source is English
            language = "en"
            perturbation = None
        
        # Build output filename
        output_parts = [language, pipeline_type]
        if perturbation:
            output_parts.append(perturbation)
        
        output_filename = "-".join(output_parts)
        if is_mini:
            output_filename += "-mini"
        output_filename += ".jsonl"
        
        # Generate output path
        output_dir = workspace_root / "QA" / "qwen3-4b"
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

                if data.get("id") in processed_ids:
                    continue

                sentence = data.get(sentence_key, None)
                questions = data.get("questions", None)

                if sentence and questions:
                    # Construct Prompt
                    prompt = qa_prompt.replace("{{sentence}}", sentence).replace("{{questions}}", questions)
                    
                    print(f"Prompting for ID {data.get('id')}...")
                    
                    # Generate Answer
                    generated_answers = self._call_model(prompt)

                    print(f"Processed: {sentence[:40]}...")
                    print(f"> Answer: {generated_answers[:40]}...")
                    print("-" * 40)

                    # Save
                    data['answers'] = generated_answers
                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                    f_out.flush()