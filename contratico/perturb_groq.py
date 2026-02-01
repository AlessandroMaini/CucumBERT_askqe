import json
import os
import argparse
from pathlib import Path
from groq import Groq
from perturb_prompt import prompts 

# Global configuration
LANGUAGE_MAP = {
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "tl": "Tagalog",
    "zh": "Chinese"
}

def call_groq_model(client, prompt, model_id):
    """
    Handles the API call to Groq.
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=model_id,
            temperature=0, # Low temperature for more deterministic results usually desired in data gen
            max_tokens=1024,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return ""

def perturb_dataset(input_file, language, perturbation_type, model_id, api_key):
    """
    Performs a specific perturbation over a specified dataset using Groq API.
    """
    print(f"Starting perturbation: {perturbation_type} for language: {language}")
    
    # Initialize Groq Client
    client = Groq(api_key=api_key)
    
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
    
    target_lang = LANGUAGE_MAP.get(language, language)
    prompt_key = f"{perturbation_type}_{language}"

    # Check if the specific prompt exists
    if prompt_key not in prompts:
        print(f"Warning: Prompt key '{prompt_key}' not found in prompts.py. Skipping.")
        return

    # Process file
    with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for line_num, line in enumerate(f_in):
            if not line.strip():
                continue
                
            data = json.loads(line)
            
            # Check if the source language exists in the record
            if language in data:
                sentence = data[language]
                
                # Format prompt
                prompt_template = prompts[prompt_key]
                prompt = prompt_template.replace("{{original}}", sentence).replace("{{target_lang}}", target_lang)
                
                print(f"[{line_num}] Processing prompt...")
                
                # Generate response via API
                response = call_groq_model(client, prompt, model_id)
                clean_response = response.strip('"\n ')
                
                # Update data record
                data["perturbation"] = perturbation_type
                data[f"pert_{language}"] = clean_response
                
                # Write to output
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                f_out.flush() # Ensure data is written incrementally

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run perturbation using Groq API.")
    
    parser.add_argument(
        "--input_file", 
        type=str, 
        required=True, 
        help="Path to the input .jsonl file (e.g., 'data/input.jsonl')"
    )
    parser.add_argument(
        "--language", 
        type=str, 
        required=True, 
        choices=LANGUAGE_MAP.keys(), 
        help="Target language code (es, fr, hi, tl, zh)"
    )
    parser.add_argument(
        "--perturbation_type", 
        type=str, 
        required=True, 
        help="Type of perturbation (must exist in prompts.py, e.g., 'formal', 'typos')"
    )
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="llama-3.3-70b-versatile", 
        help="Groq model ID to use (default: llama-3.3-70b-versatile)"
    )
    parser.add_argument(
        "--api_key", 
        type=str, 
        default=os.environ.get("GROQ_API_KEY"),
        help="Groq API Key. Defaults to GROQ_API_KEY env var."
    )

    args = parser.parse_args()

    if not args.api_key:
        print("Error: Groq API Key is required. Set GROQ_API_KEY env var or pass --api_key.")
        exit(1)

    perturb_dataset(
        input_file=args.input_file, 
        language=args.language, 
        perturbation_type=args.perturbation_type,
        model_id=args.model_id,
        api_key=args.api_key
    )