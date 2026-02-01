import json
import os
import argparse
from pathlib import Path
from groq import Groq
from fact_prompt import atomic_fact_prompt 

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
            temperature=0, # Deterministic for extraction tasks
            max_tokens=1024,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return ""

def extract_facts(input_file, model_name, groq_model_id, api_key):
    """
    Reads the input file, extracts atomic facts using Groq, and saves to the output file.
    Output is saved to QG/{model_name}/atomic_facts[-mini].jsonl
    """
    print(f"Starting Atomic Fact Extraction using {groq_model_id}...")
    
    # Initialize Groq Client
    client = Groq(api_key=api_key)
    
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
    # Note: model_name is used for the folder structure, groq_model_id is used for the API
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

    # Process file
    with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for line_num, line in enumerate(f_in):
            if not line.strip():
                continue

            data = json.loads(line)
            
            if "en" in data:
                sentence = data["en"]
                
                # Construct prompt
                prompt = atomic_fact_prompt.replace("{{sentence}}", sentence)
                
                print(f"[{line_num}] Processing: {sentence[:50]}...")
                
                # Generate response
                raw_response = call_groq_model(client, prompt, groq_model_id)
                
                # Clean response
                clean_response = raw_response.strip('"\n \'')

                # Save data
                data["atomic_facts"] = clean_response
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                f_out.flush() # Ensure data is written incrementally

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Atomic Fact Extraction using Groq API.")
    
    parser.add_argument(
        "--input_file", 
        type=str, 
        required=True, 
        help="Path to the input .jsonl file (e.g., 'data/input.jsonl')"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="llama-70b", 
        help="Name used for the output folder structure (e.g., 'llama-70b')"
    )
    parser.add_argument(
        "--groq_model_id", 
        type=str, 
        default="llama-3.3-70b-versatile", 
        help="The actual Groq model ID to call via API"
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

    extract_facts(
        input_file=args.input_file, 
        model_name=args.model_name,
        groq_model_id=args.groq_model_id,
        api_key=args.api_key
    )