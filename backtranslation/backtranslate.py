import json
import os
import argparse
import time
import sys
from pathlib import Path
from deep_translator import GoogleTranslator

def get_existing_ids(output_path):
    """Reads the output file to find IDs that are already processed."""
    processed_ids = set()
    if os.path.exists(output_path):
        print(f"Resuming from existing file: {output_path}")
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if "id" in data:
                        processed_ids.add(data["id"])
                except json.JSONDecodeError:
                    pass
    return processed_ids

def run_backtranslation(input_file, source_lang, target_lang):
    # Get the workspace root (1 level up from this script in backtranslation/)
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent
    
    # Convert input_file to absolute path if it's relative
    input_path = Path(input_file)
    if not input_path.is_absolute():
        input_path = workspace_root / input_file
    
    # 1. Validate Input
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found.")
        sys.exit(1)
    
    # Check if mini dataset
    is_mini = "-mini" in input_path.parent.name
    
    # Extract perturbation from filename
    perturbation = input_path.stem
    
    # Generate output path: backtranslation/en-{source_lang}[-mini]/bt-{perturbation}.jsonl
    if is_mini:
        output_dir = script_dir / f"en-{source_lang}-mini"
    else:
        output_dir = script_dir / f"en-{source_lang}"
    
    output_path = output_dir / f"bt-{perturbation}.jsonl"

    # 2. Setup Translator
    # Note: deep_translator makes API calls, so it requires internet access.
    try:
        translator = GoogleTranslator(source=source_lang, target=target_lang)
    except Exception as e:
        print(f"Error initializing translator: {e}")
        sys.exit(1)

    # 3. Prepare Resume Logic
    processed_ids = get_existing_ids(str(output_path))
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"--- Starting Backtranslation ---")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Mode:   {source_lang} -> {target_lang}")

    # 4. Processing Loop (Streaming)
    success_count = 0
    error_count = 0
    
    # We open output in 'append' mode to support resuming
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'a', encoding='utf-8') as f_out:
        
        for line in f_in:
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            # Skip if already done
            if data.get("id") in processed_ids:
                continue

            # Determine the key to translate
            # Based on your logic: pert_es, pert_fr, etc.
            source_key = f"pert_{source_lang}"
            
            # Key where we save the result
            # Based on your logic: bt_pert_es
            target_key = f"bt_{source_key}"

            text_to_translate = data.get(source_key)

            if text_to_translate:
                try:
                    # Perform Translation
                    translated_text = translator.translate(text_to_translate)
                    
                    # Store result
                    data[target_key] = translated_text
                    success_count += 1
                    
                    # Optional: Print progress every 10 items or on error
                    if success_count % 10 == 0:
                        print(f"Processed {success_count} lines...", end='\r')

                except Exception as e:
                    # Handle API errors (timeouts, rate limits)
                    print(f"\n[Error] ID {data.get('id')}: {e}")
                    data[target_key] = "" # Save empty string or handle differently
                    error_count += 1
                    time.sleep(2) # Wait a bit before retrying next line
            
            else:
                # If the source key is missing, just write the record as is (or skip)
                pass

            # Write immediately
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
            f_out.flush()

    print(f"\n\nDone. Success: {success_count}, Errors: {error_count}")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtranslate a JSONL dataset using Google Translate.")
    
    parser.add_argument("input_file", type=str, help="Path to the input .jsonl file (e.g., perturbed_DGT/en-es/alteration.jsonl)")
    parser.add_argument("--source_lang", type=str, required=True, help="Source language code (e.g., 'es')")
    parser.add_argument("--target_lang", type=str, default="en", help="Target language code (default: 'en')")

    args = parser.parse_args()

    run_backtranslation(
        input_file=args.input_file,
        source_lang=args.source_lang,
        target_lang=args.target_lang
    )