# 1. EXTRACT A SUBSET OF SENTENCES FROM THE DATASET
import json
import random
import re
import argparse
from pathlib import Path


def extract_subset(input_file, n=50):
    """
    Extract a random subset of records from a JSONL file.
    
    Args:
        input_file: Path to the input JSONL file (e.g., "data/processed/en-es.jsonl")
        n: Number of records to extract (default: 50)
    
    Returns:
        Path to the output file
    """
    # Get the workspace root (2 levels up from this script)
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent.parent
    
    # Convert input_file to absolute path if it's relative
    input_path = Path(input_file)
    if not input_path.is_absolute():
        input_path = workspace_root / input_path
    
    # Extract language pair from input file path
    # Example: "data/processed/en-es.jsonl" -> "es"
    match = re.search(r'en-([a-z]{2})', str(input_path))
    if not match:
        raise ValueError(f"Could not extract language from input file: {input_path}")
    
    language = match.group(1)
    output_path = workspace_root / "data" / "processed" / f"en-{language}-mini.jsonl"
    
    # Load all valid records
    all_records = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                all_records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    # Extract N sentences at random
    if len(all_records) > n:
        selected = random.sample(all_records, n)
    else:
        selected = all_records
        print(f"Note: Only {len(all_records)} records available.")
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in selected:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"Successfully extracted {len(selected)} records to {output_path}")
    return str(output_path)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Extract a random subset of records from a JSONL file"
    )
    parser.add_argument(
        "input_file",
        help="Path to the input JSONL file (e.g., data/processed/en-es.jsonl)"
    )
    parser.add_argument(
        "-n", "--num-records",
        type=int,
        default=50,
        help="Number of records to extract (default: 50)"
    )
    
    args = parser.parse_args()
    extract_subset(args.input_file, args.num_records)