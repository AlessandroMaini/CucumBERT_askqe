# Extract random IDs from a JSONL file
import json
import random
import argparse
from pathlib import Path


def extract_random_ids(input_file, n):
    """
    Extract N random IDs from a JSONL file.
    
    Args:
        input_file: Path to the input JSONL file (e.g., "data/processed/en-es.jsonl")
        n: Number of random IDs to extract
    
    Returns:
        String representation of a Python list containing the random IDs
    """
    # Get the workspace root (2 levels up from this script)
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent.parent
    
    # Convert input_file to absolute path if it's relative
    input_path = Path(input_file)
    if not input_path.is_absolute():
        input_path = workspace_root / input_path
    
    # Check if the file exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if not input_path.is_file():
        raise ValueError(f"Path is not a file: {input_path}")
    
    # Load all IDs from the file
    all_ids = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                if "id" in record:
                    all_ids.append(record["id"])
            except json.JSONDecodeError:
                continue
    
    if not all_ids:
        raise ValueError(f"No records with 'id' field found in {input_path}")
    
    # Select N random IDs
    if n > len(all_ids):
        print(f"Warning: Requested {n} IDs but only {len(all_ids)} available. Returning all IDs.", flush=True)
        selected_ids = all_ids
    else:
        selected_ids = random.sample(all_ids, n)
    
    # Return as Python list string representation
    return str(selected_ids)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Extract N random IDs from a JSONL file"
    )
    parser.add_argument(
        "input_file",
        help="Path to the input JSONL file (e.g., data/processed/en-es.jsonl)"
    )
    parser.add_argument(
        "n",
        type=int,
        help="Number of random IDs to extract"
    )
    
    args = parser.parse_args()
    
    try:
        result = extract_random_ids(args.input_file, args.n)
        print(result)
    except Exception as e:
        parser.error(str(e))