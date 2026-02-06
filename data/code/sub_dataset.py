# 1. EXTRACT A SUBSET OF SENTENCES FROM THE DATASET
import json
import re
import argparse
import ast
from pathlib import Path


def extract_subset(input_file, record_ids):
    """
    Extract specific records by their ID field from a JSONL file.
    
    Args:
        input_file: Path to the input JSONL file (e.g., "data/processed/en-es.jsonl")
        record_ids: List of record IDs to extract (matching the "id" field in the JSONL)
    
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
    
    # Extract specified records by their ID field
    record_ids_set = set(record_ids)  # Convert to set for faster lookup
    selected = []
    found_ids = set()
    
    for record in all_records:
        if "id" in record and record["id"] in record_ids_set:
            selected.append(record)
            found_ids.add(record["id"])
    
    # Warn about IDs that were not found
    missing_ids = record_ids_set - found_ids
    if missing_ids:
        print(f"Warning: The following IDs were not found: {sorted(missing_ids)}")
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in selected:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"Successfully extracted {len(selected)} records to {output_path}")
    return str(output_path)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Extract specific records by their ID field from a JSONL file"
    )
    parser.add_argument(
        "input_file",
        help="Path to the input JSONL file (e.g., data/processed/en-es.jsonl)"
    )
    parser.add_argument(
        "record_ids",
        help="Python list of record IDs as a string (e.g., \"['id1', 'id2', 'id3']\")"
    )
    
    args = parser.parse_args()
    
    # Parse the list string into an actual Python list
    try:
        ids_list = ast.literal_eval(args.record_ids)
        if not isinstance(ids_list, list):
            raise ValueError("Argument must be a list")
    except (ValueError, SyntaxError) as e:
        parser.error(f"Invalid list format: {e}. Expected format: \"['id1', 'id2', 'id3']\"")
    
    extract_subset(args.input_file, ids_list)