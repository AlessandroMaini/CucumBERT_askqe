import json
import os
import argparse
import sys
import shutil

def load_questions_map(qg_file):
    """
    Loads questions into a dictionary {id: questions} from the QG output file.
    """
    questions_map = {}
    if not os.path.exists(qg_file):
        print(f"Error: QG output file '{qg_file}' not found.")
        sys.exit(1)

    print(f"Loading questions from: {qg_file}...")
    count = 0
    with open(qg_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                if 'id' in record and 'questions' in record:
                    questions_map[record['id']] = record['questions']
                    count += 1
            except json.JSONDecodeError:
                pass
    
    print(f"Loaded {count} questions into memory.")
    return questions_map

def merge_questions(target_file, output_file, questions_map):
    """
    Reads target_file, adds questions from the map, and writes to output_file.
    """
    if not os.path.exists(target_file):
        print(f"Error: Target file '{target_file}' not found.")
        sys.exit(1)

    # Use a temp file to allow safe overwriting of the input file
    temp_output = output_file + ".tmp"
    
    print(f"Processing: {target_file} -> {output_file}")
    
    matched_count = 0
    missing_count = 0

    try:
        with open(target_file, 'r', encoding='utf-8') as f_in, \
             open(temp_output, 'w', encoding='utf-8') as f_out:
            
            for line in f_in:
                try:
                    record = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                record_id = record.get('id')

                # Logic to add questions
                if record_id and record_id in questions_map:
                    record['questions'] = questions_map[record_id]
                    matched_count += 1
                else:
                    record['questions'] = []
                    # Only warn if ID exists but wasn't found in map
                    if record_id:
                        missing_count += 1
                        # Optional: Uncomment to see specific missing IDs
                        # print(f"Warning: ID '{record_id}' not found in QG map.")

                f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
    
        # Move temp file to final destination
        shutil.move(temp_output, output_file)
        print(f"Success! Matched: {matched_count}, Missing: {missing_count}")
        print("-" * 40)

    except Exception as e:
        print(f"An error occurred: {e}")
        if os.path.exists(temp_output):
            os.remove(temp_output)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge 'questions' column from a QG file into a target JSONL dataset.")
    
    parser.add_argument("--qg_file", type=str, required=True, help="Path to the file containing generated questions")
    parser.add_argument("--target_file", type=str, required=True, help="Path to the backtranslated file to update")
    parser.add_argument("--output_file", type=str, default=None, help="Output path (optional). If not provided, overwrites target_file.")

    args = parser.parse_args()

    # Default to overwriting the target file if no output is specified
    output_path = args.output_file if args.output_file else args.target_file

    # 1. Load the Map
    q_map = load_questions_map(args.qg_file)

    # 2. Merge
    merge_questions(args.target_file, output_path, q_map)