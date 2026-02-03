import requests
import zipfile
import json
import os
import re

# --- CONFIGURATION ---
# Source URL for the English-Spanish DGT legal dataset (Moses format)
URL = "https://object.pouta.csc.fi/OPUS-DGT/v4/moses/en-es.txt.zip"
ZIP_NAME = "en-es.txt.zip"

FILE_EN = "DGT.en-es.en"
FILE_ES = "DGT.en-es.es"
FILE_IDS = "DGT.en-es.ids"
FILE_README = "README"

# Output settings
OUTPUT_JSON = "../processed/en-es-new.json"
MAX_SAMPLES = 1000  

# --- FILTER SETTINGS ---
MIN_CHARS = 50
MAX_CHARS = 250
MIN_ALPHA_RATIO = 0.7
MIN_WORDS = 8
MAX_UPPER_RATIO = 0.2

QUESTION_START_EN = re.compile(r"^(who|what|when|where|why|how|which|whom|whose)\b", re.IGNORECASE)
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
LONG_NUM_RE = re.compile(r"\d{6,}")

def alpha_ratio(text: str) -> float:
    letters = sum(1 for c in text if c.isalpha())
    total = max(len(text), 1)
    return letters / total

def upper_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 1.0
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters)

def is_question_like(text: str) -> bool:
    if text.endswith("?"):
        return True
    if QUESTION_START_EN.search(text):
        return True
    return False

def looks_like_heading(text: str) -> bool:
    return upper_ratio(text) > MAX_UPPER_RATIO

def is_valid_sentence(text: str) -> bool:
    if not text:
        return False
    if URL_RE.search(text) or EMAIL_RE.search(text):
        return False
    if LONG_NUM_RE.search(text):
        return False
    if len(text) < MIN_CHARS or len(text) > MAX_CHARS:
        return False
    if alpha_ratio(text) < MIN_ALPHA_RATIO:
        return False
    if len(text.split()) < MIN_WORDS:
        return False
    if looks_like_heading(text):
        return False
    if is_question_like(text):
        return False
    return True

def download_data(url, filename):
    """Downloads the dataset from the provided URL."""
    if os.path.exists(filename):
        print(f"{filename} already exists. Skipping download.")
        return
    
    print(f"Downloading dataset from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    except Exception as e:
        print(f"Error during download: {e}")

def extract_data(zip_path):
    """Extracts the zip archive to the current directory."""
    print(f"Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("Extraction complete.")
    except Exception as e:
        print(f"Error during extraction: {e}")

def convert_to_json(en_path, es_path, output_path, limit):
    """Parses Moses files and creates a filtered JSON dataset."""
    print(f"Generating JSON dataset (Limit: {limit} samples)...")
    data_list = []
    
    if not os.path.exists(en_path) or not os.path.exists(es_path):
        print(f"Error: Source files {en_path} or {es_path} not found.")
        return

    try:
        with open(en_path, "r", encoding="utf-8") as f_en, \
             open(es_path, "r", encoding="utf-8") as f_es:
            
            counter = 0
            for en_line, es_line in zip(f_en, f_es):
                en_text = en_line.strip()
                es_text = es_line.strip()
                
                # Filtering logic: Select clean declarative sentences suitable for Question Generation
                if is_valid_sentence(en_text):
                    data_list.append({
                        "id": counter,
                        "en": en_text,
                        "es": es_text,
                    })
                    counter += 1
                
                if counter >= limit:
                    break
        
        # Write to JSON file
        with open(output_path, "w", encoding="utf-8") as json_file:
            json.dump(data_list, json_file, ensure_ascii=False, indent=4)
        
        print(f"Success! Created {output_path} with {len(data_list)} entries.")
        
    except Exception as e:
        print(f"Error during JSON conversion: {e}")

def cleanup_files(*paths):
    """Deletes temporary/downloaded files if they exist."""
    for path in paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
                print(f"Deleted {path}.")
            except Exception as e:
                print(f"Error deleting {path}: {e}")

if __name__ == "__main__":

    download_data(URL, ZIP_NAME)
    extract_data(ZIP_NAME)
    convert_to_json(FILE_EN, FILE_ES, OUTPUT_JSON, MAX_SAMPLES)
    cleanup_files(ZIP_NAME, FILE_EN, FILE_ES, FILE_IDS, FILE_README)