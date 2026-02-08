import torch
import json
import numpy as np
import argparse
from pathlib import Path
from transformers import LongformerTokenizer, LongformerForSequenceClassification, ElectraTokenizerFast, ElectraForQuestionAnswering

class AnswerabilityChecker:
    def __init__(self, check_variant="longformer"):
        """
        Initialize the answerability checker with a specific model.
        
        Args:
            check_variant: Type of checker to use ("longformer" or "electra")
        """
        self.check_variant = check_variant
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Select model based on variant
        if check_variant == "longformer":
            model_name = "potsawee/longformer-large-4096-answerable-squad2"
        elif check_variant == "electra":
            model_name = "deepset/electra-base-squad2"
        else:
            raise ValueError(f"Unknown check_variant: {check_variant}")
        
        print(f"Loading answerability checker: {model_name} on {self.device}...")
        
        # Load model and tokenizer once
        self.tokenizer, self.model = self._load_model(model_name)
        self.model.to(self.device)
        
        print("Answerability checker loaded successfully.")
    
    def _load_model(self, model_name):
        """Load the appropriate tokenizer and model based on model name."""
        if "longformer" in model_name:
            tokenizer = LongformerTokenizer.from_pretrained(model_name)
            model = LongformerForSequenceClassification.from_pretrained(model_name)
        elif "electra" in model_name:
            tokenizer = ElectraTokenizerFast.from_pretrained(model_name)
            model = ElectraForQuestionAnswering.from_pretrained(model_name)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")
        return tokenizer, model
    
    def _preprocess_inputs(self, question, context):
        """Prepare inputs based on the check variant."""
        if self.check_variant == "longformer":
            input_text = question + ' ' + self.tokenizer.sep_token + ' ' + context
            inputs = self.tokenizer(
                input_text, 
                max_length=4096, 
                truncation=True, 
                return_tensors="pt"
            )
        elif self.check_variant == "electra":
            inputs = self.tokenizer(
                question,
                context,
                return_tensors="pt",
                max_length=512,
                truncation=True,
            )
        else:
            raise ValueError(f"Unknown check_variant: {self.check_variant}")
        return inputs
    
    def _get_answerability_score(self, inputs):
        """Calculate answerability score based on the check variant."""
        if self.check_variant == "longformer":
            prob = torch.sigmoid(self.model(**inputs).logits.squeeze(-1))
            answerability_score = prob.item()

        elif self.check_variant == "electra":
            with torch.no_grad():
                outputs = self.model(**inputs)

            # 1. Get the score for the [CLS] token (index 0), which represents "No Answer"
            null_score = outputs.start_logits[0, 0] + outputs.end_logits[0, 0]

            # 2. Get the best non-null score (best valid span in the text)
            start_logits = outputs.start_logits[0, 1:]
            end_logits = outputs.end_logits[0, 1:]
            
            # Identify the highest start and end logits in the text
            best_start_val, _ = torch.max(start_logits, dim=0)
            best_end_val, _ = torch.max(end_logits, dim=0)
            best_answer_score = best_start_val + best_end_val

            # 3. Calculate Raw Difference
            # If positive, the model believes the answer exists in the text.
            # If negative, the model believes the question is unanswerable.
            raw_diff = best_answer_score - null_score
            
            # 4. Convert to 0-100 probability using Sigmoid
            answerability_score = torch.sigmoid(raw_diff).item()
        
        else:
            raise ValueError(f"Unknown check_variant: {self.check_variant}")
        return answerability_score
    
    def check_answerability(self, context, questions, threshold=0.90):
        """
        Check answerability of questions against a context.
        
        Args:
            context: The context/sentence to check questions against
            questions: List of questions to check
            threshold: Answerability score threshold (default: 0.90)
        
        Returns:
            List of answerable questions that meet the threshold
        """
        answerable_questions = []
        for question in questions:
            print(f"Checking question: {question}")
            # Prepare input
            inputs = self._preprocess_inputs(question, context).to(self.device)

            # Get model outputs
            answerability_score = self._get_answerability_score(inputs)

            status = "✅ PASS" if answerability_score >= threshold else "❌ FAIL"
            print(f"Answerability score: {answerability_score:.2f} {status}\n")
            if answerability_score >= threshold:
                answerable_questions.append(question)
        
        return answerable_questions


def process_questions_file(input_file, anscheck_type, threshold=0.90):
    """
    Process a questions JSONL file and filter by answerability.
    
    Args:
        input_file: Path to input JSONL file with questions
        anscheck_type: Type of answerability check ("longformer", "electra")
        threshold: Answerability score threshold (default: 0.90)
    """
    # Get the workspace root (2 levels up from this script in QG/code/)
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent.parent
    
    # Convert input_file to absolute path if it's relative
    input_path = Path(input_file)
    if not input_path.is_absolute():
        input_path = workspace_root / input_path
    
    # Validate input file exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Check if input file contains "-mini" suffix
    is_mini = "-mini" in input_path.stem
    
    # Generate output path: QG/{model-name}/questions-anscheck-{anscheck_type}[-mini].jsonl
    if is_mini:
        output_filename = f"questions-anscheck-{anscheck_type}-mini.jsonl"
    else:
        output_filename = f"questions-anscheck-{anscheck_type}.jsonl"
    output_path = input_path.parent / output_filename
    
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Answerability check type: {anscheck_type}")
    print(f"Threshold: {threshold}\n")
    
    # Initialize the answerability checker
    checker = AnswerabilityChecker(check_variant=anscheck_type)
    
    processed_count = 0
    total_questions_before = 0
    total_questions_after = 0
    
    # Process the input file
    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                
                context = data.get("en", "")
                questions_raw = data.get("questions", "[]")
                
                # Parse questions from JSON string format to actual list
                if isinstance(questions_raw, str):
                    try:
                        questions = json.loads(questions_raw)
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse questions on line {line_num}. Skipping.")
                        continue
                elif isinstance(questions_raw, list):
                    questions = questions_raw
                else:
                    questions = []
                
                if not context:
                    print(f"Warning: No context found in line {line_num}. Skipping.")
                    continue
                
                if not questions:
                    print(f"Warning: No questions found in line {line_num}. Skipping.")
                    continue
                
                print(f"\n--- Processing record {line_num} ---")
                print(f"Context: {context[:100]}...")
                print(f"Total questions: {len(questions)}")
                
                total_questions_before += len(questions)
                
                # Check answerability
                answerable_questions = checker.check_answerability(
                    context, 
                    questions, 
                    threshold=threshold
                )
                
                total_questions_after += len(answerable_questions)
                
                print(f"Answerable questions: {len(answerable_questions)}/{len(questions)}")
                
                # Only write records that have at least one answerable question
                if answerable_questions:
                    # Convert the list back to JSON string format to match input format
                    data["questions"] = json.dumps(answerable_questions, ensure_ascii=False)
                    f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                    f_out.flush()
                    processed_count += 1
                else:
                    print(f"No answerable questions for this record. Skipping.")
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  - Records processed: {processed_count}")
    print(f"  - Total questions before filtering: {total_questions_before}")
    print(f"  - Total questions after filtering: {total_questions_after}")
    print(f"  - Questions retained: {total_questions_after}/{total_questions_before} ({100*total_questions_after/total_questions_before if total_questions_before > 0 else 0:.1f}%)")
    print(f"  - Output saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter questions by answerability using various QA models."
    )
    
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input JSONL file with questions (e.g., QG/qwen3-4b/questions-pipeline.jsonl)"
    )
    
    parser.add_argument(
        "--anscheck_type",
        type=str,
        required=True,
        choices=["longformer", "electra"],
        help="Type of answerability check to use"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="Answerability score threshold (0-1, default: 0.90)"
    )
    
    args = parser.parse_args()
    
    process_questions_file(
        input_file=args.input_file,
        anscheck_type=args.anscheck_type,
        threshold=args.threshold
    )
