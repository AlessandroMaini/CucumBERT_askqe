import torch
import json
import numpy as np
from transformers import LongformerTokenizer, LongformerForSequenceClassification, ElectraTokenizerFast, ElectraForQuestionAnswering


class AnswerabilityChecker:
    def __init__(self, check_variant="longformer"):
        """
        Initialize the answerability checker with a specific model.
        
        Args:
            check_variant: Type of checker to use ("longformer", "electra", or "electra-null")
        """
        self.check_variant = check_variant
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Select model based on variant
        if check_variant == "longformer":
            model_name = "potsawee/longformer-large-4096-answerable-squad2"
        elif check_variant == "electra" or check_variant == "electra-null":
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
        elif self.check_variant == "electra" or self.check_variant == "electra-null":
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
            answerability_score = prob.item() * 100

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
            answerability_score = torch.sigmoid(raw_diff).item() * 100
        
        elif self.check_variant == "electra-null":
            with torch.no_grad():
                outputs = self.model(**inputs)

            # 1. Get the score for the [CLS] token (index 0)
            null_score = outputs.start_logits[0, 0] + outputs.end_logits[0, 0]
            
            # 2. Invert it to get "Answerability"
            # If null_score is HIGH (e.g., +5), -null_score is LOW (-5) -> Sigmoid is ~0 (Not Answerable)
            # If null_score is LOW (e.g., -5), -null_score is HIGH (+5) -> Sigmoid is ~1 (Answerable)
            answerability_logit = -null_score
            
            # 3. Apply Sigmoid
            answerability_score = torch.sigmoid(answerability_logit).item() * 100
        
        else:
            raise ValueError(f"Unknown check_variant: {self.check_variant}")
        return answerability_score
    
    def check_answerability(self, context, questions, threshold=90.0):
        """
        Check answerability of questions against a context.
        
        Args:
            context: The context/sentence to check questions against
            questions: List of questions to check
            threshold: Answerability score threshold (default: 90.0)
        
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