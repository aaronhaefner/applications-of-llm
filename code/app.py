# Setup
import os
import sys
import logging
import json
from utils.globals import HFTOKEN
from utils.utils import (set_device, load_tokenizer_model, generate_questions,
                         generate_multiple_sql_queries, generate_sql_query,
                         evaluate_batch, evaluate_generated_sql)
from main import fine_tune_training
from dotenv import load_dotenv
load_dotenv()

HEALTH_DATASET = "Nicolybgs/healthcare_data"
EXAMPLES = "../input/question_query.json"
MODEL_ID = "PASS"

if __name__ == '__main__':
    device = set_device()
    model_name = "general_model"  # google/flan-t5-base
    tokenizer, model = load_tokenizer_model(model_name, device)
    examples = "../input/question_query.json"
    fine_tune_training(tokenizer, model, examples, device)
    
    # Load the fine-tuned model
    model_name = "domain_model"  # Fine-tuned model name
    tokenizer, model = load_tokenizer_model(model_name, device)

    # Input the question and expected SQL query
    input_question = "Which NPIs have practice locations in the state of California ordered by pcredential?"
    expected_sql = "SELECT npi FROM nppes WHERE plocstatename = 'CA' ORDER BY pcredential;"

    # Generate the corresponding SQL query for the input question
    generated_sql_query = generate_sql_query(input_question, model, tokenizer, device)
    logging.info(f"Generated SQL Query: {generated_sql_query}")

    # Evaluate the generated SQL query against the expected result
    # evaluation_result = evaluate_generated_sql(generated_sql_query, expected_sql)
    # logging.info(f"Evaluation Result: Match: {evaluation_result['match']}, BLEU Score: {evaluation_result['bleu_score']}")
