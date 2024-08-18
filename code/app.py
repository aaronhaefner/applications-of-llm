# Setup
import os
import sys
import logging
import json
from utils.globals import HFTOKEN
from utils.utils import (set_device, load_tokenizer_model)
from utils.generative_utils import (generate_sql_query,
                                    generate_questions,
                                    generate_multiple_questions,
                                    evaluate_generated_sql)
from main import fine_tune_training
from dotenv import load_dotenv
load_dotenv()

HEALTH_DATASET = "Nicolybgs/healthcare_data"
EXAMPLES = "../input/question_query.json"
MODEL_ID = "PASS"

if __name__ == '__main__':
    device = set_device()
    # model_name = "general_model"  # google/flan-t5-base
    model_name = "google/flan-t5-base"
    tokenizer, model = load_tokenizer_model(model_name, device)
    # examples = "../input/question_query.json"
    examples = "../input/paraphrased_examples.json"
    fine_tune_training(tokenizer, model, examples, device,
                       save_name="domain_model2",
                       save_model=True, push_to_hub=False)
    
    # Load the fine-tuned model
    model_name = "domain_model2"  # Fine-tuned model name
    tokenizer, model = load_tokenizer_model(model_name, device)

    # Input the question and expected SQL query
    input_question = "Which providers have NPIs in the state of California>"
    expected_sql = "SELECT npi FROM nppes WHERE plocstatename = 'CA';"

    # Generate the corresponding SQL query for the input question
    generated_sql_query = generate_sql_query(input_question, model, tokenizer, device)
    logging.info(f"Generated SQL Query: {generated_sql_query}")
    # INFO:root:Generated SQL Query: NPIs_state_name = 'California'

    # Evaluate the generated SQL query against the expected result
    # evaluation_result = evaluate_generated_sql(generated_sql_query, expected_sql)
    # logging.info(f"Evaluation Result: Match: {evaluation_result['match']}, BLEU Score: {evaluation_result['bleu_score']}")
