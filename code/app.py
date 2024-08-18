# Setup
import os
import sys
import logging
import json
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          Trainer, TrainingArguments)
from datasets import Dataset
from utils.globals import HFTOKEN
from utils.utils import (set_device, process_tokenizer,
                         load_tokenizer_model, load_train_test)
from utils.generative_utils import (generate_sql_query, generate_questions,
                                    generate_multiple_questions,
                                    evaluate_generated_sql)
from main import first_stage_training, fine_tune_training
from dotenv import load_dotenv
load_dotenv()

HEALTH_DATASET = "Nicolybgs/healthcare_data"
EXAMPLES = "../input/question_query.json"
MODEL_ID = "PASS"

def paraphrase_sql_questions():
    device = set_device()
    model_name = "google/flan-t5-base"
    model_type = "T5"
    model_save_name = "flan-t5-base-paraphrase"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    dataset_name = "philikai/200k-Text2SQL"

    # Load and process the datasets
    train_dataset, test_dataset = load_train_test(dataset_name)
    tokenized_train_dataset, tokenized_test_dataset = process_tokenizer(
        tokenizer, 
        train_dataset, 
        test_dataset, 
        model_type=model_type
    )

    # Train the model using the tokenized datasets
    first_stage_training(tokenizer, model, device, 
                         tokenized_train_dataset, tokenized_test_dataset,
                         model_save_name)

# Implement this once paraphrasing works for extending training data
def text_to_sql():
    device = set_device()
    model_name = "google/flan-t5-base"
    model_type = "T5"
    model_save_name = "flan-t5-base-text2sql"
    pass
 
    # # Load the fine-tuned model
    # model_name = "domain_model2"  # Fine-tuned model name
    # tokenizer, model = load_tokenizer_model(model_name, device)

    # # Input the question and expected SQL query
    # input_question = "count the total number of npis in the nppes table in the state of California"
    # expected_sql = "SELECT npi FROM nppes WHERE plocstatename = 'CA';"

    # # Generate the corresponding SQL query for the input question
    # generated_sql_query = generate_sql_query(input_question, model, tokenizer, device)
    # logging.info(f"Generated SQL Query: {generated_sql_query}")
    # INFO:root:Generated SQL Query: NPIs_state_name = 'California'd

if __name__ == '__main__':
    paraphrase_sql_questions()
