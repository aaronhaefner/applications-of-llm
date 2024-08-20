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
from utils.main import train_model
from dotenv import load_dotenv
load_dotenv()

HEALTH_DATASET = "Nicolybgs/healthcare_data"
EXAMPLES = "../input/question_query.json"
MODEL_ID = "PASS"

# train_dataset = Dataset.from_list(domain_data)

def train_flan_t5(model_size: str = "small",
                  max_train_samples: int = 5000,
                  max_test_samples: int = 1000) -> None:
    """
    Train a Flan T5 model on a SQL question-answer dataset.

    Args:
        None

    Returns: None
    """

    # Set prefs for training
    prefs = {
        'epochs': 1,
        'learning_rate': 2e-5,
        'per_device_train_batch_size': 4,
        'per_device_eval_batch_size': 4,
        'weight_decay': 0.01,
    }
    device = set_device()
    model_name = f"google/flan-t5-{model_size}"
    model_type = "T5"
    model_save_name = f"flan-t5-{model_size}-text2sql"
    dataset_name = "philikai/200k-Text2SQL"

    tokenizer, model = load_tokenizer_model(model_name, device)
    train_dataset, test_dataset = load_train_test(dataset_name,
        max_train_samples=max_train_samples, max_test_samples=max_test_samples)

    tokenized_train_dataset, tokenized_test_dataset = process_tokenizer(
        tokenizer, train_dataset, test_dataset, model_type=model_type)

    # Train the model using the tokenized datasets
    train_model(tokenizer, model, device, prefs,
                tokenized_train_dataset, tokenized_test_dataset)


def paraphrase_sql_questions():
    device = set_device()
    model_name = "google/flan-t5-base"
    model_type = "T5"
    model_save_name = "flan-t5-base-paraphrase"
    # dataset_name = "philikai/200k-Text2SQL"

    tokenizer, model = load_tokenizer_model(model_name, device)
    train_dataset, test_dataset = load_train_test(dataset_name)

    # Prepare datasets
    tokenized_train_dataset, tokenized_test_dataset = process_tokenizer(
        tokenizer, 
        train_dataset, 
        test_dataset, 
        model_type=model_type
    )

    # Train the model using the tokenized datasets
    train_model(tokenizer, model, device, 
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
    # capture argument
    if len(sys.argv) > 1:
        func_to_run = sys.argv[1]
    else:
        func_to_run = "flan"
    
    if func_to_run == "flan":
        train_flan_t5()
    elif func_to_run == "paraphrase":
        paraphrase_sql_questions()
    elif func_to_run == "text2sql":
        text_to_sql()
    else:
        raise ValueError("Invalid function to run")