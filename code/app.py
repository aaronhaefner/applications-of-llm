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
                         tokenized_train_dataset, tokenized_test_dataset)

if __name__ == '__main__':
    paraphrase_sql_questions()
