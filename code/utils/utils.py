import os
import logging
import torch
import json
from transformers import (T5Tokenizer,
                          T5ForConditionalGeneration,
                          TrainingArguments,
                          Trainer)
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

from utils.globals import (
    MODEL_TYPE, MODEL_NAME, STRATEGY, HFTOKEN, SEED,
    DATASET, MAX_TRAIN_SAMPLES, MAX_TEST_SAMPLES, TEST_SIZE, HUB_MODEL_ID
)

# Configure logging
log_file_path = f"{MODEL_TYPE}_training.log"
logging.basicConfig(
    filename=log_file_path,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def load_schema(schema_file: str) -> dict:
    """
    Load the schema from a JSON file.
    
    Args:
        schema_file (str): The schema file path.

    Returns:
        dict: The json sql schema.
    """
    with open(schema_file, 'r') as f:
        schema = json.load(f)
    return schema


def load_tokenizer_model(model_name: str, device: torch.device) -> tuple:
    """
    Load the tokenizer and model from the Hugging Face model hub.

    Args:
        model_name (str): The model name to load.
        device (torch.device): The device to load the model on.

    Returns:
        tuple: The tokenizer and model.
    """
    tokenizer = (
        T5Tokenizer.from_pretrained(
        model_name,
        token=HFTOKEN,
        legacy=True,
        clean_up_tokenization_spaces=True))
    model = (
        T5ForConditionalGeneration.from_pretrained(
        model_name,
        token=HFTOKEN).to(device))
    return tokenizer, model


def load_train_test(
        dataset_name: str,
        seed : int=SEED,
        max_train_samples: int=MAX_TRAIN_SAMPLES,
        max_test_samples: int=MAX_TEST_SAMPLES
    ):
    """
    Load dataset from Hugging Face and split into train and test sets.

    Args:
        dataset_name (str): The dataset name to load.
        seed (int): The random seed to use for shuffling.
        max_train_samples (int): The maximum number of samples for training.
        max_test_samples (int): The maximum number of samples for testing.

    Returns:
        tuple: The train and test datasets.
    """
    data = (
        load_dataset(
        dataset_name,
        split="train").train_test_split(test_size=TEST_SIZE)
    )
    
    train_dataset = (
        data["train"].shuffle(seed=seed).select(range(max_train_samples))
    )
    test_dataset = (
        data["test"].shuffle(seed=seed).select(range(max_test_samples))
    )
    return train_dataset, test_dataset


def preprocess_function(examples_to_encode: dict,
                        tokenizer: T5Tokenizer) -> dict:
    """
    Encode the input and target columns using the tokenizer.

    Args:
        examples_to_encode (dict): The input examples to encode.
        tokenizer (T5Tokenizer): The tokenizer to use.

    Returns:
        dict: The encoded input and target columns.
    """
    inputs = examples_to_encode["question"]  # input column
    targets = examples_to_encode["answer"]  # target column

    inputs_encoded = tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt")
    input_ids = inputs_encoded.input_ids

    targets_encoded = tokenizer(
        targets,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt")
    labels = targets_encoded.input_ids

    return {
        "input_ids": input_ids.squeeze(0),
        "labels": labels.squeeze(0)
    }

def process_tokenizer(tokenizer: T5Tokenizer, train_dataset: dict, test_dataset: dict) -> tuple:
    """
    Tokenize the train and test datasets.

    Args:
        tokenizer (T5Tokenizer): The tokenizer to use.
        train_dataset (dict): The training dataset.
        test_dataset (dict): The testing dataset.

    Returns:
        tuple: The tokenized train and test datasets.
    """
    tokenized_train_dataset = train_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )

    tokenized_test_dataset = test_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=test_dataset.column_names
    )
    return (tokenized_train_dataset, tokenized_test_dataset)

# Make model contiguous
def make_model_contiguous(model):
    for param in model.parameters():
        param.data = param.data.contiguous()


def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Device:", device)
    return device

def set_training_args(push_to_hub: bool=False):
    training_args = TrainingArguments(
        output_dir="output",
        logging_dir=os.path.dirname(log_file_path),
        logging_strategy=STRATEGY,
        logging_steps=100,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy=STRATEGY,
        eval_steps=100,
        save_strategy=STRATEGY,
        push_to_hub=push_to_hub,
        hub_model_id=HUB_MODEL_ID,
        hub_token=HFTOKEN,
        remove_unused_columns=False,
        gradient_accumulation_steps=2,
        fp16=False,  # Consider enabling later
        max_grad_norm=1.0,
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Added for distributed training
        # deepspeed=False,  # Set this to a config file path if using DeepSpeed
    )
    return training_args    


def set_trainer(model, training_args, train_dataset, test_dataset, tokenizer):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )
    return trainer


def generate_question(model: T5ForConditionalGeneration, schema: dict) -> str:
    """
    Generate a SQL question given a schema.

    Args:
        schema (dict): The SQL schema.

    Returns:
        str: The generated SQL question.
    """
    prompt = (
        "Given the following SQL schema: \n"
        f"{json.dumps(schema, indent=2)}\n"
        "Generate a SQL question and its corresponding SQL query."
    )
    # Use your model to generate a response
    generated_question = model.generate_question(prompt)
    return generated_question


def generate_sql_query(question: str, model: T5ForConditionalGeneration) -> str:
    """
    Generate a SQL query given a question.

    Args:
        question (str): The question to generate a SQL query for.
        model (T5ForConditionalGeneration): The model to use.

    Returns:
        str: The generated SQL query.
    """
    prompt = (
        f"Given the question: {question}\n"
        "Generate the corresponding SQL query."
    )
    # Use your model to generate a response
    generated_sql = model.generate_sql_query(prompt)
    return generated_sql


def evaluate_generated_sql(generated_sql: str, expected_sql: str) -> bool:
    """
    Evaluate the generated SQL query against the expected SQL query.

    Args:
        generated_sql (str): The generated SQL query.
        expected_sql (str): The expected SQL query.
    
    Returns:
        bool: True if the generated SQL query matches the expected SQL query,
        else False.
    """
    return generated_sql.strip() == expected_sql.strip()
