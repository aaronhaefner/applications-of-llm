# General utils functions for data processing and model training.
import os
import logging
import torch
import json
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM)
from datasets import load_dataset, Dataset
from dotenv import load_dotenv
logging.basicConfig(level=logging.INFO)
load_dotenv()

from utils.globals import (HFTOKEN)

# Configure logging
log_file_path = f"t5_training.log"
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
        AutoTokenizer.from_pretrained(
        model_name,
        token=HFTOKEN,
        legacy=True,
        clean_up_tokenization_spaces=True))
    model = (
        AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        token=HFTOKEN).to(device))
    make_model_contiguous(model)
    return tokenizer, model


# Load and split the dataset
def load_and_split_dataset(source: str,
                           dataset_identifier: str,
                           seed: int = 42,
                           max_train_samples: int = None,
                           max_test_samples: int = None,
                           test_size: float = 0.2) -> tuple:
    """
    Load a dataset from a specified source (Hugging Face or local JSON file) 
    and split into train and test sets.

    Args:
        source (str): The source of the dataset ('huggingface' or 'local').
        dataset_identifier (str): The dataset name to load if 'huggingface', 
                                  or the path to the JSON file if 'local'.
        seed (int): The random seed to use for shuffling.
        max_train_samples (int): The maximum number of samples for training.
        max_test_samples (int): The maximum number of samples for testing.
        test_size (float): Proportion of the dataset in the test split.

    Returns:
        tuple: The train and test datasets.
    """
    if source == 'huggingface':
        dataset = load_dataset(
            dataset_identifier,
            split="train").train_test_split(test_size=test_size, seed=seed)

    elif source == 'local':
        with open(dataset_identifier, 'r') as f:
            data = json.load(f)
        dataset = Dataset.from_list(data).train_test_split(
            test_size=test_size, seed=seed)

    else:
        raise ValueError("Source must be either 'huggingface' or 'local'")

    train_dataset = dataset["train"].shuffle(seed=seed)
    if max_train_samples:
        train_dataset = train_dataset.select(range(max_train_samples))

    test_dataset = dataset["test"].shuffle(seed=seed)
    if max_test_samples:
        test_dataset = test_dataset.select(range(max_test_samples))

    return train_dataset, test_dataset


def preprocess_function(examples_to_encode: dict, tokenizer) -> dict:
    """
    Encode the input and target columns using the tokenizer.

    Args:
        examples_to_encode (dict): The input examples to encode.
        tokenizer: The tokenizer to use.

    Returns:
        dict: The encoded input and target columns.
    """
    inputs = examples_to_encode["instruction_system_prompt"]  # input column
    targets = examples_to_encode["answer"]  # target column

    inputs_encoded = tokenizer(inputs, max_length=512, truncation=True, 
                                   padding="max_length", return_tensors="pt")
    targets_encoded = tokenizer(targets, max_length=512, truncation=True, 
                                    padding="max_length", return_tensors="pt")

    return {
        "input_ids": inputs_encoded.input_ids.squeeze(0),
        "labels": targets_encoded.input_ids.squeeze(0)
    }


def process_tokenizer(tokenizer,
                      train_dataset: dict,
                      test_dataset: dict) -> tuple:
    """
    Tokenize the train and test datasets.

    Args:
        tokenizer: The tokenizer to use.
        train_dataset (dict): The training dataset.
        test_dataset (dict): The testing dataset.

    Returns:
        tuple: The tokenized train and test datasets.
    """
    tokenized_train_dataset = train_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True, remove_columns= train_dataset.column_names)
    tokenized_test_dataset = test_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True, remove_columns=test_dataset.column_names)
    return tokenized_train_dataset, tokenized_test_dataset


# Make model contiguous
def make_model_contiguous(model) -> None:
    """
    Make the model parameters contiguous in memory.

    Args:
        model: The model to make contiguous
    
    Returns: None
    """
    for param in model.parameters():
        param.data = param.data.contiguous()


def set_device() -> torch.device:
    """
    Set the device to use for training based on availability.

    Returns:
        torch.device: The device to use.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Device:", device)
    return device
