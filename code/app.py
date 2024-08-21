# Setup
import sys
from utils.utils import (set_device, process_tokenizer,
                         load_tokenizer_model, load_and_split_dataset)
from utils.main import train_model
from dotenv import load_dotenv
load_dotenv()


def train_sql_model(model_size: str = "base",
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
        'per_device_train_batch_size': 8,
        'per_device_eval_batch_size': 8,
        'weight_decay': 0.01,
    }
    device = set_device()
    model_name = f"google/flan-t5-{model_size}"
    model_type = "T5"
    model_save_name = f"flan-t5-{model_size}-sql"
    dataset_name = "philikai/200k-Text2SQL"

    tokenizer, model = load_tokenizer_model(model_name, device)
    train_dataset, test_dataset = load_and_split_dataset(
        source = "huggingface", dataset_identifier = dataset_name,
        max_train_samples=max_train_samples, max_test_samples=max_test_samples)

    tokenized_train_dataset, tokenized_test_dataset = process_tokenizer(
        tokenizer, train_dataset, test_dataset, model_type=model_type)

    # Train the model on the SQL dataset
    train_model(tokenizer, model, device, prefs,
                tokenized_train_dataset, tokenized_test_dataset,
                model_save_name, save_model=False)


def train_health_model(model_size: str = "base",):
    """
    Train a google T5 Flan model on a SQL question-answer contextual data
    in the healthcare domain for paraphrasing.
    """
    prefs = {
        'epochs': 3,
        'learning_rate': 2e-5,
        'per_device_train_batch_size': 1,
        'per_device_eval_batch_size': 1,
        'weight_decay': 0.01,
    }
    device = set_device()
    model_name = f"google/flan-t5-{model_size}"
    model_type = "T5"
    model_save_name = f"flan-t5-{model_size}-paraphrase"
    dataset_name = "../input/question_answer.json"
    
    tokenizer, model = load_tokenizer_model(model_name, device)
    train_dataset, test_dataset = load_and_split_dataset(
        source = "local", dataset_identifier = dataset_name)

    # Prepare datasets
    tokenized_train_dataset, tokenized_test_dataset = process_tokenizer(
        tokenizer, train_dataset, test_dataset, model_type=model_type)

    # Train the model on the Healthcare domain dataset
    train_model(tokenizer, model, device, prefs,
                         tokenized_train_dataset, tokenized_test_dataset,
                         model_save_name, save_model=False)
    

if __name__ == '__main__':
    # capture argument
    if len(sys.argv) > 1:
        func_to_run = sys.argv[1]
    else:
        func_to_run = "sql"
    
    if func_to_run == "sql":
        train_sql_model()
    elif func_to_run == "health":
        train_health_model()
    else:
        raise ValueError("Invalid function to run")