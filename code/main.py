# Setup
import os
import sys
import logging
import json
import torch
from utils.utils import (
    set_device, load_tokenizer_model, make_model_contiguous,
    load_train_test, preprocess_function, process_tokenizer, 
    set_training_args, set_trainer)
from utils.globals import (DATASET, MODEL_NAME, HUB_MODEL_ID, HFTOKEN)
from transformers import (T5Tokenizer,
                          T5ForConditionalGeneration,
                          TrainingArguments,
                          Trainer)
from datasets import Dataset
from dotenv import load_dotenv
load_dotenv()

def first_stage_training(tokenizer: T5Tokenizer,
                         model: T5ForConditionalGeneration,
                         device: torch.device,
                         push_to_hub: bool=False,
                         dataset: str=DATASET) -> None:
    """
    Load the model and tokenizer from the 
    Hugging Face hub and train it on the input dataset.

    Args:
        tokenizer (T5Tokenizer): The tokenizer to use.
        model (T5ForConditionalGeneration): The model to train.
        device (torch.device): The device to use for training.
        push_to_hub (bool): Whether to push the trained model to Hugging Face.
        dataset (str): The dataset to use for training.
    
    Returns: None
    """
    # Load the dataset
    train_dataset, test_dataset = load_train_test(dataset)

    tokenized_train_dataset, tokenized_test_dataset = process_tokenizer(
        tokenizer,
        train_dataset,
        test_dataset)

    logging.info("General dataset loaded and tokenized")

    # Trainer
    training_args = set_training_args()    
    logging.info("Starting first stage training")

    trainer = set_trainer(
        model,
        training_args,
        tokenized_train_dataset,
        tokenized_test_dataset,
        tokenizer
    )

    trainer.train()
    if push_to_hub:
        trainer.push_to_hub()
    
    # Save the model and tokenizer
    model.save_pretrained("general_model")
    tokenizer.save_pretrained("general_model")

    logging.info("First stage training completed and model saved")


def fine_tune_training(tokenizer: T5Tokenizer,
                          model: T5ForConditionalGeneration,
                          examples_file: str,
                          device: torch.device,
                          push_to_hub: bool=False) -> None:
    """
    Load the previously trained model and fine-tune it on domain-specific data.

    Args:
        tokenizer (T5Tokenizer): The tokenizer to use.
        model (T5ForConditionalGeneration): The model to fine-tune.
        examples_file (str): The file containing the domain-specific data.
        device (torch.device): The device to use for training.
        push_to_hub (bool): Whether to push the trained model to Hugging Face.

    Returns: None
    """
    # Load domain-specific data
    with open(examples_file, 'r') as f:
        domain_data = json.load(f)

    train_dataset = Dataset.from_list(domain_data)

    tokenized_train_dataset = train_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )

    logging.info("Domain-specific dataset loaded and tokenized")

    # Define training arguments specifically for the fine-tuning stage
    training_args = TrainingArguments(
        output_dir="output",
        logging_dir=os.path.dirname("fine_tune.log"),
        logging_strategy="steps",
        logging_steps=100,
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=push_to_hub,
        hub_token=HFTOKEN,
        remove_unused_columns=False,
    )
    logging.info("Starting second stage training (fine-tuning)")

    trainer = set_trainer(
        model,
        training_args,
        tokenized_train_dataset,
        None,  # No evaluation dataset provided
        tokenizer
    )

    trainer.train()
    
    # Save the fine-tuned model and tokenizer
    model.save_pretrained("domain_model")
    tokenizer.save_pretrained("domain_model")

    logging.info("Fine-tuning stage completed and model saved")


if __name__ == '__main__':
    push_to_hub = False
    model_name = MODEL_NAME
    device = set_device()

    if len(sys.argv) < 2:
        raise ValueError("Please provide a mode: \
                         first_stage_training or fine_tune_training")
    mode = sys.argv[1]
    if mode == "first_stage_training":
        first_stage_training(model_name, push_to_hub)
    elif mode == "fine_tune_training":
        examples_file = "'../input/question_query.json"
        fine_tune_training(device)
    elif mode == "full":
        first_stage_training(device)
        fine_tune_training(device)
    else:
        raise ValueError("Invalid mode. Please provide either \
                         first_stage_training or fine_tune_training")