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


def unpack_prefs(prefs: dict) -> dict:
    """
    Unpack the preferences dictionary into individual variables.

    Args:
        prefs (dict): The preferences dictionary.

    Returns: dict
    """
    epochs = prefs['epochs']
    learning_rate = prefs['learning_rate']
    per_device_train_batch_size = prefs['per_device_train_batch_size']
    per_device_eval_batch_size = prefs.get('per_device_eval_batch_size', None)
    weight_decay = prefs['weight_decay']

    return (epochs, learning_rate, per_device_train_batch_size,
            per_device_eval_batch_size, weight_decay)


def train_model(tokenizer, model, device: torch.device, 
                prefs: dict, train_dataset, eval_dataset=None,
                save_name: str=None, save_model: bool=False,
                push_to_hub: bool=False, fine_tune: bool=False) -> None:
    """
    Train or fine-tune the model on the given dataset.

    Args:
        tokenizer: The tokenizer to use.
        model: The model to train or fine-tune.
        device: The device to use for training.
        prefs (dict): The preferences for training.
        train_dataset: The tokenized training dataset.
        eval_dataset: The tokenized evaluation dataset (optional).
        save_name (str): The name to save the model as.
        save_model (bool): Whether to save the model.
        push_to_hub (bool): Whether to push the trained model to Hugging Face.
        fine_tune (bool): Whether this is a fine-tuning stage.

    Returns: None
    """
    epochs, learning_rate, per_device_train_batch_size, \
    per_device_eval_batch_size, weight_decay = unpack_prefs(prefs)

    # Set output directories based on training stage
    output_dir = "./results_fine_tuning" if fine_tune else "./results"
    logging_dir = "./logs_fine_tuning" if fine_tune else "./logs"

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        logging_strategy="steps",
        logging_steps=100,
        eval_strategy="steps" if eval_dataset else "no",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        push_to_hub=push_to_hub
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )

    trainer.train()

    if push_to_hub:
        trainer.push_to_hub()

    # Save the model and tokenizer
    if save_model:
        model.save_pretrained(save_name)
        tokenizer.save_pretrained(save_name)
        logging.info(f"{'Fine-tuning' if fine_tune else 'Training'} completed and model saved as {save_name}")
    else:
        logging.info(f"{'Fine-tuning' if fine_tune else 'Training'} completed, model not saved")

