import logging
import torch
from transformers import (TrainingArguments, Trainer)
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
    dataset_name = prefs['dataset_name']

    return (epochs, learning_rate, per_device_train_batch_size,
            per_device_eval_batch_size, weight_decay, dataset_name)


def train_model(tokenizer, model, device: torch.device, 
                prefs: dict, train_dataset, eval_dataset=None,
                save_name: str=None, save_model: bool=False,
                push_to_hub: bool=False, fine_tune: bool=False) -> None:
    """
    Train or fine-tune the model on the given dataset.

    Args:
        tokenizer: The tokenizer for the model.
        model: The model to train.
        device (torch.device): The device to train the model on.
        prefs (dict): The preferences dictionary.
        train_dataset: The training dataset.
        eval_dataset: The evaluation dataset.
        save_name (str): The name to save the model as.
        save_model (bool): Whether to save the model.
        push_to_hub (bool): Whether to push the model to the Hugging Face Hub.
        fine_tune (bool): Whether to fine-tune the model.
    
    Returns: None
    """
    epochs, learning_rate, per_device_train_batch_size, \
        per_device_eval_batch_size, weight_decay, _ = unpack_prefs(prefs)

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
        push_to_hub=push_to_hub,
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
