# Setup
import os
import sys
import logging
import json
from utils.utils import (
    set_device, load_tokenizer_model, make_model_contiguous,
    load_train_test, preprocess_function, set_training_args,
    set_trainer)
from utils.globals import (DATASET, MODEL_NAME, HUB_MODEL_ID, HFTOKEN)
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset

def first_stage_training():
    device = set_device()
    tokenizer, model = load_tokenizer_model(device)

    logging.info(f"Model loaded on {device}")
    make_model_contiguous(model)

    # Load the dataset
    train_dataset, test_dataset = load_train_test(DATASET)

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
    
    # Save the model and tokenizer
    model.save_pretrained("general_model")
    tokenizer.save_pretrained("general_model")

    logging.info("First stage training completed and model saved")


def second_stage_training():
    device = set_device()

    # Load the previously trained model
    tokenizer = T5Tokenizer.from_pretrained("general_model")
    model = T5ForConditionalGeneration.from_pretrained("general_model").to(device)

    logging.info(f"Model loaded for fine-tuning on {device}")
    make_model_contiguous(model)

    # Load healthcare-specific dataset
    with open('../input/combined_questions.json', 'r') as f:
        healthcare_data = json.load(f)

    train_dataset = Dataset.from_list(healthcare_data)

    tokenized_train_dataset = train_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )

    logging.info("Healthcare-specific dataset loaded and tokenized")

    # Define training arguments specifically for the fine-tuning stage
    training_args = TrainingArguments(
        output_dir="output",
        logging_dir=os.path.dirname("fine_tune.log"),
        logging_strategy="steps",
        logging_steps=100,
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        num_train_epochs=1,
        weight_decay=0.01,
        push_to_hub=False,
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
    model.save_pretrained("healthcare_model")
    tokenizer.save_pretrained("healthcare_model")

    logging.info("Second stage training (fine-tuning) completed and model saved")


if __name__ == '__main__':
    # Capture arg from command line to set train/eval/sql_eval mode(s)
    if len(sys.argv) < 2:
        raise ValueError("Please provide a mode: first_stage_training or second_stage_training")
    mode = sys.argv[1]
    if mode == "first_stage_training":
        first_stage_training()
    elif mode == "second_stage_training":
        second_stage_training()
    else:
        raise ValueError("Mode must be either first_stage_training or second_stage_training")
