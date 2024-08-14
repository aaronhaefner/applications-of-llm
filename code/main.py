# Setup
import sys
import logging
from utils.utils import (
    set_device, load_tokenizer_model, make_model_contiguous,
    load_train_test, preprocess_function, set_training_args,
    set_trainer)
from utils.globals import (DATASET, MODEL_NAME, HUB_MODEL_ID, HFTOKEN)
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments


def train():
    device = set_device()
    tokenizer, model = load_tokenizer_model(device)

    logging.info(f"Model loaded on {device}")
    make_model_contiguous(model)

    # Load dataset, split, tokenize
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

    logging.info("Dataset loaded and tokenized")

    # Trainer
    training_args = set_training_args()    
    logging.info("Starting training")

    trainer = set_trainer(
        model,
        training_args,
        tokenized_train_dataset,
        tokenized_test_dataset,
        tokenizer
    )

    trainer.train()
    # trainer.push_to_hub()

    logging.info("Training completed and model pushed to Hugging Face Hub")


def eval():
    from datasets import load_dataset
    device = set_device()

    # Load the tokenizer and model from Hugging Face Hub
    tokenizer = T5Tokenizer.from_pretrained(HUB_MODEL_ID, token=HFTOKEN)
    model = T5ForConditionalGeneration.from_pretrained(HUB_MODEL_ID, token=HFTOKEN).to(device)

    logging.info(f"Model loaded for evaluation on {device}")
    make_model_contiguous(model)

    # Load the new dataset
    ds = load_dataset("motherduckdb/duckdb-text2sql-25k")

    # Rename the columns to match the expected input by the preprocess_function
    ds = ds.rename_column("prompt", "question")
    ds = ds.rename_column("query", "answer")

    # Tokenize the test dataset
    tokenized_test_dataset = ds["train"].map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=ds["train"].column_names
    )

    logging.info("Test dataset loaded and tokenized")

    # Define evaluation arguments
    eval_args = TrainingArguments(
        output_dir="eval_output",
        per_device_eval_batch_size=8,
        logging_dir="logs",
        logging_strategy="steps",
        logging_steps=100,
        report_to="none"
    )

    # Create a trainer for evaluation
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=tokenized_test_dataset,
        tokenizer=tokenizer
    )

    # Evaluate the model
    metrics = trainer.evaluate()
    logging.info(f"Evaluation metrics: {metrics}")

    print(f"Evaluation metrics: {metrics}")



if __name__ == '__main__':
    # Capture arg from command line to set train/eval mode(s)
    if len(sys.argv) < 2:
        raise ValueError("Please provide a mode: train or eval")
    mode = sys.argv[1]
    if mode == "train":
        train()
    elif mode == "eval":
        eval()
    else:
        raise ValueError("Mode must be either train or eval")
