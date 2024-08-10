import os
import logging
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

from code.utils.globals import (
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

# Helper functions
def preprocess_function(examples_to_encode: dict):
    inputs = examples_to_encode["question"] # input column
    targets = examples_to_encode["answer"]  # target column

    inputs_encoded = tokenizer(inputs, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
    input_ids = inputs_encoded.input_ids

    targets_encoded = tokenizer(targets, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
    labels = targets_encoded.input_ids

    return {
        "input_ids": input_ids.squeeze(0),
        "labels": labels.squeeze(0)
    }

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
    return device

def set_training_args():
    training_args = TrainingArguments(
        output_dir="output",
        logging_dir=os.path.dirname(log_file_path),
        logging_strategy=STRATEGY,
        logging_steps=100,
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1.5,
        weight_decay=0.01,
        eval_strategy=STRATEGY,
        eval_steps=100,
        save_strategy=STRATEGY,
        push_to_hub=False,
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

def load_tokenizer_model(device):
    tokenizer = T5Tokenizer.from_pretrained(
        MODEL_NAME,
        token=HFTOKEN,
        legacy=True,
        clean_up_tokenization_spaces=True
    )
    model = T5ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        token=HFTOKEN
    ).to(device)
    return tokenizer, model


def load_train_test(
        dataset_name: str,
        seed : int=SEED,
        max_train_samples: int=MAX_TRAIN_SAMPLES,
        max_test_samples: int=MAX_TEST_SAMPLES
    ):
    raw_data = load_dataset(dataset_name, split="train").train_test_split(test_size=TEST_SIZE)
    train_dataset = raw_data["train"].shuffle(seed=seed).select(range(max_train_samples))
    test_dataset = raw_data["test"].shuffle(seed=seed).select(range(max_test_samples))
    return train_dataset, test_dataset

# Main function
if __name__ == '__main__':
    device = set_device()
    tokenizer, model = load_tokenizer_model(device)

    logging.info(f"Model loaded on {device}")
    make_model_contiguous(model)

    # Load dataset, split, tokenize
    train_dataset, test_dataset = load_train_test(DATASET)

    tokenized_train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names)

    tokenized_test_dataset = test_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=test_dataset.column_names)

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