import os
import logging
import torch
import json
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          TrainingArguments, Trainer)
from datasets import load_dataset
from dotenv import load_dotenv
from nltk.translate.bleu_score import sentence_bleu
logging.basicConfig(level=logging.INFO)
load_dotenv()

from utils.globals import (
    MODEL_TYPE, MODEL_NAME, STRATEGY, HFTOKEN, SEED,
    MAX_TRAIN_SAMPLES, MAX_TEST_SAMPLES, TEST_SIZE, HUB_MODEL_ID
)

# Configure logging
log_file_path = f"{MODEL_TYPE}_training.log"
logging.basicConfig(
    filename=log_file_path,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def log_generation_results(generated_sql: str, expected_sql: str):
    logging.info(f"Generated SQL: {generated_sql}")
    logging.info(f"Expected SQL: {expected_sql}")
    result = evaluate_generated_sql(generated_sql, expected_sql)
    logging.info(f"Match: {result['match']},BLEU Score: {result['bleu_score']}")


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


# Load and preprocess dataset
def load_train_test(dataset_name: str,
                    seed: int = 42,
                    max_train_samples: int = None,
                    max_test_samples: int = None,
                    test_size: float = 0.2) -> tuple:
    """
    Load dataset from Hugging Face and split into train and test sets.

    Args:
        dataset_name (str): The dataset name to load.
        seed (int): The random seed to use for shuffling.
        max_train_samples (int): The maximum number of samples for training.
        max_test_samples (int): The maximum number of samples for testing.
        test_size (float): Proportion of the dataset in the test split.

    Returns:
        tuple: The train and test datasets.
    """
    data = load_dataset(dataset_name,
                        split="train").train_test_split(test_size=test_size)
    train_dataset = data["train"].shuffle(seed=seed)
    if max_train_samples:
        train_dataset = train_dataset.select(range(max_train_samples))

    test_dataset = data["test"].shuffle(seed=seed)
    if max_test_samples:
        test_dataset = test_dataset.select(range(max_test_samples))

    return train_dataset, test_dataset


def preprocess_function(examples_to_encode: dict,
                        tokenizer, model_type: str = "T5") -> dict:
    """
    Encode the input and target columns using the tokenizer.

    Args:
        examples_to_encode (dict): The input examples to encode.
        tokenizer: The tokenizer to use.
        model_type (str): The model type ("T5", "PEGASUS", etc.).

    Returns:
        dict: The encoded input and target columns.
    """
    inputs = examples_to_encode["question"]  # input column
    targets = examples_to_encode["answer"]  # target column

    if model_type == "T5":
        inputs_encoded = tokenizer(inputs, max_length=512, truncation=True, 
                                   padding="max_length", return_tensors="pt")
        targets_encoded = tokenizer(targets, max_length=512, truncation=True, 
                                    padding="max_length", return_tensors="pt")
    elif model_type == "PEGASUS":
        inputs_encoded = tokenizer(inputs, max_length=512, truncation=True, 
                                   padding="longest", return_tensors="pt")
        targets_encoded = tokenizer(targets, max_length=512, truncation=True, 
                                    padding="longest", return_tensors="pt")
    else:
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
                      test_dataset: dict,
                      model_type: str = "T5") -> tuple:
    """
    Tokenize the train and test datasets.

    Args:
        tokenizer: The tokenizer to use.
        train_dataset (dict): The training dataset.
        test_dataset (dict): The testing dataset.
        model_type (str): The model type ("T5", "PEGASUS", etc.).

    Returns:
        tuple: The tokenized train and test datasets.
    """
    tokenized_train_dataset = train_dataset.map(
        lambda examples: preprocess_function(
            examples, tokenizer, model_type),
            batched=True, remove_columns= train_dataset.column_names)
    tokenized_test_dataset = test_dataset.map(
        lambda examples: preprocess_function(
            examples,tokenizer, model_type),
            batched=True, remove_columns=test_dataset.column_names)
    return tokenized_train_dataset, tokenized_test_dataset


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


def generate_text(prompt: str,
                  model,
                  tokenizer,
                  device, num: int = 1, max_length: int = 100) -> list:
    """
    Generate text based on the given prompt.

    Args:
        prompt (str): The input text prompt.
        model: The model to use.
        tokenizer: The tokenizer associated with the model.
        device: The device on which to perform the generation.
        num (int): Number of generated sequences. Default is 1.
        max_length (int): The maximum length of the generated sequences. Default is 100.

    Returns:
        list: A list of generated texts.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs, max_length=max_length,
        num_return_sequences=num, do_sample=True)

    return [tokenizer.decode(output,
                             skip_special_tokens=True) for output in outputs]


def generate_questions(model, tokenizer, device, num: int = 1) -> list:
    """
    Generate SQL questions.

    Args:
        model: The model to use.
        tokenizer: The tokenizer associated with the model.
        device: The device on which to perform the generation.
        num (int): Number of questions to generate. Default is 1.

    Returns:
        list: A list of generated SQL questions.
    """
    prompt = "Generate a SQL question that \
        could be used in a typical database query."
    return generate_text(prompt, model, tokenizer, device, num)


def generate_sql_query(question: str,
                       model,
                       tokenizer,
                       device: torch.device,
                       schema_path: str = "../input/schema.json") -> str:
    """
    Generate a SQL query given a question and schema.

    Args:
        question (str): The question to generate a SQL query for.
        model: The model to use.
        tokenizer: The tokenizer associated with the model.
        device: The device on which to perform the generation.
        schema_path (str): Path to the JSON schema file.

    Returns:
        str: The generated SQL query.
    """
    prompt = (
        f"Generate a SQL query that answers the following question:\n{question}"
    )
    return generate_text(prompt, model, tokenizer, device)[0]


def generate_multiple_questions(model, schema: dict, num_examples: int = 5) -> list:
    questions = []
    for _ in range(num_examples):
        questions.append(generate_question(model, schema))
    return questions


def generate_sql_query(question: str, model, tokenizer, device) -> str:
    """
    Generate a SQL query given a question.

    Args:
        question (str): The question to generate a SQL query for.
        model: The model to use.
        tokenizer: The tokenizer associated with the model.
        device: The device on which to perform the generation.

    Returns:
        str: The generated SQL query.
    """
    prompt = f"Given the question: {question}\nGenerate the corresponding SQL query."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate the output
    outputs = model.generate(**inputs, max_length=100, num_return_sequences=1, do_sample=True)
    
    # Decode the output to get the generated SQL query
    generated_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_sql


def generate_multiple_sql_queries(model, questions: list) -> list:
    sql_queries = []
    for question in questions:
        sql_queries.append(generate_sql_query(question, model))
    return sql_queries


def evaluate_generated_sql(generated_sql: str, expected_sql: str) -> dict:
    """
    Evaluate the generated SQL query against the expected SQL query.

    Args:
        generated_sql (str): The generated SQL query.
        expected_sql (str): The expected SQL query.
    
    Returns:
        dict: A dictionary containing match status and BLEU score.
    """
    match = generated_sql.strip() == expected_sql.strip()
    bleu_score = compute_bleu_score(generated_sql, expected_sql)
    return {"match": match, "bleu_score": bleu_score}

def compute_bleu_score(generated_sql: str, reference_sql: str) -> float:
    """
    Compute the BLEU score for the 
    generated SQL query against the reference SQL query.

    Args:
        generated_sql (str): The generated SQL query.
        reference_sql (str): The reference SQL query.

    Returns:
        float: The BLEU score.
    """
    reference = [reference_sql.split()]
    candidate = generated_sql.split()
    return sentence_bleu(reference, candidate)

def evaluate_batch(generated_sqls: list, expected_sqls: list) -> list:
    results = []
    for generated_sql, expected_sql in zip(generated_sqls, expected_sqls):
        result = evaluate_generated_sql(generated_sql, expected_sql)
        results.append(result)
        log_generation_results(generated_sql, expected_sql)
    return results