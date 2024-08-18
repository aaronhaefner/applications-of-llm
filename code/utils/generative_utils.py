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
from nltk.translate.bleu_score import sentence_bleu
logging.basicConfig(level=logging.INFO)
load_dotenv()

from utils.utils import log_generation_results
from utils.globals import (MODEL_TYPE, MODEL_NAME, STRATEGY, HFTOKEN, SEED,
    DATASET, MAX_TRAIN_SAMPLES, MAX_TEST_SAMPLES, TEST_SIZE, HUB_MODEL_ID)


def generate_text(prompt: str,
                  model,
                  tokenizer,
                  device: torch.device,
                  num: int = 1,
                  max_length: int = 100) -> list:
    """
    Generate text based on the given prompt.

    Args:
        prompt (str): The input text prompt.
        model: The model to use.
        tokenizer: The tokenizer associated with the model.
        device: The device on which to perform the generation.
        num (int): Number of generated sequences. Default is 1.
        max_length (int): Max length of the generated sequences. (Default 100)

    Returns:
        list: A list of generated texts.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs,
                             max_length=max_length,
                             num_return_sequences=num,
                             do_sample=True)
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
                       device: torch.device) -> str:
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
        f"Write a SQL query to:\n{question}"
    )
    return generate_text(prompt, model, tokenizer, device)[0]


def generate_multiple_questions(model,
                                schema: dict,
                                num_examples: int = 5) -> list:
    questions = []
    for _ in range(num_examples):
        questions.append(generate_question(model, schema))
    return questions


def generate_sql_query(question: str,
                       model,
                       tokenizer,
                       device: torch.device) -> str:
    """
    Generate a SQL query given a question.

    Args:
        question (str): The question to generate a SQL query for.
        model (T5ForConditionalGeneration): The model to use.
        tokenizer: The tokenizer associated with the model.
        device: The device on which to perform the generation.

    Returns:
        str: The generated SQL query.
    """
    prompt = f"Given the question: {question}\n \
        Generate the corresponding SQL query that answers the question."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate the output
    outputs = model.generate(**inputs,
                             max_length=100,
                             num_return_sequences=1,
                             do_sample=True)
    
    # Decode the output to get the generated SQL query
    generated_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_sql


def generate_multiple_sql_queries(model, questions: list) -> list:
    """
    Generate SQL queries for multiple questions.

    Args:
        model (T5ForConditionalGeneration): The model to use.
        questions (list): A list of questions to generate SQL queries for.

    Returns:
        list: A list of generated SQL queries.
    """
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
    """
    Evaluate a batch of generated SQL queries against the expected SQL queries.

    Args:
        generated_sqls (list): A list of generated SQL queries.
        expected_sqls (list): A list of expected SQL queries.

    Returns:
        list: A list of evaluation results for each generated SQL query.
    """
    results = []
    for generated_sql, expected_sql in zip(generated_sqls, expected_sqls):
        result = evaluate_generated_sql(generated_sql, expected_sql)
        results.append(result)
        log_generation_results(generated_sql, expected_sql)
    return results