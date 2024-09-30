"""Utility functions for generative tools."""
import logging
import torch
import json
from dotenv import load_dotenv
from nltk.translate.bleu_score import sentence_bleu
from utils.utils import set_device, load_tokenizer_model

logging.basicConfig(level=logging.INFO)
load_dotenv()


def generate_text(
    prompt: str,
    model,
    tokenizer,
    device: torch.device,
    num: int = 1,
    max_length: int = 50,
) -> list:
    """Generate text based on the given prompt.

    Args:
    ----
        prompt (str): The input text prompt.
        model: The model to use.
        tokenizer: The tokenizer associated with the model.
        device: The device on which to perform the generation.
        num (int): Number of generated sequences.
        max_length (int): Max length of the generated sequences.

    Returns:
    -------
        list: A list of generated texts.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=num,
        do_sample=True,
    )
    return [
        tokenizer.decode(output, skip_special_tokens=True) for output in outputs
    ]


def generate_questions(model, tokenizer, device, num: int = 1) -> list:
    """Generate SQL questions.

    Args:
    ----
        model: The model to use.
        tokenizer: The tokenizer associated with the model.
        device: The device on which to perform the generation.
        num (int): Number of questions to generate. Default is 1.

    Returns:
    -------
        list: A list of generated SQL questions.
    """
    prompt = "Generate a SQL question that \
        could be used in a typical database query."
    return generate_text(prompt, model, tokenizer, device, num)


def generate_sql_query(
    question: str, model, tokenizer, device: torch.device
) -> str:
    """Generate a SQL query given a question and schema.

    Args:
    ----
        question (str): The question to generate a SQL query for.
        model: The model to use.
        tokenizer: The tokenizer associated with the model.
        device: The device on which to perform the generation.
        schema_path (str): Path to the JSON schema file.

    Returns:
    -------
        str: The generated SQL query.
    """
    prompt = f"Write a SQL query to:\n{question}"
    return generate_text(prompt, model, tokenizer, device)[0]


def paraphrase(
    model, tokenizer, device, text, num_return_sequences=5, num_beams=5
):
    """Generate paraphrases for a given text."""
    inputs = tokenizer(
        text, truncation=True, padding="longest", return_tensors="pt"
    ).to(device)
    outputs = model.generate(
        **inputs,
        max_length=60,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
    )
    return [
        tokenizer.decode(
            output, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        for output in outputs
    ]


def extend_training_data(model_name: str, json_file: str):
    """pass."""
    device = set_device()
    tokenizer, model = load_tokenizer_model(model_name, device)
    with open(json_file, "r") as file:
        data = json.load(file)

    paraphrased_data = []

    for entry in data:
        question = entry["question"]
        answer = entry["answer"]

        paraphrases = paraphrase(
            model, tokenizer, device, question, num_return_sequences=5
        )

        paraphrased_data.append({"question": question, "answer": answer})

        for para in paraphrases:
            paraphrased_data.append({"question": para, "answer": answer})

    with open("../input/paraphrased_question_answer.json", "w") as outfile:
        json.dump(paraphrased_data, outfile, indent=4)

    print(f"Original dataset size: {len(data)}")
    print(f"Paraphrased dataset size: {len(paraphrased_data)}")


def evaluate_generated_sql(generated_sql: str, expected_sql: str) -> dict:
    """Evaluate the generated SQL query against the expected SQL query.

    Args:
    ----
        generated_sql (str): The generated SQL query.
        expected_sql (str): The expected SQL query.

    Returns:
    -------
        dict: A dictionary containing match status and BLEU score.
    """
    match = generated_sql.strip() == expected_sql.strip()
    bleu_score = compute_bleu_score(generated_sql, expected_sql)
    return {"match": match, "bleu_score": bleu_score}


def compute_bleu_score(generated_sql: str, reference_sql: str) -> float:
    """Compute the BLEU score for the generated SQL query.

    Args:
    ----
        generated_sql (str): The generated SQL query.
        reference_sql (str): The reference SQL query.

    Returns:
    -------
        float: The BLEU score.
    """
    reference = [reference_sql.split()]
    candidate = generated_sql.split()
    return sentence_bleu(reference, candidate)


def evaluate_batch(generated_sqls: list, expected_sqls: list) -> list:
    """Evaluate a batch of generated SQL against expected.

    Args:
    ----
        generated_sqls (list): A list of generated SQL queries.
        expected_sqls (list): A list of expected SQL queries.

    Returns:
    -------
        list: A list of evaluation results for each generated SQL query.
    """
    results = []
    for generated_sql, expected_sql in zip(
        generated_sqls, expected_sqls, strict=False
    ):
        result = evaluate_generated_sql(generated_sql, expected_sql)
        results.append(result)
        # log_generation_results(generated_sql, expected_sql)
    return results
