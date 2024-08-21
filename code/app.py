# Setup
import sys
from utils.main import train_model_pipeline

from dotenv import load_dotenv
load_dotenv()


if __name__ == '__main__':
    # Default training preferences
    sql_prefs = {
        'epochs': 1,
        'learning_rate': 2e-5,
        'per_device_train_batch_size': 8,
        'per_device_eval_batch_size': 8,
        'weight_decay': 0.01,
    }

    health_prefs = {
        'epochs': 3,
        'learning_rate': 2e-5,
        'per_device_train_batch_size': 1,
        'per_device_eval_batch_size': 1,
        'weight_decay': 0.01,
    }

    # Determine which model to train based on command-line argument
    if len(sys.argv) > 1:
        func_to_run = sys.argv[1]
    else:
        func_to_run = "sql"

    if func_to_run == "sql":
        train_model_pipeline(model_size="base",
                             dataset_source="huggingface",
                             dataset_identifier="philikai/200k-Text2SQL",
                             model_save_name="flan-t5-base-sql",
                             prefs=sql_prefs,
                             max_train_samples=5000,
                             max_test_samples=1000)
    elif func_to_run == "health":
        train_model_pipeline(model_size="base",
                             dataset_source="local",
                             dataset_identifier="../input/question_answer.json",
                             model_save_name="flan-t5-base-paraphrase",
                             prefs=health_prefs)
    elif func_to_run == "twostep":
        # First step: Train on SQL data
        train_model_pipeline(model_size="base",
                             dataset_source="huggingface",
                             dataset_identifier="philikai/200k-Text2SQL",
                             model_save_name="flan-t5-base-sql",
                             prefs=sql_prefs,
                             max_train_samples=500,
                             max_test_samples=100)

        # Second step: Load the SQL-trained model and train on health data
        train_model_pipeline(model_size="base",
                             dataset_source="local",
                             dataset_identifier="../input/question_answer.json",
                             model_save_name="flan-t5-base-twostep",
                             prefs=health_prefs,
                             load_pretrained_model="flan-t5-base-sql")
    else:
        raise ValueError("Invalid function to run")