import os

HUB_NAME = "aaronhaefner"
REPO_NAME = "t5_flan_txt2sql"
MODELS = {
    "T5": ["google_t5_flan", "google/flan-t5-base"],
    "T5_small": ["google_t5_flan_small", "google/flan-t5-small"],
    "T5_large": ["google_t5_flan_large", "google/flan-t5-large"],
}
MODEL_TYPE, MODEL_NAME = MODELS["T5"]
STRATEGY = "steps"

MODEL_PARMS = {"learning_rate": 1e-5,
               "max_steps": 1000,
               "warmup_steps": 500,
               "per_device_train_batch_size": 4,
               "per_device_eval_batch_size": 4,
               "weight_decay": 0.01,
               "logging_dir": "logs",
               "logging_strategy": "steps",
               "logging_steps": 100,
               "save_strategy": "steps",
               "save_steps": 100,
               "eval_strategy": "steps",
               "eval_steps": 100,
               "num_train_epochs": 2}

HFTOKEN = os.getenv("HUGGINGFACE_TOKEN")
HF_HOME = os.getenv("HF_HOME")
MAX_TRAIN_SAMPLES = 5000
MAX_TEST_SAMPLES = 1000
TEST_SIZE = 0.2
SEED = 42

HUB_MODEL_ID = f"{HUB_NAME}/{REPO_NAME}"
BASE_SQL_DATASET = "philikai/200k-Text2SQL" # https://huggingface.co/datasets/philikai/200k-Text2SQL
HEALTHCARE_DATASET = "Nicolybgs/healthcare_data" # https://huggingface.co/datasets/Nicolybgs/healthcare_data