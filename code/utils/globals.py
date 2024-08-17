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

HFTOKEN = os.getenv("HUGGINGFACE_TOKEN")
HF_HOME = os.getenv("HF_HOME")
MAX_TRAIN_SAMPLES = 5000
MAX_TEST_SAMPLES = 1000
TEST_SIZE = 0.2
SEED = 42

HUB_MODEL_ID = f"{HUB_NAME}/{REPO_NAME}"
DATASET = "philikai/200k-Text2SQL" # https://huggingface.co/datasets/philikai/200k-Text2SQL
HEALTHCARE = "Nicolybgs/healthcare_data" # https://huggingface.co/datasets/Nicolybgs/healthcare_data