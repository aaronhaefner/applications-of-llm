import os

MODELS = {
    "T5": ["google_t5", "google/flan-t5-base"],
}
MODEL_TYPE, MODEL_NAME = MODELS["T5"]
STRATEGY = "steps"

HFTOKEN = os.getenv("HUGGINGFACE_TOKEN")
MAX_TRAIN_SAMPLES = 50000
MAX_TEST_SAMPLES = 10000
TEST_SIZE = 0.2
SEED = 42

HUB_MODEL_ID = "aaronhaefner/t5_flan_txt2sql"
DATASET = "philikai/200k-Text2SQL"
