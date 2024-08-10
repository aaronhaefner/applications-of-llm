import os

MODELS = {
    "T5": ["google_t5", "google/t5-v1_1-small"],
}
MODEL_TYPE, MODEL_NAME = MODELS["T5"]
STRATEGY = "steps"

HFTOKEN = os.getenv("HUGGINGFACE_TOKEN")
MAX_TRAIN_SAMPLES = 5000
MAX_TEST_SAMPLES = 1000
TEST_SIZE = 0.2
SEED = 42

HUB_MODEL_ID = "aaronhaefner/txt2sql_v1"
DATASET = "philikai/200k-Text2SQL"
