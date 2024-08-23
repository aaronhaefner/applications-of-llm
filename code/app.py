# Text-to-sql language model with domain-specific fine-tuning layer
import sys
import json
from utils.utils import set_device, load_tokenizer_model
from utils.main import train_model_pipeline
from utils.generative_utils import extend_training_data
from dotenv import load_dotenv
load_dotenv()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Provide 'paraphrase' or 'train' as argument.")
        sys.exit(1)
    arg = sys.argv[1]

    # Set training preferences
    max_train_samples = 50000
    max_test_samples = 10000
    scaling_factor = (max_train_samples // 50)
    size = "base"
    
    sql_prefs = {
        'epochs': 1,
        'learning_rate': 1e-6,
        'per_device_train_batch_size': 8,
        'per_device_eval_batch_size': 8,
        'weight_decay': 0.01,
        'strategy': "steps",
        'gradient_accumulation_steps': 1,
    }

    health_prefs = {
        'epochs': 3,
        'learning_rate': 1e-6,
        'per_device_train_batch_size': 1,
        'per_device_eval_batch_size': 1,
        'weight_decay': 0.01,
        'strategy': "steps",
        'gradient_accumulation_steps': 1,
    }

    if arg == "none":
        train_model_pipeline(
            model_size=size, dataset_source="huggingface",
            dataset_identifier="philikai/200k-Text2SQL",
            model_save_name=f"flan-t5-{size}-test",
            save_model=False, prefs=sql_prefs,
            max_train_samples=(max_train_samples // scaling_factor),
            max_test_samples=(max_test_samples // scaling_factor),
            load_pretrained_model=None,
            push_to_hub=False)
        sys.exit(0)

    elif arg == "train":
        # First step: Train on SQL data
        train_model_pipeline(
            model_size=size, dataset_source="huggingface",
            dataset_identifier="philikai/200k-Text2SQL",
            model_save_name=f"flan-t5-{size}-sql",
            save_model=True, prefs=sql_prefs,
            max_train_samples=max_train_samples,
            max_test_samples=max_test_samples,
            load_pretrained_model=None,
            push_to_hub=True)

        # Second step: Load the SQL-trained model and train on health data
        train_model_pipeline(
            model_size=size, dataset_source="local",
            dataset_identifier="../input/question_answer.json",
            model_save_name="flan-t5-health-finetuned",
            save_model=False, prefs=health_prefs,
            load_pretrained_model=f"flan-t5-{size}-sql",
            push_to_hub=False)
    # Use the model to generate paraphrases
    elif arg == "paraphrase":
        extend_training_data(
            model_name="flan-t5-health-finetuned",
            json_file="../input/question_answer.json")

