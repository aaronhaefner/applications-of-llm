# Text-to-sql language model with domain-specific fine-tuning layer
import sys
import json
from utils.utils import set_device, load_tokenizer_model
from utils.main import train_model_pipeline
from utils.generative_utils import (extend_training_data, generate_text, generate_questions)
from dotenv import load_dotenv
load_dotenv()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Provide 'paraphrase' or 'train' as argument.")
        sys.exit(1)
    arg = sys.argv[1]

    # Set training preferences
    max_train_samples = None
    max_test_samples = None
    base_model = "flan-t5-base"
    
    sql_prefs = {
        'epochs': 2,
        'learning_rate': 5e-7,
        'per_device_train_batch_size': 2,
        'per_device_eval_batch_size': 2,
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

    # Train on general context data
    if arg == "train":
        print("Running training pipeline.")
        # write function to spit out specs
        train_model_pipeline(
            dataset_source="huggingface",
            # dataset_identifier="philikai/200k-Text2SQL",
            dataset_identifier="philikai/SPIDER_SQL_synth_data_w_Claude3_Haiku",
            model_save_name=f"{base_model}-sql-v2",
            save_model=True, prefs=sql_prefs,
            max_train_samples=max_train_samples,
            max_test_samples=max_test_samples,
            load_pretrained_model=None,
            push_to_hub=False)

    # Fine-tune on domain-specific data
    elif arg == "finetune":
        print("Fine-tuning on domain data.")
        train_model_pipeline(
            dataset_source="local",
            dataset_identifier="../input/question_answer.json",
            model_save_name=f"{base_model}-health-finetuned",
            save_model=True, prefs=health_prefs,
            load_pretrained_model=f"flan-t5-base-sql-v2",
            push_to_hub=False)

    # Generative text applications
    # Paraphrasing questions to extend question-answer training data
    elif arg == "paraphrase":
        extend_training_data(
            model_name="flan-t5-health-finetuned",
            json_file="../input/question_answer.json")
    
    # Generate SQL queries from questions
    elif arg == "generate":
        device = set_device()
        # tokenizer, model = load_tokenizer_model(f"flan-t5-{size}-sql-v2", device)
        tokenizer, model = load_tokenizer_model("flan-t5-health-finetuned", device)
        text = "Write a question that would generate the following..."
        #print(generate_text(text, model, tokenizer, device))