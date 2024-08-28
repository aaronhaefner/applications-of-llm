import sys
import json
import itertools
from utils.utils import set_device, load_tokenizer_model, load_and_split_dataset, process_tokenizer
from utils.main import train_model_pipeline, unpack_prefs, train_model
from utils.generative_utils import (extend_training_data, generate_text, generate_questions)
from dotenv import load_dotenv
load_dotenv()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Provide 'paraphrase', 'train', 'finetune', or 'experiment' as argument.")
        sys.exit(1)
    arg = sys.argv[1]

    max_train_samples = 500
    max_test_samples = 100
    base_model = "flan-t5-base"
    
    # Fixed parameters
    fixed_prefs = {
        'learning_rate': 5e-7,
        'per_device_train_batch_size': 2,
        'per_device_eval_batch_size': 2,
        'weight_decay': 0.01,
        'strategy': "steps",
        'gradient_accumulation_steps': 1,
    }

    # Varying parameter: epochs
    varying_param = 'epochs'
    varying_values = [1, 2, 3]

    # Train on general context data
    if arg == "train":
        print("Running training pipeline.")
        results = train_model_pipeline(
            dataset_source="huggingface",
            dataset_identifier="philikai/200k-Text2SQL",
            model_save_name=f"{base_model}-sql-v2",
            save_model=False, prefs=fixed_prefs,
            varying_param=varying_param, varying_values=varying_values,
            max_train_samples=max_train_samples,
            max_test_samples=max_test_samples,
            load_pretrained_model=None,
            push_to_hub=False)

        for result in results:
            print(f"Varying parameter: {result['changed_param']}, "
                  f"Value: {result['param_value']}, "
                  f"Loss: {result['final_loss']}")
        
        # save results as json for later
        with open("results.json", "w") as f:
            json.dump(results, f)

    # Fine-tune on domain-specific data
    elif arg == "finetune":
        print("Fine-tuning on domain data.")
        train_model_pipeline(
            dataset_source="local",
            dataset_identifier="../input/question_answer.json",
            model_save_name=f"{base_model}-health-finetuned",
            save_model=False, prefs=fixed_prefs,
            varying_param='epochs', varying_values=epochs_values,
            load_pretrained_model=f"flan-t5-base-sql-v2",
            push_to_hub=False)
