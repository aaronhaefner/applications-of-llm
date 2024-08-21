# Text-to-sql language model with domain-specific fine-tuning layer
import sys
import json
from utils.utils import set_device, load_tokenizer_model
from utils.main import train_model_pipeline
from dotenv import load_dotenv
load_dotenv()


def paraphrase(model, tokenizer, device,
               text, num_return_sequences=5, num_beams=5):
    """
    Generate paraphrases for a given text.
    """
    inputs = tokenizer(text, truncation=True, padding='longest',
                       return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_length=60,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
    )
    return [tokenizer.decode(
        output, skip_special_tokens=True, clean_up_tokenization_spaces=True
        ) for output in outputs]


def extend_training_data(model_name: str, json_file: str):
    """
    """
    device = set_device()
    tokenizer, model = load_tokenizer_model(model_name, device)
    with open(json_file, "r") as file:
        data = json.load(file)
    
    paraphrased_data = []

    for entry in data:
        question = entry["question"]
        answer = entry["answer"]
        
        paraphrases = paraphrase(
            model, tokenizer, device, question, num_return_sequences=5)
        
        paraphrased_data.append({
            "question": question,
            "answer": answer
        })
        
        for para in paraphrases:
            paraphrased_data.append({
                "question": para,
                "answer": answer
            })
    
    with open("../input/paraphrased_question_answer.json", "w") as outfile:
        json.dump(paraphrased_data, outfile, indent=4)
    
    print(f"Original dataset size: {len(data)}")
    print(f"Paraphrased dataset size: {len(paraphrased_data)}")

    

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Provide 'paraphrase' or 'train' as argument.")
        sys.exit(1)
    arg = sys.argv[1]

    # Set training preferences
    max_train_samples = 5000
    max_test_samples = 1000
    scaling_factor = 10
    SAVE_MODEL = False
    
    sql_prefs = {
        'epochs': 2,
        'learning_rate': 1e-6,
        'per_device_train_batch_size': 8,
        'per_device_eval_batch_size': 8,
        'weight_decay': 0.01,
    }

    health_prefs = {
        'epochs': 3,
        'learning_rate': 1e-6,
        'per_device_train_batch_size': 1,
        'per_device_eval_batch_size': 1,
        'weight_decay': 0.01,
    }
    if SAVE_MODEL:
        print("WARNING: SAVE_MODEL is set to True. This will save the model.")
        confirm = input("Do you want to continue? (y/n): ")
        if confirm.lower() != "y":
            sys.exit(0)
    if arg == "none":
        train_model_pipeline(
            model_size="base", dataset_source="huggingface",
            dataset_identifier="philikai/200k-Text2SQL",
            model_save_name="flan-t5-base-sql",
            save_model=False, prefs=sql_prefs,
            max_train_samples=(max_train_samples // scaling_factor),
            max_test_samples=(max_test_samples // scaling_factor),
            load_pretrained_model=None)
        sys.exit(0)

    elif arg == "train":
        # First step: Train on SQL data
        train_model_pipeline(
            model_size="base", dataset_source="huggingface",
            dataset_identifier="philikai/200k-Text2SQL",
            model_save_name="flan-t5-base-sql",
            save_model=SAVE_MODEL, prefs=sql_prefs,
            max_train_samples=max_train_samples,
            max_test_samples=max_test_samples,
            load_pretrained_model=None)

        # Second step: Load the SQL-trained model and train on health data
        train_model_pipeline(
            model_size="base", dataset_source="local",
            dataset_identifier="../input/question_answer.json",
            model_save_name="flan-t5-health-finetuned",
            save_model=SAVE_MODEL, prefs=health_prefs,
            load_pretrained_model="flan-t5-base-sql")
    # Use the model to generate paraphrases
    elif arg == "paraphrase":
        extend_training_data(
            model_name="flan-t5-health-finetuned",
            json_file="../input/question_answer.json")

