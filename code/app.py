"""Application entry point."""
import sys
import json
from utils.main import train_model_pipeline
from dotenv import load_dotenv

load_dotenv()


def run_experiment(varying_param: str, varying_values: list):
    """Run the training pipeline with varying hyperparameters."""
    max_train_samples = 50000
    max_test_samples = 10000
    base_model = "flan-t5-base"

    # Fixed parameters
    fixed_prefs = {
        "epochs": 1,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "weight_decay": 0.01,
        "strategy": "steps",
        "gradient_accumulation_steps": 1,
    }

    print(f"Running training pipeline with varying {varying_param}.")
    results = train_model_pipeline(
        dataset_source="huggingface",
        dataset_identifier="philikai/200k-Text2SQL",
        model_save_name=f"{base_model}-sql-v2",
        save_model=False,
        prefs=fixed_prefs,
        varying_param=varying_param,
        varying_values=varying_values,
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
        load_pretrained_model=None,
        push_to_hub=False,
    )

    print(f"Results for varying {varying_param}:")
    for result in results:
        print(
            f"Training time: {result['train_runtime']:.2f} s, "
            f"Value: {result['param_value']}, "
            f"Loss: {result['final_loss']:.5f}"
        )

    # save results as json for later
    with open(f"results_{varying_param}.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Provide an argument: 'train' or 'finetune'")
        sys.exit(1)
    arg = sys.argv[1]

    if arg == "train":
        run_experiment(
            varying_param="learning_rate",
            varying_values=[1e-7, 5e-7, 1e-6, 5e-6],
        )

    elif arg == "finetune":
        run_experiment(
            varying_param="learning_rate", varying_values=[1e-7, 5e-7, 1e-6]
        )
