import json
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi, HfFolder


def json_to_hf_dataset(json_file: str) -> DatasetDict:
    """
    Convert a JSON file into a Hugging Face DatasetDict.
    Handles additional columns in the JSON structure if they are added later.

    Args:
        json_file (str): Path to the JSON file.

    Returns:
        DatasetDict: The Hugging Face DatasetDict.
    """
    with open(json_file, "r") as file:
        data = json.load(file)
    
    # Extract the keys from the first entry to determine the structure
    keys = data[0].keys()

    # Prepare the data for Dataset
    formatted_data = {key: [entry.get(key, None) for entry in data] for key in keys}

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_dict(formatted_data)

    # Create a DatasetDict to handle potential train/test splits later
    dataset_dict = DatasetDict({"train": dataset})

    return dataset_dict


def save_hf_dataset(dataset_dict: DatasetDict, save_path: str):
    """
    Save the Hugging Face DatasetDict to the specified path.

    Args:
        dataset_dict (DatasetDict): The DatasetDict to save.
        save_path (str): The path where the dataset will be saved.
    """
    dataset_dict.save_to_disk(save_path)


def push_dataset_to_hub(dataset_dict: DatasetDict, repo_name: str, private: bool = True):
    """
    Push the Hugging Face DatasetDict to the Hugging Face Hub.

    Args:
        dataset_dict (DatasetDict): The DatasetDict to push.
        repo_name (str): The name of the repository on the Hugging Face Hub.
        private (bool): Whether the repository should be private. Default is True.
    """
    # Initialize the HfApi
    api = HfApi()
    
    # Push the dataset to the hub
    dataset_dict.push_to_hub(repo_name, private=private)
    print(f"Dataset pushed to the Hugging Face Hub under the repo: {repo_name}")


if __name__ == "__main__":
    json_file = "../input/question_answer.json"
    hf_dataset = json_to_hf_dataset(json_file)
    
    # Optionally, save the dataset
    save_path = "../output/hf_dataset"
    save_hf_dataset(hf_dataset, save_path)
    
    # Static parameters for pushing to Hugging Face Hub
    PUSH_TO_HUB = True  # Set to False if you don't want to push to the Hub
    REPO_NAME = "my-hf-dataset-repo"  # Set your Hugging Face repository name here
    PRIVATE_REPO = True  # Set to False if you want the repo to be public

    if PUSH_TO_HUB:
        push_dataset_to_hub(hf_dataset, REPO_NAME, PRIVATE_REPO)

    print("Hugging Face Dataset created and saved successfully.")
