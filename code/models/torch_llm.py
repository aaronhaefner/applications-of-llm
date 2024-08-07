import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Set device to MPS - Metal Performance Shaders for Apple GPUs
def create_pipeline(task: str, model_name: str, device: torch.device) -> pipeline:
    """
    Create a Hugging Face pipeline and move to the desired device.

    Args:
        task: str, the task for which to create the pipeline (e.g., 'text-generation')
        model_name: str, the model to use
        device: torch.device, the device to use for computation

    Returns:
        pipeline object
    """
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGINGFACE_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=HUGGINGFACE_TOKEN).to(device)

    return pipeline(task, model=model, tokenizer=tokenizer, device=device, clean_up_tokenization_spaces=False)

def generate_text(pipe: pipeline, prompt: str) -> str:
    """
    Generate text from a prompt using the pipeline.

    Args:
        pipe: pipeline object
        prompt: str, input text

    Returns:
        str, generated text
    """
    result = pipe(prompt)
    return result[0]['generated_text']

