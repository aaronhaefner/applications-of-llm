import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Set device to MPS - Metal Performance Shaders for Apple GPUs
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

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

    return pipeline(task, model=model, tokenizer=tokenizer, device=device)

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

if __name__ == "__main__":
    # Source: https://huggingface.co/EleutherAI/gpt-neo-1.3B
    model_name = "EleutherAI/gpt-neo-1.3B"  # Medium-sized model for text generation
    task = "text-generation"

    pipe = create_pipeline(task, model_name, device)
    
    prompt = "The future of AI is"
    print(generate_text(pipe, prompt))