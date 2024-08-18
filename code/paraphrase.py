import json
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          Trainer, TrainingArguments)
# Load pre-trained PEGASUS model and tokenizer
# model_name = "tuner007/pegasus_paraphrase"  # Pre-trained for paraphrasing
model_name = "flan-t5-base-paraphrase"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def paraphrase(text, model, tokenizer, num_return_sequences=5, num_beams=5):
    """
    Generate paraphrases for a given text.
    """
    inputs = tokenizer(text, truncation=True, padding='longest', return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=60,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        temperature=1.5,
        do_sample=True  # Enable sampling to use temperature
    )
    return [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output in outputs]

# Load the original dataset
with open("../input/question_query_v2.json", "r") as file:
    data = json.load(file)

# Prepare a list to hold the new paraphrased data
paraphrased_data = []

# Paraphrase each question and add new entries to the dataset
for entry in data:
    question = entry["question"]
    answer = entry["answer"]
    
    # Generate paraphrases for the question
    paraphrases = paraphrase(question, model, tokenizer, num_return_sequences=5)
    
    # Add the original entry
    paraphrased_data.append({
        "question": question,
        "answer": answer
    })
    
    # Add each paraphrased question with the same answer
    for para in paraphrases:
        paraphrased_data.append({
            "question": para,
            "answer": answer
        })

# Save the new dataset to a JSON file
with open("../input/paraphrased_examples_v2.json", "w") as outfile:
    json.dump(paraphrased_data, outfile, indent=4)

print(f"Original dataset size: {len(data)}")
print(f"Paraphrased dataset size: {len(paraphrased_data)}")
