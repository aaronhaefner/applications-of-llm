# SQL-Trained LLM

## Overview

This repository contains exploratory work on training large language models (LLMs) using the Hugging Face library, focusing on text-to-text models.

## Model

The model used for this exploration is the [Google T5 Small](https://huggingface.co/google-t5/t5-small).
T5 (Text-To-Text Transfer Transformer) is designed to handle to NLP tasks by treating them as text-to-text problems.

## Training Details

- **Training Dataset**: SQL text-to-text data, including SQL queries and their corresponding text descriptions, to help the model learn to generate and understand SQL queries from natural language.

## Repository Structure

- **code/**: Contains all the code related to training and evaluation.
  - **utils/**: Utility functions and helpers.
  - **notebooks/**: Jupyter notebooks for experimentation and analysis.

## Getting Started

To replicate the training process or use the model, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/aaronhaefner/sql-trained-llm.git
   ```

2. **Install Dependencies**:
   ```bash
   make setup
   ```

3. In progress...

## Model on Hugging Face

You can find the trained model on Hugging Face [here](https://huggingface.co/aaronhaefner/sql-trained-llm).

## Notes

- **Future Work**: This project is ongoing, with plans for further improvements and experiments, including fine-tuning on additional datasets and evaluating on different benchmarks.

## Contact

For questions or contributions, please contact me at [aaronhaefner@gmail.com](mailto:aaronhaefner@gmail.com).
