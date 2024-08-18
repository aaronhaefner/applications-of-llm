# Applications of LLMs

## Overview

This repository is a collection of tools and functions that leverage pre-trained Large Language Models (LLMs) for various applications in healthcare and finance. The project is currently in its early stages, focusing on building a versatile suite of utilities that can be applied across different domains. The initial focus is on generating and understanding SQL queries from natural language, with plans to expand into other applications.

## Current Work

### Text-to-SQL Generation

The primary work in this repository revolves around fine-tuning and applying pre-trained LLMs to the task of generating SQL queries from natural language questions. This involves:

- **Fine-tuning** the [Google FLAN-T5 Base](https://huggingface.co/google/flan-t5-base) model, a versatile text-to-text model, on SQL-related data to improve its ability to understand and generate SQL queries.
- **Paraphrasing SQL Questions**: Using the fine-tuned model to generate paraphrased questions that map to the same SQL query. This approach effectively increases the size of the training dataset, enhancing the model’s performance without the need for manually creating unique examples for each question.

### Future Work

- **Domain-Specific SQL Query Generation**: The next step is to expand the training dataset using the generated paraphrases and fine-tune the model on healthcare-specific datasets to generate SQL queries relevant to that domain.
- **Exploration of Other Applications**: Beyond SQL, there are plans to apply LLMs to other tasks within healthcare and finance, leveraging the flexibility and power of models like FLAN-T5.

## Model

The primary model used in this exploration is the [Google FLAN-T5 Base](https://huggingface.co/google/flan-t5-base). T5 (Text-To-Text Transfer Transformer) is designed to handle a wide range of NLP tasks by framing them as text-to-text problems. This model has been fine-tuned on SQL-related data to improve its performance in generating SQL queries from natural language inputs.

## Repository Structure

- **code/**: Contains all code related to the development and application of the tools.
  - **utils/**: Utility functions and helpers used throughout the project.
  - **notebooks/**: Jupyter notebooks for experimentation, analysis, and fine-tuning.

## Getting Started

To replicate the experiments or use the tools provided in this repository, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/aaronhaefner/applications-of-llm.git
```

### 2. Install Dependencies

```bash
make setup
```

### 3. Set Up Environment Variables

Create a `.env` file in the root directory of the project and add the necessary environment variables. For example:

```bash
HFTOKEN=<your-hugging-face-token>
HF_HOME=<path-to-hugging-face-cache>
```

### 4. Running the Application

The main application script (`app.py`) implements the core functionality, such as the SQL paraphrasing process. In this workflow, pre-labeled SQL data is used to fine-tune the T5 base model. This fine-tuned model is then used to generate paraphrased questions that map to the same SQL queries, effectively expanding the training dataset without needing to create unique examples manually.

### 5. Future Development

The expanded training dataset will be used to further fine-tune the model on healthcare-specific data, enabling it to generate SQL queries tailored to the healthcare domain.

## Model on Hugging Face

You can find the fine-tuned model, ready for generating SQL queries from natural language, on Hugging Face [here](https://huggingface.co/aaronhaefner/txt2sql_v1).

## Future Work

This project is ongoing, with several planned developments:

- **Tool and Function Development**: Continued development of tools and functions to extend the applicability of LLMs in healthcare, finance, and beyond.
- **Dataset Integration**: Integration of additional datasets to enhance the model's capability across different domains.
- **Evaluation and Refinement**: Continuous evaluation of the LLM applications and refinement based on performance metrics.

## Contact

For questions, suggestions, or contributions, feel free to reach out to me at [aaronhaefner@gmail.com](mailto:aaronhaefner@gmail.com).
