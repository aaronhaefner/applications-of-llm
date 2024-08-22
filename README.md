# Applications of LLMs

**Development in progess...**

## Overview

This repository contains tools and functions leveraging pre-trained Large Language Models (LLMs) for various applications in healthcare and finance. The current focus is on generating and understanding SQL queries from natural language, with plans to expand into other domains.

## Current Work

### Text-to-SQL Generation

- **Fine-Tuning**: The primary model in use is [FLAN-T5 Base](https://huggingface.co/google/flan-t5-base), fine-tuned on SQL-related data to enhance its ability to generate SQL queries.
- **Paraphrasing SQL Questions**: This process uses the fine-tuned model to generate paraphrased questions that map to the same SQL query, effectively expanding the training dataset.

### Future Work

- **Domain-Specific SQL Query Generation**: Expand the dataset with generated paraphrases and fine-tune the model on healthcare-specific data.
- **Exploration of Other Applications**: Apply LLMs to other tasks within healthcare and finance.

## Repository Structure

- **code/**: All project code.
  - **utils/**: Utility functions and helpers.
  - **notebooks/**: Jupyter notebooks for experimentation and analysis.

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/aaronhaefner/applications-of-llm.git
```

### 2. Install Dependencies

```bash
make setup
```

### 3. Set Up Environment Variables

Create a `.env` file and add the necessary variables:

```bash
HFTOKEN=<your-hugging-face-token>
HF_HOME=<path-to-hugging-face-cache>
```

### 4. Running the Application

The `app.py` script implements the core functionality, such as SQL paraphrasing. The process involves fine-tuning the T5 model on SQL data and using it to generate paraphrased questions, thereby expanding the dataset.

### 5. Future Development

Use the expanded dataset to further fine-tune the model on healthcare-specific data for SQL query generation.

## Model on Hugging Face

The models are still being developed but metrics are available on [Hugging Face](https://huggingface.co/aaronhaefner/).

## Future Work

- **Tool and Function Development**: Continue developing tools for broader LLM applications.
- **Dataset Integration**: Incorporate additional datasets for enhanced performance across domains.
- **Evaluation and Refinement**: Ongoing evaluation and improvement of LLM applications.

## Contact

For questions or contributions, contact [aaronhaefner@gmail.com](mailto:aaronhaefner@gmail.com).
