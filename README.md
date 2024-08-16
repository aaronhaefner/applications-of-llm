# Applications of LLMs

## Overview

This repository serves as a collection of tools and functions that leverage pre-trained large language models (LLMs) in various applications within healthcare and finance. The repository is in its early stages, with a primary focus on building a suite of utilities that can be applied across different domains.

## Current Work

Currently, the repository includes exploratory work on fine-tuning and applying pre-trained LLMs using the Hugging Face library. The initial focus has been on text-to-text models, particularly for generating and understanding SQL queries from natural language, with plans to expand to other applications.

## Model

The model used in the initial exploration is the [Google T5 Small](https://huggingface.co/google/t5-small). T5 (Text-To-Text Transfer Transformer) is designed to handle a wide range of NLP tasks by framing them as text-to-text problems.

## Repository Structure

- **code/**: Contains all the code related to the development and application of the tools.
  - **utils/**: Utility functions and helpers.
  - **notebooks/**: Jupyter notebooks for experimentation and analysis.

## Getting Started

To replicate the experiments or use the tools provided, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/aaronhaefner/applications-of-llm.git
   ```

2. **Install Dependencies**:
   ```bash
   make setup
   ```

3. In progress...

## Model on Hugging Face

You can find the trained model on Hugging Face [here](https://huggingface.co/aaronhaefner/txt2sql_v1).

## Future Work

This project is ongoing, with plans for further development of tools and functions, as well as the integration of additional datasets and the evaluation of LLM applications across different domains.

## Contact

For questions or contributions, please contact me at [aaronhaefner@gmail.com](mailto:aaronhaefner@gmail.com).
