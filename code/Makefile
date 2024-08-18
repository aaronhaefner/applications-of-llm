SHELL=bash

all: requirements.txt

# Variables
VENV_DIR = venv
REQUIREMENTS = requirements.txt
PACKAGES = torch torchvision torchaudio transformers datasets ipython SentencePiece accelerate dotenv nltk
PYTHON = python3

# Recipes
$(REQUIREMENTS):
	$(MAKE) setup

.PHONY: setup clean

# Create the virtual environment and install the required packages
setup:
	$(PYTHON) -m venv $(VENV_DIR)
	source $(VENV_DIR)/bin/activate &&
	pip install --upgrade pip
	pip install $(PACKAGES)
	pip freeze > $(REQUIREMENTS)

# Reset venv
clean:
	rm -rf $(VENV_DIR) $(REQUIREMENTS)