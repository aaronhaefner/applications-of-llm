SHELL=bash

all: requirements.txt

# Variables
VENV_DIR = venv
REQUIREMENTS = requirements.txt
PACKAGES = torch torchvision torchaudio transformers datasets ipython jupyterlab matplotlib ipykernel IProgress python-dotenv ipywidgets
PYTHON = python3.12

# Recipes
$(REQUIREMENTS):
	$(MAKE) setup

.PHONY: ipy setup clean
# Launch interactive Python in the virtual environment
ipy:
	source $(VENV_DIR)/bin/activate && IPython

# Create the virtual environment and install the required packages
setup:
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install $(PACKAGES)
	$(VENV_DIR)/bin/pip freeze > $(REQUIREMENTS)

# Reset venv
clean:
	rm -rf $(VENV_DIR) $(REQUIREMENTS)