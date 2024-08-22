# Variables
VENV_DIR = venv
REQUIREMENTS = ../requirements.txt
PACKAGES = torch torchvision torchaudio transformers datasets ipython SentencePiece accelerate dotenv nltk
PYTHON = python3
APP_SCRIPT = app.py

all: $(REQUIREMENTS)

# Create the virtual environment and install the required packages
$(REQUIREMENTS):
	$(PYTHON) -m venv $(VENV_DIR)
	. $(VENV_DIR)/bin/activate && \
	pip install --upgrade pip && \
	pip install $(PACKAGES) && \
	pip freeze > $(REQUIREMENTS)

# Run app.py with different input arguments
.PHONY: run_none run_train run_paraphrase clean

run_none:
	. $(VENV_DIR)/bin/activate && \
	$(PYTHON) -m $(APP_SCRIPT) none

run_train:
	. $(VENV_DIR)/bin/activate && \
	$(PYTHON) -m $(APP_SCRIPT) train

run_paraphrase:
	. $(VENV_DIR)/bin/activate && \
	$(PYTHON) -m $(APP_SCRIPT) paraphrase

clean:
	rm -rf $(VENV_DIR) $(REQUIREMENTS)