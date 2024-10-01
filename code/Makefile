SHELL := bash

all: requirements.txt

include .env
export

REQUIREMENTS := requirements.txt
CONDA_ENV := torchenv
PYTHON_VERSION := 3.11
PYTHON := python
PACKAGES = torch torchvision torchaudio transformers datasets ipython SentencePiece accelerate python-dotenv nltk
APP_SCRIPT = app.py

# Install the required packages in the conda environment
$(REQUIREMENTS):
	conda activate $(CONDA_ENV) && \
	pip install --upgrade pip && \
	pip install $(PACKAGES) && \
	pip freeze > $(REQUIREMENTS)

# Run app.py with different input arguments locally
.PHONY: install-packages setup activate run_none run_train run_paraphrase clean submit

install-packages:
	@echo "Activating conda environment '$(CONDA_ENV)' and installing packages..."
	. $$(conda info --base)/etc/profile.d/conda.sh && \
	conda activate $(CONDA_ENV) && \
	pip install $(PACKAGES) && \
	pip freeze > $(REQUIREMENTS)
	@echo "Packages installed. Requirements exported to $(REQUIREMENTS)"

setup:
	conda init bash
	conda create -n $(CONDA_ENV) python=$(PYTHON_VERSION) -y
	$(MAKE) activate
	source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $(CONDA_ENV)
	$(MAKE) install-packages
	@echo "Now, you activate the conda environment with 'conda activate $(CONDA_ENV)'"

remake: clean
	@echo "Recreating conda environment '$(CONDA_ENV)' with Python $(PYTHON_VERSION)..."
	$(MAKE) setup

run_none:
	conda activate $(CONDA_ENV) && \
	$(PYTHON) $(APP_SCRIPT) none

run_train:
	conda activate $(CONDA_ENV) && \
	$(PYTHON) $(APP_SCRIPT) train

run_paraphrase:
	conda activate $(CONDA_ENV) && \
	$(PYTHON) $(APP_SCRIPT) paraphrase

# Submit a job using SLURM
submit:
	sbatch submit_job.sbatch

del_logs:
	rm -rf slurmlogs/*
