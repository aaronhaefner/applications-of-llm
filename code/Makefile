# Load environment variables from .env file
include .env
export

# Variables
REQUIREMENTS = ../requirements.txt
PACKAGES = torch torchvision torchaudio transformers datasets ipython SentencePiece accelerate dotenv nltk
PYTHON = python3
APP_SCRIPT = app.py
SBATCH_FILE = submit_job.sbatch

# Default target
all: $(REQUIREMENTS)

# Install the required packages in the conda environment
$(REQUIREMENTS):
	conda activate $(CONDA_ENV) && \
	pip install --upgrade pip && \
	pip install $(PACKAGES) && \
	pip freeze > $(REQUIREMENTS)

# Run app.py with different input arguments locally
.PHONY: run_none run_train run_paraphrase clean submit

run_none:
	conda activate $(CONDA_ENV) && \
	$(PYTHON) $(APP_SCRIPT) none

run_train:
	conda activate $(CONDA_ENV) && \
	$(PYTHON) $(APP_SCRIPT) train

run_paraphrase:
	conda activate $(CONDA_ENV) && \
	$(PYTHON) $(APP_SCRIPT) paraphrase

# Clean the environment
clean:
	rm -rf $(CONDA_ENV) $(REQUIREMENTS)

# Submit a job using SLURM
submit:
	sbatch $(SBATCH_FILE)
