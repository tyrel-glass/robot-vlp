#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = robot-vlp
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 robot_vlp
	isort --check --diff --profile black robot_vlp
	black --check --config pyproject.toml robot_vlp

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml robot_vlp


.PHONY reset:
reset:
##	Remove processed dataset
	rm data/processed/model_train_test_data.pickle
##	Remove all generated path data
	rm data/interim/path_data 
##	Remove vlp models
	rm models/vlp/vlp_models.pkl
## Remove vlp dataset
	rm data/external/vlp_dataset.csv


## Set up python interpreter environment
.PHONY: create_environment
create_environment:
	
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



## Test the model


## Train the model
.PHONY: train_model
train_model : models/model_01.keras
models/model_01.keras: robot_vlp/modeling/train_mlp.py data/processed/data.pickle 
	$(PYTHON_INTERPRETER) robot_vlp/modeling/train_mlp.py


## Process the paths into a dataset
.PHONY: data
high_data : data/processed/data.pickle
data/processed/data.pickle : robot_vlp/data/preprocessing.py data/interim/path_data
	$(PYTHON_INTERPRETER) robot_vlp/data/preprocessing.py \
	--dataset-save-name data.pickle \
	--window-len 30 \
	--overlap 0.8

#	# --skip err_2 \
	# --skip err_3 \
	# --skip low_acc \
	# --skip med_acc
#
## Generate path datasets
#
# .PHONY:  run_robot_path
# run_robot_path: data/interim/path_data
data/interim/path_data : models/vlp/vlp_models.pkl robot_vlp/data/path_generation.py robot_vlp/robot.py
	$(PYTHON_INTERPRETER) robot_vlp/data/path_generation.py

## Create VLP models
models/vlp/vlp_models.pkl : data/external/vlp_dataset.csv robot_vlp/data/gen_vlp_model.py
	$(PYTHON_INTERPRETER) robot_vlp/data/gen_vlp_model.py

## Pull VLP Dataset
data/external/vlp_dataset.csv : robot_vlp/data/pull_vlp_data.py
	$(PYTHON_INTERPRETER) robot_vlp/data/pull_vlp_data.py

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
