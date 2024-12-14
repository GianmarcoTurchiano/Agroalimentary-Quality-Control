#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = agroalimentary_quality_control
PYTHON_VERSION = 3.10
PYTHON_ENV_NAME = .venv
REQUIREMENTS_FILE_NAME = requirements.in
PYTHON_INTERPRETER = python$(PYTHON_VERSION)
VENV_BIN_PATH = $(PYTHON_ENV_NAME)/bin
PYTHON_INTERPRETER_VENV = $(VENV_BIN_PATH)/python$(PYTHON_VERSION)

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Create and then activate a new python virtual environment
.PHONY: create_environment
create_environment:
	$(PYTHON_INTERPRETER) -m venv $(PYTHON_ENV_NAME)

.PHONY: repro
repro:
	$(VENV_BIN_PATH)/dvc repro

## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER_VENV) -m pip install -U pip
	$(PYTHON_INTERPRETER_VENV) -m pip install -r requirements.txt
	
## Exports dependencies and compiles them into a conflict-free requirements file.
.PHONY: export_requirements
export_requirements:
	$(PYTHON_INTERPRETER_VENV) -m pip freeze > $(REQUIREMENTS_FILE_NAME)
	pip-compile $(REQUIREMENTS_FILE_NAME)
	
## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Checks the integrity of the DVC pipeline
.PHONY: pipeline_check
pipeline_check:
	dvc repro --dry

## Launches behavioral tests on the current model.
.PHONY: test_behavior
test_behavior:
	$(PYTHON_INTERPRETER_VENV) -m pytest $(PROJECT_NAME)/modeling/tests/functional_tests/

## Soft resets to the previous git commit
.PHONY: rollback
rollback:
	git reset --soft HEAD~1

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



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
	@$(PYTHON_INTERPRETER_VENV) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
