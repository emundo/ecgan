PYTHON_INTERPRETER = python
SHELL := /bin/bash
SOURCES = ecgan
MAX_LINE_LENGTH = 120
CURRENT_DIR = $(shell pwd)
VENV_NAME = .venv

install:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -U -r requirements.txt

setup:
	make install
	mkdir results && mkdir artifacts

setup_dev:
	make setup
	$(PYTHON_INTERPRETER) -m pip install mypy pytest pylint black pydocstyle isort types-all
	pre-commit install

env:
	test -d $(VENV_NAME) || virtualenv --python=/usr/bin/python3.8 $(VENV_NAME)

check:
	$(PYTHON_INTERPRETER) -m pylint $(SOURCES) --fail-under=9.8 --rcfile=.pylintrc
	$(PYTHON_INTERPRETER) -m mypy $(SOURCES)
	$(PYTHON_INTERPRETER) -m pydocstyle $(SOURCES) --ignore=D203,D212,D413,D407,D406,D107 --ignore-decorators=property
	${PYTHON_INTERPRETER} -m isort ${SOURCES} --line-length=${MAX_LINE_LENGTH} --multi-line=3 --ensure-newline-before-comments --trailing-comma
	$(PYTHON_INTERPRETER) -m black $(SOURCES) --line-length=${MAX_LINE_LENGTH} --skip-string-normalization
