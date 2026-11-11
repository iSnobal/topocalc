.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

help:
	@sed -n 's/^\([A-Za-z0-9_.-]*\):.*## \(.*\)$$/\1:\2/p' $(MAKEFILE_LIST) | sort | column -t -s ':'

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +
	find . -name '*.so' -type f -delete

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## Remove test artifacts
	rm -fr .pytest_cache

tests: build-cython ## Run tests
	python -m unittest discover topocalc/tests

install: clean ## Install the package
	python -m pip install -e .

install-dev: clean build-cython ## Install the package in development environment
	python -m pip install --no-deps -e .

build-cython: clean-pyc ## Build Cython extension
	python setup.py build_ext --inplace

