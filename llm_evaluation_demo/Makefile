SHELL := /usr/bin/env bash -O globstar

.PHONY: fix
fix:
	poetry run black .
	poetry run isort .

.PHONY: lint
lint:
	poetry run black --check --diff .
	poetry run isort --check .
	poetry run flake8 .
	poetry run pylint gms_llm_gateway/**/*.py
	poetry run mypy --ignore-missing-imports .

