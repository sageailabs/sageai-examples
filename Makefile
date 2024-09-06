SHELL := /usr/bin/env bash -O globstar

.PHONY: fix
fix:
	poetry run black .
	poetry run isort .

.PHONY: install
install:
	LDFLAGS="${LDFLAGS} -fno-lto -Wl,--no-as-needed" poetry install --no-root

.PHONY: lint
lint:
	poetry run black --check --diff .
	poetry run isort --check .
	poetry run flake8 .
	poetry run mypy --ignore-missing-imports .
