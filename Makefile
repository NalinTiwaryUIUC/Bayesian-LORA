venv:
	python -m venv .venv

install:
	pip install -e .
	pip install -r requirements.txt

test:
	pytest -q

format:
	black . && ruff check --fix .
