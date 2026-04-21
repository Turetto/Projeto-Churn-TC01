.PHONY: install lint lint-fix test test-cov run-api train clean

install:
	uv sync --all-extras

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

lint-fix:
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/

test:
	uv run pytest tests/

test-cov:
	uv run pytest tests/ --cov=src --cov-report=html

run-api:
	uv run uvicorn src.api:app --reload --port 8000

train:
	uv run python -m src.train

clean:
	find . -type f -name "*.py[cod]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/ htmlcov/ dist/