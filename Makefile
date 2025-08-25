.PHONY: lint format precommit clean

lint:
	ruff check .

format:
	ruff format .

precommit:
	pre-commit install

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name "*.pyc" -delete
	@echo "Cleaned caches (__pycache__, *.pyc)"
