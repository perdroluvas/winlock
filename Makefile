.PHONY: setup lint fmt test nb

setup:
	uv sync
	uv run nbstripout --install

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

nb:
	uv run jupyter lab notebooks/
