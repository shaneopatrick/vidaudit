.PHONY: install test lint typecheck check demo eval

install:
	uv sync
	uv run python -m spacy download en_core_web_sm

test:
	uv run pytest tests/ -v

lint:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy vidaudit/

check: lint typecheck test

demo:
	uv run vidaudit audit \
		--video examples/sample.mp4 \
		--descriptions examples/sample_descriptions.json \
		--output report.json

eval:
	uv run python eval/run_eval.py
