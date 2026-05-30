.PHONY: install test lint typecheck check demo eval

EXAMPLE_CLIP_URL := https://www.pexels.com/download/video/37552285/?fps=25.0&h=720&w=1280

install: examples/clip.mp4
	uv sync
	uv run python -m spacy download en_core_web_sm

examples/clip.mp4:
	curl -L --fail -o examples/clip.mp4 "$(EXAMPLE_CLIP_URL)"

test:
	uv run pytest tests/ -v

lint:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy vidaudit/ eval/ tests/

check: lint typecheck test

demo:
	uv run vidaudit audit \
		--video examples/clip.mp4 \
		--descriptions examples/descs.json \
		--output report.json

eval:
	uv run python eval/run_eval.py
