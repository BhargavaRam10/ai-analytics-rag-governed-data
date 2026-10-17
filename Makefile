.PHONY: setup data index eval test clean

setup:
	pip install -r requirements.txt
	cp -n .env.example .env || true

data:
	python data/generate_governed_docs.py

index:
	python -m src.build_index

eval:
	python -m src.eval

test:
	python -m pytest tests/ -q

clean:
	rm -rf data/index __pycache__ .pytest_cache
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
