.PHONY: install extension test assets smoke reproduce render validate clean

install:
	python -m pip install -e ".[test]"

extension:
	quarto add computorg/computo-quarto-extension@v1.0.4 --no-prompt

test:
	PYTHONPATH=src pytest

assets:
	PYTHONPATH=src python scripts/make_manuscript_assets.py

smoke:
	PYTHONPATH=src python -m discoverzo.cli --config configs/quick.yaml --output results/quick

reproduce:
	PYTHONPATH=src python scripts/run_verified.py

render: extension
	quarto render

validate: test assets
	python scripts/validate_research_artifact.py
	python scripts/validate_manuscript.py

clean:
	rm -rf scratch .pytest_cache _site .quarto _freeze
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
