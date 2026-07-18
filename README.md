# Discovering active subspaces from noisy function values

Computo-format source and complete reproducibility artifact for the manuscript **Discovering active subspaces from noisy function values: Cross-fitted orthogonal sketches for dimension-adaptive zeroth-order optimization**.

## Contents

- `main.qmd`: complete article, theorem statements, proofs, executable examples, figures, and tables;
- `_quarto.yml`: Computo project metadata and render targets;
- `src/discoverzo/`: installable Python implementation;
- `results/raw/` and `results/summary/`: verified cached experimental outputs;
- `datasets/`: public static instances, seed ledger, and mirrors of raw/summary outputs;
- `figures/manuscript/`: all seven figures in PDF, SVG, and 300-dpi PNG;
- `tables/manuscript/`: all manuscript tables in Markdown and CSV;
- `scripts/`: experiment, analysis, asset-generation, and validation programs;
- `tests/`: unit and integration tests;
- `.github/workflows/`: Computo build and independent validation workflows.

## Build

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
quarto add computorg/computo-quarto-extension@v1.0.4 --no-prompt
quarto render
```

The official `computo-html` and `computo-pdf` outputs are written to `_site/`.

## Regenerate manuscript figures and tables

```bash
python scripts/make_manuscript_assets.py
```

## Validate code and cached results

```bash
pytest
python scripts/validate_research_artifact.py
python scripts/validate_manuscript.py
```

See `REPRODUCIBILITY.md`, `docs/experiment_protocol.md`, `docs/theory_audit.md`, and `docs/VALIDATION.md` for details.

## Required metadata edit

Before submission, replace `replace-with-github-user` in `_quarto.yml` and `main.qmd`, and add the corresponding-author email. The supplied manuscript deliberately omits a DOI and final publication URL.
