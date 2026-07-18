# Validation record

Validation date: 2026-07-18

## Environment used

- Linux x86_64
- Python 3.13.5
- NumPy 2.3.5
- SciPy 1.17.0
- pandas 2.2.3
- Matplotlib 3.10.8
- PyYAML 6.0.3
- tqdm 4.67.3
- pytest 9.0.2
- Pandoc 3.1.11.1
- XeLaTeX / TeX Live 2025 development snapshot

The sandbox did not contain the Quarto CLI. The final repository therefore includes both the pinned official Computo-extension setup command and a GitHub Actions workflow that performs the exact `quarto render` in a standard Quarto environment.

## Tests actually executed

```bash
python -m pip install -e ".[test]"
pytest -q
```

Result: **12 passed**.

The tests cover benchmark optima, query counting, cross-moment estimation, PSD handling, rank selection, full/restricted optimizer descent, compact experiment schemas, and the arbitrary-orthogonal-batch population formula.

## Data validation actually executed

```bash
python scripts/validate_research_artifact.py
```

Result: passed.

| Raw file | Rows | Missing-value/schema check |
|---|---:|---|
| `moment_bias.csv` | 420 | passed |
| `dimension_scaling.csv` | 120 | passed |
| `optimization.csv` | 480 | passed |
| `amortization.csv` | 120 | passed |

The validator also checks the committed SHA-256 values of these files.

## Manuscript validation actually executed

```bash
python scripts/make_manuscript_assets.py
python scripts/validate_manuscript.py
```

Result: passed with **16 cited bibliography keys**, **32 unique labeled identifiers**, and **7 figures in PDF, SVG, and PNG**. All tables were regenerated from committed summary CSV files. Paired intervals use 20,000 fixed-seed bootstrap resamples.

## Fallback compilation actually executed

Because Quarto itself was unavailable, a validation-only Markdown copy was produced by replacing Quarto executable-fence syntax and humanizing native cross-reference tokens. It was then compiled with Pandoc and XeLaTeX. This fallback is not included as the journal output and does not emulate the Computo title template.

Validation outcome:

- 7/7 manuscript images recognized by Pandoc;
- 25-page PDF produced;
- zero XeLaTeX/Pandoc PDF warnings;
- all fonts embedded;
- PDF preflight passed and all 25 pages rendered to images for visual inspection;
- no clipped figures, broken glyphs, or overlapping text observed.

Exact Computo rendering is performed by:

```bash
quarto add computorg/computo-quarto-extension@v1.0.4 --no-prompt
quarto render
```

and is encoded in `.github/workflows/validation.yml`.

## Experiment execution record

The four manuscript-facing experiment components were run during artifact construction and their repetition-level results are committed. The optimization matrix was executed in five deterministic batches and merged into ten repetitions per benchmark-method-structure setting. The reduced end-to-end CLI run also completed.

A later clean invocation of the entire verified runner in a single sandbox command exceeded the 20-minute command window after regenerating the calibration component. This was a wall-clock interruption rather than an exception in an experiment component. The canonical runner is restartable, and the manuscript uses only the previously completed, integrity-checked files.

## Optional COCO limitation

The compiled `cocoex` dependency was not used for the cached results because its build failed under Python 3.13 in the validation environment. The default study uses auditable canonical BBOB formulas. An optional adapter and separate environment specification are included for a COCO-supported Python version.

## Submission metadata still required

Before journal submission, the author must replace `replace-with-github-user` and add a corresponding-author email. No DOI or final publication URL is included.
