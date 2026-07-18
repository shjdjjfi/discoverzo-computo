# Build validation summary

Date: 2026-07-18

## Completed checks

- Python package installed in editable mode.
- Unit and integration tests: **12 passed**.
- Raw-result integrity: **420 + 120 + 480 + 120 rows**, with schema, missing-value, and SHA-256 checks passed.
- Manuscript audit: **16 cited bibliography keys**, **32 unique cross-reference identifiers**, and **7 manuscript figures in PDF, SVG, and 300-dpi PNG**.
- Manuscript assets regenerated from committed CSV files.
- Generic Pandoc/XeLaTeX fallback: **25 pages**, **7/7 images**, zero PDF warnings, embedded fonts, preflight passed.
- Every fallback PDF page was rendered to PNG and visually inspected for clipping, overlaps, missing figures, and broken glyphs.

## Exact Computo build

The repository pins the official Computo Quarto extension at version `v1.0.4` and supplies the exact commands:

```bash
quarto add computorg/computo-quarto-extension@v1.0.4 --no-prompt
quarto render
```

The sandbox used for artifact construction did not contain the Quarto CLI, so the official Computo template itself was not rendered locally. `.github/workflows/validation.yml` installs Quarto, runs all tests and validators, installs the pinned extension, renders the article, and uploads `_site/`.

The validation-only generic PDF and its temporary Markdown/TeX files are intentionally excluded from the submission ZIP to avoid confusing them with the official Computo output.
