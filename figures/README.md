# Figure reproduction

## Manuscript figures

The seven figures cited by `main.qmd` are stored under `figures/manuscript/`. Every figure is supplied in:

- PDF for vector-quality print rendering;
- SVG for Computo HTML rendering and inspection;
- 300-dpi PNG as a portable fallback.

Regenerate all manuscript figures and tables with:

```bash
python scripts/make_manuscript_assets.py
```

The script reads committed raw and summary CSV files under `results/` and does not rerun optimization.

## Stand-alone audit figures

The original one-figure-per-script audit pipeline is retained:

- `data/figure_XX_*.csv`: exact figure-level input;
- `code/figure_XX_*.py`: independent plotting code;
- `output/figure_XX_*.png`: generated diagnostic output.

Run all stand-alone audit figures with:

```bash
python figures/code/run_all_figures.py
```

The manuscript does not cite the stand-alone diagnostic images; they are included to make individual result columns easy to inspect.
