from pathlib import Path
import hashlib, json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
expected = {"moment_bias.csv": 420, "dimension_scaling.csv": 120,
            "optimization.csv": 480, "amortization.csv": 120}
report = {"status": "passed", "files": []}
for name, n in expected.items():
    p = ROOT / "datasets" / "raw_outputs" / name
    df = pd.read_csv(p)
    assert len(df) == n and not df.isna().any().any()
    report["files"].append({"path": str(p.relative_to(ROOT)), "rows": n,
                            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
assert len(list((ROOT / "figures" / "code").glob("figure_*.py"))) == 7
assert len(list((ROOT / "figures" / "data").glob("figure_*.csv"))) == 7
assert len(list((ROOT / "figures" / "output").glob("figure_*.png"))) == 7
assert len(list((ROOT / "figures" / "manuscript").glob("*.pdf"))) == 7
assert len(list((ROOT / "figures" / "manuscript").glob("*.svg"))) == 7
assert len(list((ROOT / "figures" / "manuscript").glob("*.png"))) == 7
assert (ROOT / "tables" / "TABLES.md").stat().st_size > 5000
assert (ROOT / "tables" / "manuscript" / "TABLES.md").stat().st_size > 3000
assert (ROOT / "REPRODUCIBILITY.md").stat().st_size > 5000
assert (ROOT / "main.qmd").stat().st_size > 40000
for p in (ROOT / "datasets" / "generated_inputs").glob("*.npz"):
    assert p.stat().st_size > 0
print(json.dumps(report, indent=2))
