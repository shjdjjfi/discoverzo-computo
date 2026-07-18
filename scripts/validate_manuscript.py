#!/usr/bin/env python3
"""Static validation for the Computo manuscript source."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
main = (ROOT / "main.qmd").read_text(encoding="utf-8")
quarto = (ROOT / "_quarto.yml").read_text(encoding="utf-8")
bib = (ROOT / "references.bib").read_text(encoding="utf-8")

errors: list[str] = []
for required in ["computo-html", "computo-pdf", "status: submitted"]:
    if required not in main + "\n" + quarto:
        errors.append(f"missing Computo metadata: {required}")

bib_keys = set(re.findall(r"@[A-Za-z]+\{([^,]+)", bib))
all_refs = set(re.findall(r"(?<![#A-Za-z0-9_-])@([A-Za-z][A-Za-z0-9_:-]+)", main))
prefixes = ("fig-", "tbl-", "thm-", "prp-", "cor-", "lem-", "eq-", "sec-", "lst-", "algo-")
citations = {x.rstrip(":;,. ") for x in all_refs if not x.startswith(prefixes)}
missing = sorted(citations - bib_keys)
if missing:
    errors.append(f"missing bibliography keys: {missing}")

ids = re.findall(r"\{#([A-Za-z0-9_-]+)(?:\s+[^}]*)?\}", main)
duplicates = sorted({x for x in ids if ids.count(x) > 1})
if duplicates:
    errors.append(f"duplicate cross-reference IDs: {duplicates}")
chunk_labels = set(re.findall(r"#\| label:\s*([A-Za-z0-9_-]+)", main))
known = set(ids) | chunk_labels
refs = {x.rstrip(":;,. ") for x in all_refs if x.startswith(prefixes)}
unresolved = sorted(refs - known)
if unresolved:
    errors.append(f"unresolved cross references: {unresolved}")

for rel in re.findall(r"\]\(([^)]+)\)", main):
    if rel.startswith(("http://", "https://", "mailto:")):
        continue
    if not (ROOT / rel).exists():
        errors.append(f"missing linked asset: {rel}")

for stem in [f"fig{i:02d}" for i in range(1, 8)]:
    matches = list((ROOT / "figures" / "manuscript").glob(stem + "_*"))
    exts = {p.suffix for p in matches}
    if not {".pdf", ".svg", ".png"}.issubset(exts):
        errors.append(f"missing publication formats for {stem}: {sorted(exts)}")

if errors:
    print("Manuscript validation failed:")
    for e in errors:
        print(" -", e)
    sys.exit(1)
print(f"Manuscript validation passed: {len(citations)} citations, {len(ids)} unique IDs, 7 figures in 3 formats.")
