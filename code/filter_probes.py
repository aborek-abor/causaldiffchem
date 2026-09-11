"""
filter_probes.py
================
Removes Affymetrix/Illumina probe ID columns from expression files
that contain a mix of probe IDs and gene symbols.

Probe ID patterns detected:
  - 1234567_at, 1234567_s_at, 1234567_x_at   (Affymetrix)
  - 1234567_PM_at, 1234567_PM_s_at            (Affymetrix PM)
  - ILMN_1234567                               (Illumina)
  - Pure numeric column names

Gene symbols are kept: XIST, HLA-DRB4, SCGB2A2, ARG1 etc.

Usage:
    py -3.11 filter_probes.py
"""

import re
import pandas as pd
from pathlib import Path

DISEASES = ['dengue', 'sarcoidosis2', 'breast_cancer', 'breast_cancer2']

# Probe ID patterns
probe_patterns = [
    re.compile(r'^\d+_[a-z]_at$'),           # 123_s_at, 123_x_at
    re.compile(r'^\d+_at$'),                   # 123_at
    re.compile(r'^\d+_PM_at$'),               # 123_PM_at
    re.compile(r'^\d+_PM_[a-z]_at$'),         # 123_PM_s_at
    re.compile(r'^\d+_PM_x_at$'),             # 123_PM_x_at
    re.compile(r'^ILMN_\d+$'),                # Illumina probes
    re.compile(r'^\d+$'),                      # pure numeric
    re.compile(r'^\d{6,}_.*_at$'),            # long numeric prefix _at
    re.compile(r'^\d{6,}_.*at$'),             # variants
]

def is_probe_id(col):
    col = str(col).strip()
    for p in probe_patterns:
        if p.match(col):
            return True
    return False

print(f"{'Disease':20s}  {'Total':>6s}  {'Probes':>7s}  {'Symbols':>8s}  Status")
print('-' * 60)

for disease in DISEASES:
    path = Path(f'expression/{disease}_expression.csv')
    if not path.exists():
        print(f"{disease:20s}  FILE NOT FOUND")
        continue

    df = pd.read_csv(path)

    # Separate meta columns from gene columns
    meta_cols = [c for c in df.columns if c in ('label', 'Unnamed: 0', 'disease_subtype')]
    gene_cols = [c for c in df.columns if c not in meta_cols]

    probe_cols  = [c for c in gene_cols if is_probe_id(c)]
    symbol_cols = [c for c in gene_cols if not is_probe_id(c)]

    print(f"{disease:20s}  {len(gene_cols):6d}  {len(probe_cols):7d}  "
          f"{len(symbol_cols):8d}  ", end='')

    if len(symbol_cols) < 10:
        print("WARNING: very few gene symbols — check file")
        continue

    # Keep only gene symbol columns + meta columns
    keep_cols = meta_cols + symbol_cols
    df_clean = df[keep_cols]

    # Verify still z-scored
    vals = df_clean[symbol_cols].values.astype(float)
    mean_val = vals.mean()

    df_clean.to_csv(path, index=False)
    print(f"saved ({len(symbol_cols)} symbols, mean={mean_val:.3f})")

print(f"\nDone. Now rerun NOTEARS for all 4 diseases:")
print("  py -3.11 proper_notears.py --disease dengue")
print("  py -3.11 proper_notears.py --disease sarcoidosis2")
print("  py -3.11 proper_notears.py --disease breast_cancer")
print("  py -3.11 proper_notears.py --disease breast_cancer2")
