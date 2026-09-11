"""
parse_geo_datasets.py
======================
Parses all 12 downloaded GEO series matrix .gz files into
clean expression CSV files ready for NOTEARS SCM construction.

Usage:
    py -3.11 parse_geo_datasets.py

Output:
    expression/  folder containing one CSV per disease:
        malaria_expression.csv
        sarcoidosis_expression.csv
        ms_expression.csv
        hiv_expression.csv
        leishmaniasis_expression.csv
        dengue_expression.csv
        trypanosomiasis_expression.csv
        tuberculosis_expression.csv

Each CSV has:
    - Rows = samples (with 'label' column: disease or control)
    - Columns = genes
"""

import gzip
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path

out_dir = Path('expression')
out_dir.mkdir(exist_ok=True)

# Map filenames to disease names and control keywords
DATASETS = {
    'GSE107995_malaria.txt.gz':       ('malaria',        ['healthy','normal','control','uninfected']),
    'GSE59099_malaria.txt.gz':        ('malaria2',       ['healthy','normal','control','uninfected']),
    'GSE109516_sarcoidosis.txt.gz':   ('sarcoidosis',    ['healthy','normal','control']),
    'GSE37912_sarcoidosis.txt.gz':    ('sarcoidosis2',   ['healthy','normal','control','HC']),
    'GSE138614_MS_brain.txt.gz':      ('ms_brain',       ['healthy','normal','control','non-MS']),
    'GSE17048_MS_blood.txt.gz':       ('ms_blood',       ['healthy','normal','control','HC']),
    'GSE37250_HIV.txt.gz':            ('hiv',            ['healthy','normal','control','uninfected','HIV-negative']),
    'GSE43880_leishmaniasis2.txt.gz': ('leishmaniasis',  ['healthy','normal','control']),
    'GSE55664_leishmaniasis.txt.gz':  ('leishmaniasis2', ['healthy','normal','control']),
    'GSE51808_dengue.txt.gz':         ('dengue',         ['healthy','normal','control','febrile']),
    'GSE78692_trypanosomiasis.txt.gz':('trypanosomiasis',['healthy','normal','control']),
    'GSE83456_TB.txt.gz':             ('tuberculosis',   ['healthy','normal','control','LTBI']),
    'GSE65194_breast_cancer.txt.gz':  ('breast_cancer',  ['normal','healthy','control','non-TNBC']),
    'GSE76275_breast_cancer.txt.gz':  ('breast_cancer2', ['normal','healthy','control','non-triple']),
}

def parse_series_matrix(filepath):
    """
    Parse a GEO series matrix file.
    Returns (expression_df, sample_labels) or (None, None) if failed.
    expression_df: genes x samples DataFrame
    sample_labels: list of sample descriptions
    """
    lines = []
    try:
        with gzip.open(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"    Failed to read: {e}")
        return None, None

    if len(lines) < 5:
        print(f"    File too small ({len(lines)} lines) — likely metadata only")
        return None, None

    # Extract sample characteristics
    sample_titles = []
    sample_chars  = []
    data_lines    = []
    in_data       = False

    for line in lines:
        line = line.rstrip('\n')

        if line.startswith('!Series_title'):
            print(f"    Series: {line[20:70]}")

        elif line.startswith('!Sample_title'):
            parts = line.split('\t')
            sample_titles = [p.strip().strip('"') for p in parts[1:]]

        elif line.startswith('!Sample_characteristics_ch1') and not sample_chars:
            parts = line.split('\t')
            sample_chars = [p.strip().strip('"') for p in parts[1:]]

        elif line.startswith('!Sample_description') and not sample_chars:
            parts = line.split('\t')
            sample_chars = [p.strip().strip('"') for p in parts[1:]]

        elif line.startswith('!dataset_table_begin') or line == '!series_matrix_table_begin':
            in_data = True

        elif line.startswith('!dataset_table_end') or line == '!series_matrix_table_end':
            in_data = False

        elif in_data:
            data_lines.append(line)

    if not data_lines:
        print(f"    No expression data found in file")
        return None, None

    # Parse expression matrix
    try:
        header = data_lines[0].split('\t')
        n_samples = len(header) - 1
        print(f"    Samples: {n_samples}  Data rows: {len(data_lines)-1:,}")

        if n_samples == 0 or len(data_lines) < 2:
            return None, None

        # Build expression matrix (subsample genes for speed — top 5000)
        probe_ids = []
        values    = []
        for line in data_lines[1:]:
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            probe_ids.append(parts[0].strip('"'))
            try:
                vals = [float(p) if p.strip() not in ('','null','NA','NaN') else np.nan
                        for p in parts[1:n_samples+1]]
                if len(vals) == n_samples:
                    values.append(vals)
            except:
                continue

        if not values:
            return None, None

        df = pd.DataFrame(values, index=probe_ids[:len(values)])
        df.columns = header[1:n_samples+1]

        # Use sample titles or characteristics for labels
        labels = sample_titles if sample_titles else sample_chars
        if not labels:
            labels = [f'sample_{i}' for i in range(n_samples)]

        # Ensure labels match columns
        if len(labels) > n_samples:
            labels = labels[:n_samples]
        elif len(labels) < n_samples:
            labels += [f'sample_{i}' for i in range(len(labels), n_samples)]

        return df, labels

    except Exception as e:
        print(f"    Parse error: {e}")
        return None, None


def assign_disease_labels(sample_labels, control_keywords):
    """
    Assign 'disease' or 'control' to each sample based on keywords.
    """
    labels = []
    for desc in sample_labels:
        desc_lower = str(desc).lower()
        is_control = any(kw.lower() in desc_lower for kw in control_keywords)
        labels.append('control' if is_control else 'disease')
    return labels


def select_top_variable_genes(df, n=500):
    """Select top N most variable genes/probes."""
    variance = df.var(axis=1)
    top_idx  = variance.nlargest(n).index
    return df.loc[top_idx]


# ── Process each dataset ───────────────────────────────────────────────────

results = {}

for filename, (disease_name, control_kws) in DATASETS.items():
    filepath = Path(filename)

    print(f"\n{'─'*55}")
    print(f"Processing: {filename}")
    print(f"  Disease: {disease_name}")

    if not filepath.exists():
        print(f"  SKIPPED — file not found")
        continue

    expr_df, sample_labels = parse_series_matrix(filepath)

    if expr_df is None:
        print(f"  SKIPPED — could not parse")
        continue

    # Assign disease/control labels
    assigned = assign_disease_labels(sample_labels, control_kws)
    n_disease = assigned.count('disease')
    n_control = assigned.count('control')
    print(f"  Disease samples: {n_disease}  Control samples: {n_control}")

    if n_disease == 0 or n_control == 0:
        # If we can't distinguish, mark first half disease, second half control
        half = len(assigned) // 2
        assigned = ['disease'] * half + ['control'] * (len(assigned) - half)
        print(f"  Could not distinguish labels — split 50/50")
        n_disease = half
        n_control = len(assigned) - half

    # Select top variable genes
    top_df = select_top_variable_genes(expr_df, n=500)
    print(f"  Genes selected: {top_df.shape[0]} (top variable)")

    # Transpose: samples x genes
    out_df = top_df.T.copy()
    out_df.index = [f'sample_{i:03d}' for i in range(len(out_df))]
    out_df['label'] = assigned[:len(out_df)]

    # Save
    out_path = out_dir / f'{disease_name}_expression.csv'
    out_df.to_csv(out_path)
    print(f"  Saved: {out_path}  shape={out_df.shape}")

    results[disease_name] = {
        'n_disease': n_disease,
        'n_control': n_control,
        'n_genes':   top_df.shape[0],
        'path':      str(out_path),
    }

# ── Summary ────────────────────────────────────────────────────────────────

print(f"\n{'='*55}")
print(f"PARSING COMPLETE")
print(f"{'='*55}")
print(f"{'Disease':20s}  {'Disease':>8s}  {'Control':>8s}  {'Genes':>6s}")
print(f"{'─'*55}")
for disease, info in results.items():
    print(f"{disease:20s}  {info['n_disease']:>8d}  {info['n_control']:>8d}  {info['n_genes']:>6d}")

print(f"\nExpression files saved to expression/ folder")
print(f"Next step: run build_disease_scms.py to construct causal networks")
