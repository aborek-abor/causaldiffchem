"""
export_expression_vectors.py
===============================
Packages v_disease/v_healthy for all diseases into one CSV, so they can be
uploaded in a single file instead of 20+ separate .npy files.

Run from the causaldiffchem/ folder (same level as data/):
    py -3.11 export_expression_vectors.py

Writes: expression_vectors_all_diseases.csv
  columns: disease, gene, v_disease, v_healthy
"""

import numpy as np
import pandas as pd
from pathlib import Path

data_dir = Path('data')

DISEASES = ['ms_blood', 'hiv', 'ad', 'tuberculosis_clean', 'dengue', 'sarcoidosis2',
            'leishmaniasis2', 'malaria2', 'breast_cancer', 'breast_cancer2', 'sch_haem']

rows = []
missing = []

for disease in DISEASES:
    v_dis_path = data_dir / f'v_disease_{disease}.npy'
    v_hea_path = data_dir / f'v_healthy_{disease}.npy'
    genes_path = data_dir / f'{disease}_scm_genes.txt'

    if not (v_dis_path.exists() and v_hea_path.exists() and genes_path.exists()):
        missing.append(disease)
        print(f"  {disease}: MISSING one or more files, skipping")
        continue

    v_dis = np.load(v_dis_path)[:20]
    v_hea = np.load(v_hea_path)[:20]
    genes = open(genes_path).read().strip().split('\n')[:20]

    for g, vd, vh in zip(genes, v_dis, v_hea):
        rows.append({'disease': disease, 'gene': g, 'v_disease': vd, 'v_healthy': vh})

    print(f"  {disease}: {len(genes)} genes exported")

df = pd.DataFrame(rows)
df.to_csv('expression_vectors_all_diseases.csv', index=False)
print(f"\nSaved: expression_vectors_all_diseases.csv ({len(df)} rows, "
      f"{df['disease'].nunique()} diseases)")
if missing:
    print(f"\nMissing data for: {missing} -- these diseases' node-movement charts "
          f"can't be built until their v_disease_*.npy / v_healthy_*.npy files are found.")
