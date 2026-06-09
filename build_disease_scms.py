"""
build_disease_scms.py
======================
Builds NOTEARS structural causal models for all diseases
and computes pathway conditioning vectors Δv.

Usage:
    py -3.11 build_disease_scms.py

Output in data/ folder:
    W_{disease}.npy          - NOTEARS adjacency matrix
    delta_v_{disease}.npy    - pathway conditioning vector
    v_disease_{disease}.npy  - disease mean expression
    v_healthy_{disease}.npy  - healthy mean expression
"""

import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
import json

data_dir = Path('data')
data_dir.mkdir(exist_ok=True)

expr_dir = Path('expression')

# All disease expression files
DISEASES = [
    'malaria2', 'sarcoidosis2', 'ms_blood', 'hiv',
    'leishmaniasis', 'leishmaniasis2', 'dengue', 'tuberculosis',
    'dipg',   # synthetic — from earlier
    'ad',     # synthetic — from earlier
]

def notears_fast(X, lambda1=0.05, max_iter=150, h_tol=1e-6):
    """Fast NOTEARS SCM learning."""
    n, d = X.shape
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    W = np.zeros((d, d))
    rho, alpha, h_prev = 1.0, 0.0, np.inf

    def h_func(W):
        return float(np.sum(W * W))

    def obj(w_flat):
        W_ = w_flat.reshape(d, d)
        np.fill_diagonal(W_, 0)
        loss = 0.5 / n * np.sum((X - X @ W_) ** 2)
        hval = h_func(W_)
        pen  = lambda1 * np.sum(np.abs(W_))
        return loss + 0.5 * rho * hval**2 + alpha * hval + pen

    for it in range(max_iter):
        result = minimize(obj, W.flatten(), method='L-BFGS-B',
                         options={'maxiter': 50, 'ftol': 1e-10})
        W = result.x.reshape(d, d)
        np.fill_diagonal(W, 0)
        h_new = h_func(W)
        if h_new > 0.25 * h_prev:
            rho = min(rho * 5, 1e10)
        alpha += rho * h_new
        if h_new <= h_tol:
            break
        h_prev = h_new

    return W

results = {}

for disease in DISEASES:
    # Find expression file
    csv_path = expr_dir / f'{disease}_expression.csv'

    if not csv_path.exists():
        print(f"\n  {disease}: expression file not found — skipping")
        continue

    print(f"\n{'─'*50}")
    print(f"Building SCM: {disease.upper()}")

    df = pd.read_csv(csv_path, index_col=0)
    gene_cols = [c for c in df.columns if c != 'label']

    disease_df = df[df['label'] == 'disease'][gene_cols].values
    control_df = df[df['label'] == 'control'][gene_cols].values

    print(f"  Disease: {len(disease_df)} samples  Control: {len(control_df)} samples  Genes: {len(gene_cols)}")

    if len(control_df) == 0:
        print(f"  No control samples — skipping")
        continue

    # Use top 20 most variable genes for NOTEARS (fast)
    var = disease_df.var(0)
    top_idx   = np.argsort(var)[::-1][:20]
    top_genes = [gene_cols[i] for i in top_idx]
    X = disease_df[:, top_idx]

    print(f"  Running NOTEARS on top 20 variable genes...")
    W = notears_fast(X, lambda1=0.05, max_iter=150)

    n_edges = int(np.sum(np.abs(W) > 0.05))
    print(f"  SCM: 20 nodes, {n_edges} edges")

    # Compute Δv over ALL genes
    v_disease = disease_df.mean(0)
    v_healthy = control_df.mean(0)
    delta_v   = v_healthy - v_disease

    # Normalize delta_v to unit norm
    norm = np.linalg.norm(delta_v)
    if norm > 0:
        delta_v_norm = delta_v / norm
    else:
        delta_v_norm = delta_v

    # Save everything
    np.save(data_dir / f'W_{disease}.npy',         W)
    np.save(data_dir / f'v_disease_{disease}.npy',  v_disease)
    np.save(data_dir / f'v_healthy_{disease}.npy',  v_healthy)
    np.save(data_dir / f'delta_v_{disease}.npy',    delta_v_norm)

    with open(data_dir / f'{disease}_scm_genes.txt', 'w') as f:
        f.write('\n'.join(top_genes))

    # Top up/down genes
    top5_up   = [gene_cols[i] for i in np.argsort(delta_v)[::-1][:5]]
    top5_down = [gene_cols[i] for i in np.argsort(delta_v)[:5]]
    print(f"  Need UP:   {', '.join(top5_up)}")
    print(f"  Need DOWN: {', '.join(top5_down)}")

    results[disease] = {
        'n_disease': len(disease_df),
        'n_control': len(control_df),
        'n_edges':   n_edges,
        'top_up':    top5_up,
        'top_down':  top5_down,
    }

# Save results summary
with open(data_dir / 'scm_summary.json', 'w') as f:
    json.dump(results, f, indent=2)

# Final summary
print(f"\n{'='*50}")
print(f"SCM CONSTRUCTION COMPLETE")
print(f"{'='*50}")
print(f"{'Disease':20s}  {'Samples':>8s}  {'Edges':>6s}")
print(f"{'─'*40}")
for disease, info in results.items():
    total = info['n_disease'] + info['n_control']
    print(f"{disease:20s}  {total:>8d}  {info['n_edges']:>6d}")

print(f"\nFiles saved to data/")
print(f"  W_{{disease}}.npy         - adjacency matrices")
print(f"  delta_v_{{disease}}.npy   - conditioning vectors")
print(f"\nNext: run train_causal.py with any disease")
print(f"  py -3.11 train_causal.py --disease ms_blood --epochs 150 --batch_size 128")
