"""
proper_notears.py
==================
Runs full NOTEARS with proper exponential matrix acyclicity constraint.
Handles NaN/inf values automatically.

Usage:
    py -3.11 proper_notears.py --disease ad
    py -3.11 proper_notears.py --disease ms_blood
    py -3.11 proper_notears.py --disease tuberculosis
    py -3.11 proper_notears.py --disease hiv
    py -3.11 proper_notears.py --disease leishmaniasis2
    py -3.11 proper_notears.py --disease sarcoidosis2
    py -3.11 proper_notears.py --disease dengue
    py -3.11 proper_notears.py --disease malaria2
    py -3.11 proper_notears.py --disease breast_cancer

Run ALL diseases at once:
    py -3.11 proper_notears.py --all
"""

import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import expm
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--disease', type=str, default='ad')
parser.add_argument('--n_genes', type=int, default=20)
parser.add_argument('--lambda1', type=float, default=0.05)
parser.add_argument('--all', action='store_true', help='Run all diseases')
args = parser.parse_args()

data_dir = Path('data')
expr_dir = Path('expression')

ALL_DISEASES = [
    'ad', 'ms_blood', 'hiv', 'tuberculosis',
    'leishmaniasis2', 'sarcoidosis2', 'dengue', 'malaria2',
    'breast_cancer', 'leishmaniasis',
]

def find_expression_file(disease):
    """Find expression file for a disease, trying multiple name patterns."""
    candidates = [
        expr_dir / f'{disease}_annotated.csv',
        expr_dir / f'{disease}_real_expression.csv',
        expr_dir / f'{disease}_expression.csv',
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def clean_array(X):
    """Replace NaN/inf with column medians, then zeros."""
    X = X.copy().astype(float)
    for col in range(X.shape[1]):
        col_vals = X[:, col]
        finite_mask = np.isfinite(col_vals)
        if finite_mask.sum() > 0:
            median_val = np.median(col_vals[finite_mask])
            col_vals[~finite_mask] = median_val
        else:
            col_vals[:] = 0.0
        X[:, col] = col_vals
    return X

def h_acyclic(W):
    return np.trace(expm(W * W)) - W.shape[0]

def notears_proper(X, lambda1=0.05, max_iter=500, h_tol=1e-8):
    n, d = X.shape
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    W = np.zeros((d, d))
    rho, alpha, h_prev = 1.0, 0.0, np.inf

    for it in range(max_iter):
        def obj(wf):
            W_ = wf.reshape(d, d)
            np.fill_diagonal(W_, 0)
            loss = 0.5/n * np.sum((X - X@W_)**2)
            hval = h_acyclic(W_)
            pen  = lambda1 * np.sum(np.abs(W_))
            return loss + 0.5*rho*hval**2 + alpha*hval + pen

        res = minimize(obj, W.flatten(), method='L-BFGS-B',
                      options={'maxiter': 200, 'ftol': 1e-14})
        W = res.x.reshape(d, d)
        np.fill_diagonal(W, 0)
        hn = h_acyclic(W)
        n_edges = int(np.sum(np.abs(W) > 0.05))

        if it % 50 == 0:
            print(f"  iter {it:3d}: h={hn:.6f}  edges={n_edges}  loss={res.fun:.4f}")

        if hn > 0.25 * h_prev:
            rho = min(rho * 10, 1e12)
        alpha += rho * hn
        if hn <= h_tol:
            print(f"  Converged at iter={it}  h={hn:.2e}")
            break
        h_prev = hn

    return W

def run_disease(disease, n_genes=20, lambda1=0.05):
    print(f"\n{'='*60}")
    print(f"NOTEARS: {disease.upper()}")
    print(f"{'='*60}")

    csv_path = find_expression_file(disease)
    if csv_path is None:
        print(f"  No expression file found — skipping")
        return None

    print(f"  Loading: {csv_path}")
    df = pd.read_csv(csv_path, index_col=0)
    gene_cols = [c for c in df.columns if c != 'label']

    disease_arr = df[df['label']=='disease'][gene_cols].apply(
        pd.to_numeric, errors='coerce').fillna(0).values
    control_arr = df[df['label']=='control'][gene_cols].apply(
        pd.to_numeric, errors='coerce').fillna(0).values

    print(f"  Disease: {len(disease_arr)}  Control: {len(control_arr)}  Genes: {len(gene_cols)}")

    if len(disease_arr) < 5 or len(control_arr) < 5:
        print(f"  Too few samples — skipping")
        return None

    # Clean NaN/inf
    disease_arr = clean_array(disease_arr)
    control_arr = clean_array(control_arr)

    # Check for remaining NaN
    nan_count = np.sum(~np.isfinite(disease_arr))
    if nan_count > 0:
        print(f"  WARNING: {nan_count} non-finite values after cleaning — replacing with 0")
        disease_arr = np.nan_to_num(disease_arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Top N most variable genes
    var = disease_arr.var(0)
    top_idx = np.argsort(var)[::-1][:n_genes]
    top_genes = [gene_cols[i] for i in top_idx]
    X = disease_arr[:, top_idx]

    print(f"  Top {n_genes} genes: {top_genes[:3]}...")
    print(f"  Running NOTEARS ({len(disease_arr)} samples, {n_genes} genes, lambda={lambda1})...")

    W = notears_proper(X, lambda1=lambda1)
    n_edges = int(np.sum(np.abs(W) > 0.05))
    print(f"  SCM: {n_genes} nodes, {n_edges} directed edges")

    # Top edges
    if n_edges > 0:
        edges = []
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                if i != j and abs(W[i,j]) > 0.05:
                    edges.append((abs(W[i,j]), top_genes[i], top_genes[j], W[i,j]))
        edges.sort(reverse=True)
        print(f"  Top 5 edges:")
        for w, src, dst, raw_w in edges[:5]:
            print(f"    {src[:20]:20s} → {dst[:20]:20s}  w={raw_w:+.3f}")

    # Compute delta_v
    v_disease = disease_arr.mean(0)
    v_healthy = control_arr.mean(0)
    delta_v   = v_healthy - v_disease
    norm = np.linalg.norm(delta_v)
    delta_v_norm = delta_v / norm if norm > 0 else delta_v

    # Save
    np.save(data_dir/f'W_{disease}_proper.npy',    W)
    np.save(data_dir/f'W_{disease}.npy',            W)  # also overwrite main
    np.save(data_dir/f'delta_v_{disease}_proper.npy', delta_v_norm)
    np.save(data_dir/f'delta_v_{disease}.npy',     delta_v_norm)  # overwrite main
    np.save(data_dir/f'v_disease_{disease}.npy',   v_disease)
    np.save(data_dir/f'v_healthy_{disease}.npy',   v_healthy)

    with open(data_dir/f'{disease}_scm_genes.txt','w') as f:
        f.write('\n'.join(top_genes))

    print(f"  Saved: data/W_{disease}.npy  data/delta_v_{disease}.npy")
    return n_edges

# ── Run ───────────────────────────────────────────────────────────────────

if args.all:
    diseases = ALL_DISEASES
else:
    diseases = [args.disease]

results = {}
for disease in diseases:
    n_edges = run_disease(disease, n_genes=args.n_genes, lambda1=args.lambda1)
    if n_edges is not None:
        results[disease] = n_edges

print(f"\n{'='*60}")
print(f"NOTEARS COMPLETE")
print(f"{'='*60}")
for disease, n_edges in results.items():
    print(f"  {disease:20s}: {n_edges:3d} edges")
print(f"\nAll SCM files saved to data/")
print(f"delta_v files updated — models will use real causal conditioning on next training run")
