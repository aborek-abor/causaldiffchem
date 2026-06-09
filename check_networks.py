"""
check_networks.py
Prints the top causal edges per disease from the NOTEARS W matrices.
Run: py -3.11 check_networks.py
"""
import numpy as np
from pathlib import Path

ALL_DISEASES = [
    'ms_blood','hiv','ad','tuberculosis','dengue',
    'sarcoidosis2','leishmaniasis2','malaria2',
    'breast_cancer','breast_cancer2',
]

data_dir = Path('data')

for disease in ALL_DISEASES:
    W_f    = data_dir/f'W_{disease}.npy'
    gene_f = data_dir/f'{disease}_scm_genes.txt'

    if not W_f.exists() or not gene_f.exists():
        print(f"\n{disease.upper()}: missing files"); continue

    W     = np.load(W_f)
    genes = open(gene_f).read().strip().split('\n')
    genes = [g.replace('SEX_CONFOUNDER_','')+'*'
             if g.startswith('SEX_CONFOUNDER_') else g
             for g in genes]

    d = W.shape[0]
    n_edges = int(np.sum(np.abs(W) > 0.05))

    # Collect all edges above threshold
    edges = []
    for i in range(d):
        for j in range(d):
            if abs(W[i,j]) > 0.05:
                edges.append((genes[i], genes[j], round(float(W[i,j]),3)))

    # Sort by absolute weight
    edges.sort(key=lambda x: abs(x[2]), reverse=True)

    print(f"\n{'='*65}")
    print(f"DISEASE: {disease.upper()}")
    print(f"SCM: {d} nodes, {n_edges} edges (|w| > 0.05)")
    print(f"{'='*65}")
    print(f"{'Source':22s} {'Target':22s} {'Weight':8s}  Type")
    print(f"{'-'*65}")

    for src, tgt, w in edges[:15]:
        etype = 'POSITIVE (+)' if w > 0 else 'NEGATIVE (-)'
        print(f"{src:22s} {tgt:22s} {w:+8.3f}  {etype}")

    print(f"\n  Total edges: {n_edges}")
    print(f"  Positive: {sum(1 for _,_,w in edges if w>0.05)}")
    print(f"  Negative: {sum(1 for _,_,w in edges if w<-0.05)}")
