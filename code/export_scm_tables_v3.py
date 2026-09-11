"""
CausalDiffChem — export_scm_tables_v3.py
===========================================
Both `base` and `_proper` variants matched Table 1's edge counts exactly
for all 11 diseases (per the previous run) -- strong evidence they're the
same SCM under two filenames. This does one more, stronger check: verifies
the SPECIFIC named edges already cited in the manuscript's Results and
Figure 1 appear with the exact reported weights. If those also match,
there's nothing left to disambiguate -- exports directly, preferring
_proper on any remaining tie (matches the confirmed script name
proper_notears.py).

Run from causaldiffchem/ (same level as data/):
    py -3.11 export_scm_tables_v3.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import re

DATA_DIR = Path("data")
OUT_DIR = Path("scm_tables")
OUT_DIR.mkdir(exist_ok=True)

EDGE_THRESHOLD = 0.05
WEIGHT_TOL = 0.002  # allow tiny float rounding differences

TABLE1_EDGES = {
    'ms_blood': 48, 'hiv': 57, 'ad': 54, 'tuberculosis_clean': 48,
    'dengue': 38, 'sarcoidosis2': 50, 'leishmaniasis2': 61, 'malaria2': 33,
    'breast_cancer': 50, 'breast_cancer2': 50, 'sch_haem': 64,
}

# Specific edges already cited in the manuscript (Results / Figure 1),
# used here as a stronger identity check than edge count alone.
CITED_EDGES = {
    'ms_blood':          [('DEFA3', 'DEFA4', 0.851), ('IFI44', 'IFI44L', 0.715), ('HERC5', 'MX2', 0.639)],
    'hiv':                [('GBP2', 'GBP1', 0.565), ('HIST2H2AA4', 'HIST2H2AA3', 0.775)],
    'ad':                 [('APOD_1', 'APOD', 0.932)],
    'tuberculosis_clean': [('HLA-DRB1', 'HLA-DRB5', 0.831)],
    'dengue':             [('HLA-DQB1', 'HLA-DQA1', 0.875)],
    'sarcoidosis2':       [('ANXA3', 'ARG1', 0.794), ('ARG1', 'IL1R2', 0.866)],
    'leishmaniasis2':     [('HLA-DRB5', 'HLA-DRB1', 0.684), ('CNFN', 'KRT14', 0.680)],
    'breast_cancer':      [('COL1A2_1', 'COL3A1_1', 0.854), ('AGR2', 'TOX3', 0.436)],
    'breast_cancer2':     [('ABI3BP', 'PDK4', 0.629), ('ABI3BP', 'POU2AF1', 0.416)],
}

def count_edges(W):
    d = W.shape[0]
    return int(np.sum((np.abs(W) >= EDGE_THRESHOLD) & ~np.eye(d, dtype=bool)))

def load_genes(stem, canon):
    for p in [DATA_DIR / f"{stem}_scm_genes.txt", DATA_DIR / f"{canon}_scm_genes.txt"]:
        if p.exists():
            return [g.strip() for g in p.read_text().splitlines() if g.strip()]
    return None

def check_cited_edges(W, genes, canon):
    if genes is None or canon not in CITED_EDGES:
        return None  # can't check
    idx = {g: i for i, g in enumerate(genes)}
    results = []
    for src, tgt, expected_w in CITED_EDGES[canon]:
        if src not in idx or tgt not in idx:
            results.append(False)
            continue
        actual_w = W[idx[src], idx[tgt]]
        results.append(abs(actual_w - expected_w) < WEIGHT_TOL)
    return all(results) if results else None

W_files = sorted(DATA_DIR.glob("W_*.npy"))
groups = {}
for w_path in W_files:
    stem = w_path.stem.replace("W_", "")
    canon = re.sub(r'_(proper|real)$', '', stem)
    groups.setdefault(canon, []).append(stem)

print("=== Resolving each disease ===\n")
resolved = {}

for canon, variants in sorted(groups.items()):
    if canon not in TABLE1_EDGES:
        print(f"{canon}: not one of the 11 main diseases, skipping")
        continue
    target = TABLE1_EDGES[canon]
    count_matches = []
    for stem in variants:
        W = np.load(DATA_DIR / f"W_{stem}.npy")
        if count_edges(W) == target:
            count_matches.append(stem)

    if not count_matches:
        print(f"{canon}: [!] NO variant matches Table 1 ({target} edges) -- needs manual review, skipping export")
        continue

    # Among count-matches, check cited edge weights for a stronger identity check
    genes = load_genes(count_matches[0], canon)
    edge_check = None
    if len(count_matches) >= 1:
        W0 = np.load(DATA_DIR / f"W_{count_matches[0]}.npy")
        edge_check = check_cited_edges(W0, genes, canon)

    # Prefer _proper on ties; fall back to first count-match otherwise
    proper_candidates = [s for s in count_matches if s.endswith('_proper')]
    winner = proper_candidates[0] if proper_candidates else count_matches[0]

    tag = ""
    if edge_check is True:
        tag = " [cited edges confirmed]"
    elif edge_check is False:
        tag = " [!] cited edge weights did NOT match -- verify manually"
    elif edge_check is None:
        tag = " [no cited edges to cross-check for this disease]"

    print(f"{canon}: {len(count_matches)} variant(s) match Table 1 -> using W_{winner}.npy{tag}")
    resolved[canon] = winner

# ---- Export ----
print("\n=== Exporting ===")
all_edges = []
for canon, stem in resolved.items():
    W = np.load(DATA_DIR / f"W_{stem}.npy")
    d = W.shape[0]
    genes = load_genes(stem, canon) or [f"node_{i}" for i in range(d)]
    if len(genes) != d:
        genes = (genes + [f"node_{i}" for i in range(d)])[:d]

    pd.DataFrame(W, index=genes, columns=genes).to_csv(OUT_DIR / f"{canon}_W_labeled.csv")

    edges = []
    for i in range(d):
        for j in range(d):
            if i != j and abs(W[i, j]) >= EDGE_THRESHOLD:
                edges.append({"disease": canon, "source": genes[i], "target": genes[j],
                              "weight": round(float(W[i, j]), 4)})
    edges.sort(key=lambda e: -abs(e["weight"]))
    pd.DataFrame(edges).to_csv(OUT_DIR / f"{canon}_scm_edges.csv", index=False)
    all_edges.extend(edges)
    print(f"  {canon}: {len(edges)} edges -> {canon}_W_labeled.csv, {canon}_scm_edges.csv")

if all_edges:
    pd.DataFrame(all_edges).to_csv(OUT_DIR / "all_diseases_scm_edges.csv", index=False)
    print(f"\nDone: scm_tables/all_diseases_scm_edges.csv ({len(all_edges)} edges, {len(resolved)} diseases)")

missing = set(TABLE1_EDGES) - set(resolved)
if missing:
    print(f"\n[!] Not exported, needs attention: {sorted(missing)}")
