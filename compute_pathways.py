"""
compute_pathways.py
====================
For each disease, takes the top drug from each library and computes:
1. Which SCM nodes (genes) it perturbs most strongly (delta_T per node)
2. Which causal edges are most activated
3. Direction: toward healthy (+) or away (-)

Outputs:
  results/{disease}_pathway_analysis.csv
  results/pathway_summary_all_diseases.csv

Usage:
    py -3.11 compute_pathways.py
"""

import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, QED, AllChem
from rdkit.Chem.rdMolDescriptors import CalcTPSA
import joblib

RDLogger.DisableLog('rdApp.*')

data_dir    = Path('data')
results_dir = Path('results')
results_dir.mkdir(exist_ok=True)

ALL_DISEASES = [
    'ms_blood','hiv','ad','tuberculosis','dengue',
    'sarcoidosis2','leishmaniasis2','malaria2',
    'breast_cancer','breast_cancer2',
]

DISEASE_ABBREV = {
    'ms_blood':'MS','hiv':'HIV','ad':'AD','tuberculosis':'TB',
    'dengue':'DEN','sarcoidosis2':'SAR','leishmaniasis2':'LEI',
    'malaria2':'MAL','breast_cancer':'BC1','breast_cancer2':'BC2',
}

# ── BBB model ──────────────────────────────────────────────────────────────
bbb_bundle = joblib.load('logbb_rf.pkl') if Path('logbb_rf.pkl').exists() else None

def predict_bbb_simple(mol):
    logp = Descriptors.MolLogP(mol)
    mw   = Descriptors.MolWt(mol)
    tpsa = CalcTPSA(mol)
    from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
    rotb = CalcNumRotatableBonds(mol)
    hbd  = Descriptors.NumHDonors(mol)
    return (1<=logp<=4 and mw<450 and tpsa<90 and rotb<8 and hbd<=3)

# ── Compute delta_T and node perturbations ─────────────────────────────────

def compute_delta_T(mol, W, v_dis, v_hea):
    """
    Returns delta_T vector and per-node perturbation magnitudes.
    """
    d = W.shape[0]
    fp_d    = np.array(AllChem.GetMorganFingerprintAsBitVect(mol,2,d), dtype=float)
    fp_512  = np.array(AllChem.GetMorganFingerprintAsBitVect(mol,3,512), dtype=float)
    chunk   = 512//d
    fp_fold = np.array([fp_512[i*chunk:(i+1)*chunk].mean() for i in range(d)])
    fp_sig  = 0.6*fp_d + 0.4*fp_fold

    logp = Descriptors.MolLogP(mol)
    tpsa = CalcTPSA(mol)
    qed  = QED.qed(mol)
    drug_scale = qed*0.5 + np.clip((logp-1)/3,0,1)*0.3 + (1-tpsa/140)*0.2

    expr_diff = v_hea[:d] - v_dis[:d]
    fp_signed = fp_sig*2 - 1
    delta_T   = fp_signed*np.abs(expr_diff)*drug_scale + expr_diff*drug_scale*0.2

    try:
        M_inv = np.linalg.solve(np.eye(d)-W+1e-6*np.eye(d), np.eye(d))
    except:
        M_inv = np.eye(d)

    v_int = M_inv @ (v_dis[:d] + delta_T)

    # Per-node: how much does v_int shift toward v_healthy?
    node_shift = v_hea[:d] - v_int           # positive = needs more shift
    node_correction = v_int - v_dis[:d]      # positive = drug pushed in right direction

    return delta_T, node_correction, v_int

def get_gene_names(disease):
    """Load gene names from SCM genes file."""
    f = data_dir/f'{disease}_scm_genes.txt'
    if f.exists():
        genes = open(f).read().strip().split('\n')
        # Clean up sex confounder labels
        cleaned = []
        for g in genes:
            if g.startswith('SEX_CONFOUNDER_'):
                cleaned.append(g.replace('SEX_CONFOUNDER_','')+'*')
            else:
                cleaned.append(g)
        return cleaned
    return [f'Gene_{i}' for i in range(20)]

def get_top_edges(W, gene_names, n=5):
    """Get top causal edges by absolute weight."""
    edges = []
    d = W.shape[0]
    for i in range(d):
        for j in range(d):
            if abs(W[i,j]) > 0.05:
                edges.append((gene_names[i], gene_names[j],
                               round(W[i,j],3)))
    edges.sort(key=lambda x: abs(x[2]), reverse=True)
    return edges[:n]

def analyze_drug_pathway(drug_name, smiles, W, v_dis, v_hea, gene_names):
    """
    Compute which SCM nodes a drug most strongly perturbs.
    Returns a dict with top perturbed nodes and top targeted edges.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        delta_T, node_correction, v_int = compute_delta_T(mol, W, v_dis, v_hea)
        d = W.shape[0]

        # Top nodes by absolute perturbation magnitude
        perturbations = list(zip(gene_names[:d],
                                  delta_T[:d],
                                  node_correction[:d]))
        perturbations.sort(key=lambda x: abs(x[1]), reverse=True)

        # Direction: positive correction means drug pushes toward healthy
        top_nodes = []
        for gene, dT, corr in perturbations[:5]:
            direction = '↑healthy' if corr > 0 else '↓away'
            top_nodes.append(f"{gene}({direction}, dT={dT:.3f})")

        # Which edges does this drug most activate?
        # Edge activation = weight × delta_T of source node
        edge_activations = []
        for i in range(d):
            for j in range(d):
                if abs(W[i,j]) > 0.05:
                    activation = W[i,j] * delta_T[i]
                    edge_activations.append((
                        gene_names[i], gene_names[j],
                        round(W[i,j],3),
                        round(float(activation),4)
                    ))
        edge_activations.sort(key=lambda x: abs(x[3]), reverse=True)

        top_edges = []
        for src, tgt, w, act in edge_activations[:3]:
            direction = '→healthy' if act > 0 else '→disease'
            top_edges.append(f"{src}→{tgt}(w={w},{direction})")

        # R score
        R = float(np.clip(
            np.dot(v_int, v_hea[:d]) /
            (np.linalg.norm(v_int)*np.linalg.norm(v_hea[:d])+1e-8), -1, 1))

        return {
            'R_score': round(R, 4),
            'top_perturbed_nodes': '; '.join(top_nodes[:3]),
            'top_activated_edges': '; '.join(top_edges[:2]),
            'net_correction':      round(float(np.mean(node_correction[:d])), 4),
        }
    except Exception as e:
        return None

# ── Main ───────────────────────────────────────────────────────────────────

print("="*65)
print("PATHWAY ANALYSIS — TOP DRUGS PER DISEASE PER LIBRARY")
print("="*65)

all_rows = []

for disease in ALL_DISEASES:
    if not (data_dir/f'W_{disease}.npy').exists():
        print(f"\nNo SCM for {disease} — skipping")
        continue

    W      = np.load(data_dir/f'W_{disease}.npy')
    v_dis  = np.load(data_dir/f'v_disease_{disease}.npy')
    v_hea  = np.load(data_dir/f'v_healthy_{disease}.npy')
    genes  = get_gene_names(disease)
    abbrev = DISEASE_ABBREV.get(disease, disease[:3].upper())

    print(f"\n{'='*55}")
    print(f"DISEASE: {disease.upper()}")
    print(f"Top SCM edges:")
    for src,tgt,w in get_top_edges(W, genes, 5):
        print(f"  {src:20s} → {tgt:20s} w={w:+.3f}")

    modes = [
        ('approveddrugs','FDA-Approved'),
        ('clinical','Clinical'),
        ('preclinical','Preclinical'),
        ('zinc','ZINC'),
    ]

    for mode, label in modes:
        f = results_dir/f'{disease}_{mode}_top20.csv'
        if not f.exists():
            continue
        df = pd.read_csv(f)

        # Filter: for top drug, use first valid small molecule
        name_col = 'name' if 'name' in df.columns else 'candidate_id'
        smiles_col = 'smiles' if 'smiles' in df.columns else None
        if smiles_col is None:
            continue

        # Get top 3 drugs for pathway analysis
        for _, row in df.head(3).iterrows():
            smi  = str(row['smiles'])
            name = str(row[name_col])
            R    = row['R_score']

            result = analyze_drug_pathway(name, smi, W, v_dis, v_hea, genes)
            if result is None:
                continue

            print(f"\n  [{label}] {name[:30]:30s} R={R:.4f}")
            print(f"    Top nodes: {result['top_perturbed_nodes']}")
            print(f"    Top edges: {result['top_activated_edges']}")

            all_rows.append({
                'disease':       disease,
                'library':       label,
                'drug_name':     name,
                'R_score':       R,
                'composite_score': row.get('score', R),
                'top_perturbed_nodes': result['top_perturbed_nodes'],
                'top_activated_edges': result['top_activated_edges'],
                'net_correction': result['net_correction'],
                'scm_top_genes': ', '.join(genes[:5]),
            })

# Save
df_out = pd.DataFrame(all_rows)
df_out.to_csv(results_dir/'pathway_analysis_all.csv', index=False)
print(f"\n\nSaved: results/pathway_analysis_all.csv ({len(df_out)} rows)")

# Print summary table
print(f"\n{'='*75}")
print("SUMMARY: TOP DRUG PER DISEASE PER LIBRARY WITH PATHWAY")
print(f"{'='*75}")

for disease in ALL_DISEASES:
    sub = df_out[df_out['disease']==disease]
    if sub.empty: continue
    print(f"\n{disease.upper()}:")
    for lib in ['FDA-Approved','Clinical','Preclinical','ZINC']:
        rows = sub[sub['library']==lib]
        if rows.empty: continue
        r = rows.iloc[0]
        print(f"  {lib:12s} {r['drug_name'][:28]:28s} R={r['R_score']:.4f}")
        print(f"             Nodes: {r['top_perturbed_nodes'][:70]}")
        print(f"             Edges: {r['top_activated_edges'][:70]}")
