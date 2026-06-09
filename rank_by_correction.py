"""
rank_by_correction.py
======================
Ranks drugs by how much they push the disease expression network
toward the healthy state — the true objective of CausalDiffChem.

Metrics computed per drug:
  R_score       : cosine similarity of v_int to v_healthy (existing)
  net_correction: mean positive node shift toward healthy / total disease gap
                  = mean(max(v_int - v_dis, 0)) / mean(|v_hea - v_dis|)
  pct_nodes_H   : percentage of SCM nodes pushed toward healthy
  mean_node_H   : mean normalised shift toward healthy across all nodes

Saves:
  results/{disease}_{lib}_top20_corrected.csv  (ranked by net_correction)
  results/correction_summary_all.csv

Usage:
    py -3.11 rank_by_correction.py
"""

import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, QED, AllChem
from rdkit.Chem.rdMolDescriptors import CalcTPSA, CalcNumRotatableBonds
import joblib, json
from collections import defaultdict

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

LIBRARY_LABELS = {
    'approved_drugs':        'FDA-Approved',
    'clinical_compounds':    'Clinical',
    'preclinical_compounds': 'Preclinical',
}

# ── BBB model ──────────────────────────────────────────────────────────────
bbb_bundle = joblib.load('logbb_rf.pkl') if Path('logbb_rf.pkl').exists() else None

def predict_bbb(mol):
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = CalcTPSA(mol)
    rotb = CalcNumRotatableBonds(mol)
    hbd  = Descriptors.NumHDonors(mol)
    proxy = (1<=logp<=4 and mw<450 and tpsa<90 and rotb<8 and hbd<=3)
    if bbb_bundle is None:
        return bool(proxy), round(logp*0.3-1.2,3)
    try:
        fp    = AllChem.GetMorganFingerprintAsBitVect(mol,2,64)
        desc  = np.array([mw,logp,tpsa,hbd,
                          Descriptors.NumHAcceptors(mol),rotb,
                          Descriptors.RingCount(mol),
                          Descriptors.FractionCSP3(mol),
                          Descriptors.NumAromaticRings(mol),
                          Descriptors.NumHeteroatoms(mol)])
        feats = np.concatenate([desc,np.array(fp)]).reshape(1,-1)
        fc    = bbb_bundle['clf_scaler'].transform(feats)
        prob  = float(bbb_bundle['classifier'].predict_proba(fc)[0][1])
        fr    = bbb_bundle['reg_scaler'].transform(feats)
        logbb = float(bbb_bundle['regressor'].predict(fr)[0])
        return prob>0.5, round(logbb,3)
    except:
        return bool(proxy), round(logp*0.3-1.2,3)

# ── Core: compute full network correction ──────────────────────────────────

def compute_network_correction(mol, W, v_dis, v_hea, gene_names):
    """
    Computes how much the drug pushes the ENTIRE disease network toward healthy.

    Returns:
      R_score       : cosine similarity (existing metric)
      net_correction: mean positive shift / total gap (0-1 scale)
      pct_nodes_H   : % of nodes pushed toward healthy
      mean_node_H   : mean normalised positive shift across all nodes
      top_nodes_H   : top 3 nodes pushed toward healthy
      top_nodes_D   : top 3 nodes pushed away from healthy (off-target)
      top_edge      : most activated causal edge
    """
    d = W.shape[0]

    # Fingerprint → perturbation
    fp_d    = np.array(AllChem.GetMorganFingerprintAsBitVect(mol,2,d),dtype=float)
    fp_512  = np.array(AllChem.GetMorganFingerprintAsBitVect(mol,3,512),dtype=float)
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
        M_inv = np.linalg.solve(np.eye(d)-W+1e-6*np.eye(d),np.eye(d))
    except:
        M_inv = np.eye(d)

    v_int = M_inv @ (v_dis[:d] + delta_T)

    # R score
    R = float(np.clip(
        np.dot(v_int,v_hea[:d]) /
        (np.linalg.norm(v_int)*np.linalg.norm(v_hea[:d])+1e-8),-1,1))

    # Per-node correction toward healthy
    # positive = drug moved this node closer to healthy
    # negative = drug moved this node away from healthy
    node_correction = v_int - v_dis[:d]  # how much drug moved each node
    disease_gap     = v_hea[:d] - v_dis[:d]  # full gap from disease to healthy

    # Normalise by mean absolute disease gap
    gap_scale = max(np.abs(disease_gap).mean(), 1e-8)
    node_corr_norm = node_correction / gap_scale

    # Net correction: average positive correction as fraction of disease gap
    positive_corrections = np.maximum(node_corr_norm, 0)
    net_correction = float(positive_corrections.mean())

    # % of nodes pushed toward healthy
    # A node is "toward healthy" if correction aligns with disease gap
    aligned = np.sign(node_correction) == np.sign(disease_gap)
    pct_nodes_H = float(aligned.mean()) * 100

    # Mean normalised shift toward healthy
    mean_node_H = float(node_corr_norm[aligned].mean()) if aligned.any() else 0.0

    # Top 3 nodes pushed toward healthy
    corr_signed = [(gene_names[i], node_corr_norm[i])
                   for i in range(d) if aligned[i]]
    corr_signed.sort(key=lambda x: abs(x[1]), reverse=True)
    top_H = '; '.join(f"{g}(+{abs(v):.3f})" for g,v in corr_signed[:3])

    # Top 3 nodes pushed away from healthy (off-target)
    corr_away = [(gene_names[i], node_corr_norm[i])
                 for i in range(d) if not aligned[i]]
    corr_away.sort(key=lambda x: abs(x[1]), reverse=True)
    top_D = '; '.join(f"{g}(−{abs(v):.3f})" for g,v in corr_away[:3])

    # Top activated causal edge
    delta_T_norm = delta_T / gap_scale
    edge_acts = []
    for i in range(d):
        for j in range(d):
            if abs(W[i,j]) > 0.05:
                act = float(W[i,j]*delta_T_norm[i])
                edge_acts.append((gene_names[i],gene_names[j],
                                  round(float(W[i,j]),2),round(act,4)))
    edge_acts.sort(key=lambda x: abs(x[3]),reverse=True)
    if edge_acts:
        src,tgt,w,act = edge_acts[0]
        direction = '→H' if act>0 else '→D'
        top_edge = f"{src}→{tgt}(w={w:+.2f},{direction})"
    else:
        top_edge = ''

    # OTCI
    otci = float(np.mean(np.abs(v_int-v_hea[:d])) /
                 (np.abs(v_hea[:d]).mean()+1e-8))

    return {
        'R_score':        round(R,4),
        'net_correction': round(net_correction,4),
        'pct_nodes_H':    round(pct_nodes_H,1),
        'mean_node_H':    round(mean_node_H,4),
        'OTCI':           round(otci,4),
        'nodes_toward_H': top_H,
        'nodes_away_H':   top_D,
        'top_edge':       top_edge,
    }

# ── Load gene names ────────────────────────────────────────────────────────

def load_gene_names(disease):
    f = data_dir/f'{disease}_scm_genes.txt'
    if not f.exists():
        return [f'Gene_{i}' for i in range(20)]
    genes = open(f).read().strip().split('\n')
    return [g.replace('SEX_CONFOUNDER_','')+'*'
            if g.startswith('SEX_CONFOUNDER_') else g
            for g in genes]

# ── Main ───────────────────────────────────────────────────────────────────

print("Loading drug library...")
drug_index = json.load(open('drug_library_index.json'))
drug_mols  = {}
for entry in drug_index:
    smi = entry['smiles']
    if smi not in drug_mols:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            drug_mols[smi] = (mol, entry)

print(f"Loaded {len(drug_mols)} unique molecules")

all_summary = []

for disease in ALL_DISEASES:
    if not (data_dir/f'W_{disease}.npy').exists():
        print(f"\nSkipping {disease} — no SCM"); continue

    W          = np.load(data_dir/f'W_{disease}.npy')
    v_dis      = np.load(data_dir/f'v_disease_{disease}.npy')
    v_hea      = np.load(data_dir/f'v_healthy_{disease}.npy')
    gene_names = load_gene_names(disease)
    abbrev     = DISEASE_ABBREV.get(disease,'???')

    print(f"\n{'='*60}")
    print(f"DISEASE: {disease.upper()}")
    print(f"{'='*60}")

    for lib_key, label in [
        ('approveddrugs','FDA-Approved'),
        ('clinical','Clinical'),
        ('preclinical','Preclinical'),
        ('zinc','ZINC'),
    ]:
        # Load existing top20 CSV to get the drug list
        f = results_dir/f'{disease}_{lib_key}_top20.csv'
        if not f.exists():
            print(f"  [{label}] missing"); continue

        df_existing = pd.read_csv(f)
        name_col = 'name' if 'name' in df_existing.columns else 'candidate_id'

        rows = []
        for _, row in df_existing.iterrows():
            smi  = str(row['smiles'])
            name = str(row[name_col])
            mol  = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                metrics = compute_network_correction(
                    mol, W, v_dis, v_hea, gene_names)
                bbb_pass, logbb = predict_bbb(mol)
                mw  = round(Descriptors.MolWt(mol),1)
                qed = round(QED.qed(mol),3)
                rows.append({
                    'rank_by_correction': 0,  # filled after sort
                    'name':           name,
                    'library':        label,
                    'disease':        disease,
                    'mw':             mw,
                    'qed':            qed,
                    'bbb_pass':       bbb_pass,
                    'logbb':          logbb,
                    **metrics,
                })
            except:
                continue

        if not rows:
            continue

        # Sort by net_correction descending
        df_out = pd.DataFrame(rows).sort_values(
            'net_correction', ascending=False).reset_index(drop=True)
        df_out['rank_by_correction'] = range(1, len(df_out)+1)

        out_path = results_dir/f'{disease}_{lib_key}_top20_corrected.csv'
        df_out.to_csv(out_path, index=False)

        # Print top 10
        print(f"\n  ── {label} — {disease.upper()} ──")
        print(f"  Ranked by net_correction (mean positive node shift toward healthy)")
        print(f"  {'Rank':4s} {'Name':28s} {'R':7s} {'NetCorr':8s} "
              f"{'%NodH':6s} {'BBB':4s} {'Nodes toward healthy (top 3)'}")
        print(f"  {'-'*95}")
        for _, r in df_out.head(10).iterrows():
            b   = '+' if r['bbb_pass'] else '-'
            nh  = str(r['nodes_toward_H'])[:35]
            print(f"  {int(r['rank_by_correction']):4d} "
                  f"{r['name'][:28]:28s} "
                  f"{r['R_score']:7.4f} "
                  f"{r['net_correction']:8.4f} "
                  f"{r['pct_nodes_H']:6.1f}% "
                  f"{b:>3}  {nh}")

        # Add top entry to summary
        top = df_out.iloc[0]
        all_summary.append({
            'disease':          disease,
            'library':          label,
            'top_drug':         top['name'],
            'top_R':            top['R_score'],
            'top_net_corr':     top['net_correction'],
            'top_pct_nodes_H':  top['pct_nodes_H'],
            'top_nodes_H':      top['nodes_toward_H'],
            'top_edge':         top['top_edge'],
        })

# Summary
pd.DataFrame(all_summary).to_csv(
    results_dir/'correction_summary_all.csv', index=False)

print(f"\n{'='*70}")
print("CROSS-DISEASE SUMMARY — RANKED BY NETWORK CORRECTION")
print(f"{'='*70}")
print(f"{'Disease':15s} {'Library':12s} {'Top drug':28s} "
      f"{'R':7s} {'NetCorr':8s} {'%H':6s}")
print(f"{'-'*85}")
for r in all_summary:
    print(f"{r['disease']:15s} {r['library']:12s} "
          f"{r['top_drug'][:28]:28s} "
          f"{r['top_R']:7.4f} "
          f"{r['top_net_corr']:8.4f} "
          f"{r['top_pct_nodes_H']:5.1f}%")

print(f"\nSaved corrected rankings to results/{{disease}}_{{lib}}_top20_corrected.csv")
