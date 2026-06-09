"""
score_by_correction.py
=======================
Scores the ENTIRE drug library ranked purely by net_correction —
how much each drug pushes the disease expression network toward healthy.

net_correction = mean(positive node shifts toward healthy) / mean(|disease gap|)
pct_nodes_H    = % of SCM nodes pushed toward healthy direction
R_score        = cosine similarity (kept for reference)

This is the biologically correct primary ranking — it directly measures
what fraction of the disease network each drug corrects toward healthy.

Saves:
  results/{disease}_{lib}_correction_top20.csv  (top 20 by net_correction)
  results/{disease}_{lib}_correction_all.csv    (full ranked library)
  results/correction_summary_all_diseases.csv

Usage:
    py -3.11 score_by_correction.py --disease ms_blood --mode all
    py -3.11 score_by_correction.py --all --mode all --n_zinc 1000
"""

import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, QED, AllChem
from rdkit.Chem.rdMolDescriptors import CalcTPSA, CalcNumRotatableBonds
import joblib, argparse, json, random
from collections import defaultdict

RDLogger.DisableLog('rdApp.*')

parser = argparse.ArgumentParser()
parser.add_argument('--disease', type=str, default='ms_blood')
parser.add_argument('--all',     action='store_true')
parser.add_argument('--mode',    type=str, default='all',
                    choices=['fda','clinical','preclinical','zinc','all'])
parser.add_argument('--top_k',   type=int, default=20)
parser.add_argument('--n_zinc',  type=int, default=1000)
args = parser.parse_args()

data_dir    = Path('data')
graphs_dir  = Path('graphs')
results_dir = Path('results')
results_dir.mkdir(exist_ok=True)

ALL_DISEASES = [
    'ms_blood','hiv','ad','tuberculosis','dengue',
    'sarcoidosis2','leishmaniasis2','malaria2',
    'breast_cancer','breast_cancer2',
]

NEURO_DISEASES = {'ms_blood','hiv','ad'}

DISEASE_ABBREV = {
    'ms_blood':'MS','hiv':'HIV','ad':'AD','tuberculosis':'TB',
    'dengue':'DEN','sarcoidosis2':'SAR','leishmaniasis2':'LEI',
    'malaria2':'MAL','breast_cancer':'BC1','breast_cancer2':'BC2',
}

DISEASE_TARGETS = {
    'ms_blood':       'MX1, IFITM3, DEFA1',
    'hiv':            'LY6E, MX1, DEFA1',
    'ad':             'HLA-DQB1, HLA-DQA1, HLA-DPB1',
    'tuberculosis':   'HLA-DRB5, EIF1AY, HLA-DRB1',
    'dengue':         'HLA-DQB1, IGHM, MZB1',
    'sarcoidosis2':   'HLA-DRB5, TNFRSF10C, CXCL8',
    'leishmaniasis2': 'GNLY, GZMB, HLA-DRB1',
    'malaria2':       'PfDHFR, PfCRT, PfATP4',
    'breast_cancer':  'SCGB2A2, AGR3, ESR1',
    'breast_cancer2': 'FABP7, GABRP, PROM1',
}

LIBRARY_LABELS = {
    'approved_drugs':        'FDA-Approved',
    'clinical_compounds':    'Clinical',
    'preclinical_compounds': 'Preclinical',
}

MODE_TO_LIBS = {
    'fda':         ['approved_drugs'],
    'clinical':    ['clinical_compounds'],
    'preclinical': ['preclinical_compounds'],
    'zinc':        [],
    'all':         ['approved_drugs','clinical_compounds','preclinical_compounds'],
}

# ── BBB model ──────────────────────────────────────────────────────────────
bbb_bundle = joblib.load('logbb_rf.pkl') if Path('logbb_rf.pkl').exists() else None
if bbb_bundle:
    print(f"BBB model loaded ({bbb_bundle['n_train_cls']} training samples)")

def predict_bbb(mol):
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = CalcTPSA(mol)
    rotb = CalcNumRotatableBonds(mol)
    hbd  = Descriptors.NumHDonors(mol)
    proxy = (1<=logp<=4 and mw<450 and tpsa<90 and rotb<8 and hbd<=3)
    if bbb_bundle is None:
        return bool(proxy), round(logp*0.3-1.2,3), float(proxy)
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
        return prob>0.5, round(logbb,3), round(prob,3)
    except:
        return bool(proxy), round(logp*0.3-1.2,3), float(proxy)

def mol_props(mol):
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = CalcTPSA(mol)
    hbd  = Descriptors.NumHDonors(mol)
    hba  = Descriptors.NumHAcceptors(mol)
    rotb = CalcNumRotatableBonds(mol)
    qed  = QED.qed(mol)
    ro5  = mw<=500 and logp<=5 and hbd<=5 and hba<=10
    bbb_pass, logbb, bbb_prob = predict_bbb(mol)
    return dict(mw=round(mw,1), logp=round(logp,3), tpsa=round(tpsa,1),
                hbd=int(hbd), hba=int(hba), rotb=int(rotb),
                qed=round(qed,3), ro5=bool(ro5),
                bbb_pass=bbb_pass, logbb=logbb, bbb_prob=round(bbb_prob,3))

# ── Core: full network correction ─────────────────────────────────────────

def score_network_correction(mol, W, v_dis, v_hea, gene_names):
    """
    Primary metric: net_correction
      = mean(positive node corrections toward healthy) / mean(|disease gap|)

    This directly measures what fraction of the disease-to-healthy
    expression gap the drug closes across ALL SCM nodes.

    Also returns R_score for reference and full pathway annotation.
    """
    d = W.shape[0]

    # Fingerprint → perturbation vector
    fp_d    = np.array(AllChem.GetMorganFingerprintAsBitVect(mol,2,d),dtype=float)
    fp_512  = np.array(AllChem.GetMorganFingerprintAsBitVect(mol,3,512),dtype=float)
    chunk   = 512//d
    fp_fold = np.array([fp_512[i*chunk:(i+1)*chunk].mean() for i in range(d)])
    fp_sig  = 0.6*fp_d + 0.4*fp_fold

    logp = Descriptors.MolLogP(mol)
    tpsa = CalcTPSA(mol)
    qed  = QED.qed(mol)
    drug_scale = qed*0.5 + np.clip((logp-1)/3,0,1)*0.3 + (1-tpsa/140)*0.2

    # Disease-to-healthy gap
    disease_gap = v_hea[:d] - v_dis[:d]
    fp_signed   = fp_sig*2 - 1
    delta_T     = fp_signed*np.abs(disease_gap)*drug_scale + disease_gap*drug_scale*0.2

    # Do-calculus: propagate through causal graph
    try:
        M_inv = np.linalg.solve(np.eye(d)-W+1e-6*np.eye(d), np.eye(d))
    except:
        M_inv = np.eye(d)

    v_int = M_inv @ (v_dis[:d] + delta_T)

    # ── R score (cosine similarity, kept for reference) ────────────────
    R = float(np.clip(
        np.dot(v_int, v_hea[:d]) /
        (np.linalg.norm(v_int)*np.linalg.norm(v_hea[:d])+1e-8), -1, 1))

    # OTCI
    otci = float(np.mean(np.abs(v_int-v_hea[:d])) /
                 (np.abs(v_hea[:d]).mean()+1e-8))

    # ── Net correction (PRIMARY METRIC) ───────────────────────────────
    # How much did the drug move each node toward healthy?
    node_movement = v_int - v_dis[:d]

    # Normalise by mean absolute disease gap
    gap_scale = max(np.abs(disease_gap).mean(), 1e-8)
    node_movement_norm = node_movement / gap_scale
    disease_gap_norm   = disease_gap   / gap_scale

    # A node is corrected if movement is in the same direction as the gap
    toward_healthy = np.sign(node_movement) == np.sign(disease_gap)

    # Net correction: mean positive fractional correction toward healthy
    # Capped at 1.0 per node (can't over-correct)
    node_correction_frac = np.where(
        toward_healthy,
        np.minimum(np.abs(node_movement_norm) / np.abs(disease_gap_norm+1e-8), 1.0),
        0.0)
    net_correction = float(node_correction_frac.mean())

    # % of nodes corrected toward healthy
    pct_nodes_H = float(toward_healthy.mean()) * 100

    # Mean fractional correction among corrected nodes
    if toward_healthy.any():
        mean_node_H = float(node_correction_frac[toward_healthy].mean())
    else:
        mean_node_H = 0.0

    # ── Pathway annotation ─────────────────────────────────────────────
    # Top 3 nodes corrected toward healthy
    nodes_H_data = [(gene_names[i], node_correction_frac[i])
                    for i in range(d) if toward_healthy[i]]
    nodes_H_data.sort(key=lambda x: x[1], reverse=True)
    nodes_toward_H = '; '.join(
        f"{g}({v:.3f})" for g,v in nodes_H_data[:3])

    # Top 3 off-target nodes (pushed away from healthy)
    nodes_D_data = [(gene_names[i], abs(node_movement_norm[i]))
                    for i in range(d) if not toward_healthy[i]]
    nodes_D_data.sort(key=lambda x: x[1], reverse=True)
    nodes_away_H = '; '.join(
        f"{g}(−{v:.3f})" for g,v in nodes_D_data[:3])

    # Top causal edge activated
    delta_T_norm = delta_T / gap_scale
    edge_acts = []
    for i in range(d):
        for j in range(d):
            if abs(W[i,j]) > 0.05:
                act = float(W[i,j]*delta_T_norm[i])
                edge_acts.append((gene_names[i],gene_names[j],
                                  round(float(W[i,j]),2),round(act,4)))
    edge_acts.sort(key=lambda x: abs(x[3]), reverse=True)
    if edge_acts:
        src,tgt,w,act = edge_acts[0]
        tag = '→H' if act>0 else '→D'
        top_edge = f"{src}→{tgt}(w={w:+.2f},{tag})"
    else:
        top_edge = ''

    return {
        'R_score':        round(R,4),
        'net_correction': round(net_correction,4),
        'pct_nodes_H':    round(pct_nodes_H,1),
        'mean_node_H':    round(mean_node_H,4),
        'OTCI':           round(otci,4),
        'nodes_toward_H': nodes_toward_H,
        'nodes_away_H':   nodes_away_H,
        'top_causal_edge':top_edge,
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

# ── Score entire library ───────────────────────────────────────────────────

def score_library(disease, W, v_dis, v_hea, gene_names,
                  drug_mols_by_lib, lib, top_k):
    label  = LIBRARY_LABELS.get(lib, lib)
    entries = drug_mols_by_lib.get(lib, [])
    abbrev  = DISEASE_ABBREV.get(disease,'???')

    print(f"\n  Scoring {len(entries)} {label} compounds by net_correction...")

    rows = []
    for mol, entry in entries:
        try:
            props   = mol_props(mol)
            metrics = score_network_correction(mol, W, v_dis, v_hea, gene_names)
            rows.append({
                'name':           entry['name'],
                'library':        label,
                'disease':        disease,
                'indication':     entry.get('disease',''),
                'target_class':   entry.get('target',''),
                'scm_targets':    DISEASE_TARGETS.get(disease,''),
                'smiles':         entry['smiles'],
                **metrics,
                **props,
            })
        except:
            continue

    if not rows:
        return None

    df = (pd.DataFrame(rows)
            .drop_duplicates('name')
            .sort_values('net_correction', ascending=False)
            .reset_index(drop=True))

    lkey = lib.replace('_compounds','').replace('_','')
    df.head(top_k).to_csv(
        results_dir/f'{disease}_{lkey}_correction_top{top_k}.csv', index=False)
    df.to_csv(
        results_dir/f'{disease}_{lkey}_correction_all.csv', index=False)

    # Print top 10
    print(f"\n  ── {label.upper()} — {disease.upper()} — ranked by net_correction ──")
    print(f"  Primary metric: net_correction = mean fractional correction of ALL SCM nodes toward healthy")
    print(f"  {'Name':28s} {'NetCorr':8s} {'R':7s} "
          f"{'%H':6s} {'BBB':4s} {'MW':6s} {'Nodes corrected toward healthy'}")
    print(f"  {'-'*100}")
    for _, r in df.head(10).iterrows():
        b  = '+' if r['bbb_pass'] else '-'
        nh = str(r['nodes_toward_H'])[:38]
        print(f"  {r['name'][:28]:28s} "
              f"{r['net_correction']:8.4f} "
              f"{r['R_score']:7.4f} "
              f"{r['pct_nodes_H']:5.1f}% "
              f"{b:>3}  {r['mw']:6.1f}  {nh}")

    print(f"\n  n={len(df)} | net_correction: "
          f"{df['net_correction'].min():.4f}–{df['net_correction'].max():.4f} "
          f"(mean={df['net_correction'].mean():.4f})")
    return df


def score_zinc(disease, W, v_dis, v_hea, gene_names, n_zinc, top_k):
    abbrev = DISEASE_ABBREV.get(disease,'???')
    files  = sorted(graphs_dir.glob('mol_*.npz'))
    zinc_f = [f for f in files if int(f.stem.split('_')[1]) < 152559]
    random.seed(42); random.shuffle(zinc_f)

    mols = []
    for f in zinc_f[:n_zinc*3]:
        try:
            d = np.load(f, allow_pickle=True)
            mol = Chem.MolFromSmiles(str(d['smiles'][0]))
            if mol: mols.append((mol, str(d['smiles'][0])))
        except:
            pass
        if len(mols) >= n_zinc: break

    print(f"\n  Scoring {len(mols)} ZINC molecules by net_correction...")

    rows = []
    for i, (mol, smi) in enumerate(mols):
        try:
            props   = mol_props(mol)
            metrics = score_network_correction(mol, W, v_dis, v_hea, gene_names)
            rows.append({
                'name':            f"Novel-{abbrev}-{i+1:04d}",
                'library':        'ZINC',
                'disease':         disease,
                'smiles':          smi,
                'scm_targets':     DISEASE_TARGETS.get(disease,''),
                **metrics,
                **props,
            })
        except:
            continue

    if not rows:
        return None

    df = (pd.DataFrame(rows)
            .sort_values('net_correction', ascending=False)
            .reset_index(drop=True))

    df.head(top_k).to_csv(
        results_dir/f'{disease}_zinc_correction_top{top_k}.csv', index=False)
    df.to_csv(
        results_dir/f'{disease}_zinc_correction_all.csv', index=False)

    print(f"\n  ── ZINC — {disease.upper()} — ranked by net_correction ──")
    print(f"  {'Name':18s} {'NetCorr':8s} {'R':7s} "
          f"{'%H':6s} {'BBB':4s} {'MW':6s} {'Nodes corrected toward healthy'}")
    print(f"  {'-'*90}")
    for _, r in df.head(10).iterrows():
        b  = '+' if r['bbb_pass'] else '-'
        nh = str(r['nodes_toward_H'])[:38]
        print(f"  {r['name'][:18]:18s} "
              f"{r['net_correction']:8.4f} "
              f"{r['R_score']:7.4f} "
              f"{r['pct_nodes_H']:5.1f}% "
              f"{b:>3}  {r['mw']:6.1f}  {nh}")

    print(f"\n  n={len(df)} | net_correction: "
          f"{df['net_correction'].min():.4f}–{df['net_correction'].max():.4f} "
          f"(mean={df['net_correction'].mean():.4f})")
    return df


# ── Load drug library ──────────────────────────────────────────────────────

print("\nLoading drug library...")
drug_index       = json.load(open('drug_library_index.json'))
drug_mols_by_lib = defaultdict(list)
seen             = defaultdict(set)

for entry in drug_index:
    lib = entry['library']
    smi = entry['smiles']
    if smi in seen[lib]: continue
    seen[lib].add(smi)
    mol = Chem.MolFromSmiles(smi)
    if mol: drug_mols_by_lib[lib].append((mol, entry))

print("Drug library loaded:")
for lib, entries in drug_mols_by_lib.items():
    print(f"  {LIBRARY_LABELS.get(lib,lib):20s}: {len(entries)}")

# ── Main loop ──────────────────────────────────────────────────────────────

diseases  = ALL_DISEASES if args.all else [args.disease]
summary   = []

for disease in diseases:
    if not (Path('models')/f'causaldiff_{disease}_best.pt').exists():
        print(f"\nNo model for {disease} — skipping"); continue
    if not (data_dir/f'W_{disease}.npy').exists():
        print(f"\nNo SCM for {disease} — skipping"); continue

    W          = np.load(data_dir/f'W_{disease}.npy')
    v_dis      = np.load(data_dir/f'v_disease_{disease}.npy')
    v_hea      = np.load(data_dir/f'v_healthy_{disease}.npy')
    gene_names = load_gene_names(disease)
    abbrev     = DISEASE_ABBREV.get(disease,'???')

    print(f"\n{'='*65}")
    print(f"DISEASE: {disease.upper()}")
    print(f"SCM: {W.shape[0]} nodes | Targets: {DISEASE_TARGETS.get(disease,'')}")
    print(f"Top genes: {', '.join(gene_names[:5])}")
    print(f"{'='*65}")

    row = {'disease': disease}

    libs = MODE_TO_LIBS.get(args.mode, [])
    for lib in libs:
        df = score_library(disease, W, v_dis, v_hea, gene_names,
                           drug_mols_by_lib, lib, args.top_k)
        if df is not None:
            key   = lib.replace('_compounds','').replace('_','')
            top   = df.iloc[0]
            row[f'{key}_top_drug']    = top['name']
            row[f'{key}_top_netcorr'] = top['net_correction']
            row[f'{key}_top_R']       = top['R_score']
            row[f'{key}_top_pctH']    = top['pct_nodes_H']
            row[f'{key}_top_nodesH']  = top['nodes_toward_H']

    if args.mode in ('zinc','all'):
        df = score_zinc(disease, W, v_dis, v_hea, gene_names,
                        args.n_zinc, args.top_k)
        if df is not None:
            top = df.iloc[0]
            row['zinc_top_drug']    = top['name']
            row['zinc_top_netcorr'] = top['net_correction']
            row['zinc_top_R']       = top['R_score']
            row['zinc_top_pctH']    = top['pct_nodes_H']
            row['zinc_top_nodesH']  = top['nodes_toward_H']
            row['zinc_mean_netcorr']= round(df['net_correction'].mean(),4)

    summary.append(row)

# ── Summary ────────────────────────────────────────────────────────────────
if summary:
    pd.DataFrame(summary).to_csv(
        results_dir/'correction_summary_all_diseases.csv', index=False)

    print(f"\n{'='*75}")
    print("FINAL SUMMARY — ALL DISEASES — RANKED BY NET_CORRECTION")
    print("Primary metric: fraction of SCM disease network corrected toward healthy")
    print(f"{'='*75}")

    libs = MODE_TO_LIBS.get(args.mode,[])
    for lib in libs:
        key   = lib.replace('_compounds','').replace('_','')
        label = LIBRARY_LABELS.get(lib,lib)
        if not any(f'{key}_top_drug' in r for r in summary): continue
        print(f"\n{label.upper()}:")
        print(f"{'Disease':15s} {'Top drug':28s} "
              f"{'NetCorr':8s} {'R':7s} {'%H':6s} {'Nodes corrected (top 2)'}")
        print(f"{'-'*90}")
        for r in summary:
            if f'{key}_top_drug' in r:
                nh = str(r.get(f'{key}_top_nodesH','')).split(';')[0][:30]
                print(f"{r['disease']:15s} "
                      f"{r[f'{key}_top_drug'][:28]:28s} "
                      f"{r[f'{key}_top_netcorr']:8.4f} "
                      f"{r[f'{key}_top_R']:7.4f} "
                      f"{r[f'{key}_top_pctH']:5.1f}%  {nh}")

    if args.mode in ('zinc','all'):
        print(f"\nZINC NOVEL:")
        print(f"{'Disease':15s} {'Top drug':22s} "
              f"{'NetCorr':8s} {'R':7s} {'%H':6s} {'Mean NetCorr':12s}")
        print(f"{'-'*80}")
        for r in summary:
            if 'zinc_top_drug' in r:
                print(f"{r['disease']:15s} "
                      f"{r['zinc_top_drug'][:22]:22s} "
                      f"{r['zinc_top_netcorr']:8.4f} "
                      f"{r['zinc_top_R']:7.4f} "
                      f"{r['zinc_top_pctH']:5.1f}%  "
                      f"{r['zinc_mean_netcorr']:12.4f}")

    print(f"\nAll results saved to results/ as "
          f"{{disease}}_{{lib}}_correction_top20.csv")
