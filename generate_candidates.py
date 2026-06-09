"""
generate_candidates.py (v8)
============================
Scientifically correct scoring + pathway annotation saved in every CSV.

Scoring schemes:
  FDA/Clinical non-neuro : R - 0.15*OTCI
  FDA/Clinical neuro     : (0.80*R + 0.20*BBB - 0.10*OTCI) / 1.10
  Preclinical non-neuro  : 0.80*R + 0.20*QED - 0.10*OTCI
  Preclinical neuro      : (0.70*R + 0.15*QED + 0.15*BBB - 0.10*OTCI) / 1.10
  ZINC all diseases      : 0.50*R + 0.20*QED + 0.15*BBB + 0.10*Ro5 - 0.05*OTCI

Pathway annotation columns added to every CSV:
  pathway_nodes : top 3 SCM genes perturbed by the drug
  pathway_edges : top 2 causal edges activated by the drug

Usage:
    py -3.11 generate_candidates.py --disease ms_blood --mode all --top_k 20
    py -3.11 generate_candidates.py --all --mode all --top_k 20 --n_zinc 1000
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

# ── Arguments ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--disease', type=str, default='ms_blood')
parser.add_argument('--all',    action='store_true')
parser.add_argument('--mode',   type=str, default='all',
                    choices=['fda','clinical','preclinical','zinc','all'])
parser.add_argument('--top_k',  type=int, default=20)
parser.add_argument('--n_zinc', type=int, default=2000)
args = parser.parse_args()

# ── Paths ──────────────────────────────────────────────────────────────────
data_dir    = Path('data')
graphs_dir  = Path('graphs')
results_dir = Path('results')
results_dir.mkdir(exist_ok=True)

# ── Disease configurations ─────────────────────────────────────────────────
ALL_DISEASES = [
    'ms_blood','hiv','ad','tuberculosis','dengue',
    'sarcoidosis2','leishmaniasis2','malaria2',
    'breast_cancer','breast_cancer2',
]

NEURO_DISEASES = {'ms_blood', 'hiv', 'ad'}

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
        return bool(proxy), round(logp*0.3-1.2, 3), float(proxy)
    try:
        fp    = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 64)
        desc  = np.array([mw, logp, tpsa, hbd,
                          Descriptors.NumHAcceptors(mol), rotb,
                          Descriptors.RingCount(mol),
                          Descriptors.FractionCSP3(mol),
                          Descriptors.NumAromaticRings(mol),
                          Descriptors.NumHeteroatoms(mol)])
        feats = np.concatenate([desc, np.array(fp)]).reshape(1,-1)
        fc    = bbb_bundle['clf_scaler'].transform(feats)
        prob  = float(bbb_bundle['classifier'].predict_proba(fc)[0][1])
        fr    = bbb_bundle['reg_scaler'].transform(feats)
        logbb = float(bbb_bundle['regressor'].predict(fr)[0])
        return prob>0.5, round(logbb,3), round(prob,3)
    except:
        return bool(proxy), round(logp*0.3-1.2,3), float(proxy)

def mol_base_props(mol):
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

# ── Core computation: R score + pathway annotation ─────────────────────────

def compute_R_and_pathway(mol, W, v_dis, v_hea, gene_names):
    """
    Computes:
      R     — phenotype reversal score (do-calculus cosine similarity)
      otci  — off-target causal impact
      pathway_nodes — top 3 SCM genes this drug perturbs most
      pathway_edges — top 2 causal edges this drug most activates
    """
    d = W.shape[0]

    # Molecule fingerprint → perturbation vector
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

    # Do-calculus: propagate through causal graph
    try:
        M_inv = np.linalg.solve(np.eye(d)-W+1e-6*np.eye(d), np.eye(d))
    except:
        M_inv = np.eye(d)

    v_int = M_inv @ (v_dis[:d] + delta_T)

    # R score
    R = float(np.clip(
        np.dot(v_int, v_hea[:d]) /
        (np.linalg.norm(v_int)*np.linalg.norm(v_hea[:d])+1e-8), -1, 1))

    # Off-target causal impact
    otci = float(np.mean(np.abs(v_int-v_hea[:d])) /
                 (np.abs(v_hea[:d]).mean()+1e-8))

    # ── Pathway annotation ────────────────────────────────────────────────

    # Top 3 nodes: which genes does this drug perturb most strongly?
    # Normalize delta_T by mean absolute expression difference for interpretability
    expr_scale = max(np.abs(expr_diff).mean(), 1e-8)
    delta_T_norm = delta_T / expr_scale

    node_corr = v_int - v_dis[:d]
    node_data = list(zip(gene_names[:d], delta_T_norm[:d], node_corr[:d]))
    node_data.sort(key=lambda x: abs(x[1]), reverse=True)

    top_nodes = []
    for gene, dT_n, corr in node_data[:3]:
        direction = '+' if corr > 0 else '-'
        top_nodes.append(f"{gene}({direction}{abs(dT_n):.3f})")
    pathway_nodes = '; '.join(top_nodes)

    # Top 2 edges: which causal edges does this drug most activate?
    # edge activation = W[i,j] * delta_T[i]  (how much source node is perturbed × edge weight)
    # positive activation = drug pushes along an edge toward healthy expression
    edge_acts = []
    for i in range(d):
        for j in range(d):
            if abs(W[i,j]) > 0.05:
                activation = float(W[i,j] * delta_T_norm[i])
                edge_acts.append((
                    gene_names[i], gene_names[j],
                    round(float(W[i,j]),2),
                    round(activation,4)
                ))
    edge_acts.sort(key=lambda x: abs(x[3]), reverse=True)

    top_edges = []
    for src, tgt, w, act in edge_acts[:2]:
        direction = 'H' if act > 0 else 'D'  # H=toward healthy, D=toward disease
        top_edges.append(f"{src}→{tgt}(w={w:+.2f},→{direction})")
    pathway_edges = '; '.join(top_edges)

    return R, otci, pathway_nodes, pathway_edges

# ── Scoring functions ──────────────────────────────────────────────────────

def composite_score(R, qed, bbb_pass, ro5, otci, lib, is_neuro):
    if lib in ('approved_drugs','clinical_compounds'):
        if is_neuro:
            return round((0.80*R + 0.20*float(bbb_pass)
                          - 0.10*min(otci,2.0)/2.0) / 1.10, 4)
        else:
            return round(R - 0.15*min(otci,2.0)/2.0, 4)
    elif lib == 'preclinical_compounds':
        if is_neuro:
            return round((0.70*R + 0.15*qed + 0.15*float(bbb_pass)
                          - 0.10*min(otci,2.0)/2.0) / 1.10, 4)
        else:
            return round(0.80*R + 0.20*qed - 0.10*min(otci,2.0)/2.0, 4)
    else:  # ZINC
        return round(0.50*R + 0.20*qed + 0.15*float(bbb_pass)
                     + 0.10*float(ro5) - 0.05*min(otci,2.0)/2.0, 4)

def scoring_note(lib, is_neuro):
    if lib in ('approved_drugs','clinical_compounds'):
        return 'R+BBB-OTCI' if is_neuro else 'R-OTCI'
    elif lib == 'preclinical_compounds':
        return 'R+QED+BBB-OTCI' if is_neuro else 'R+QED-OTCI'
    else:
        return '0.50R+0.20QED+0.15BBB+0.10Ro5-0.05OTCI'

# ── Load gene names ────────────────────────────────────────────────────────

def load_gene_names(disease):
    f = data_dir/f'{disease}_scm_genes.txt'
    if not f.exists():
        return [f'Gene_{i}' for i in range(20)]
    genes = open(f).read().strip().split('\n')
    return [g.replace('SEX_CONFOUNDER_','')+'*'
            if g.startswith('SEX_CONFOUNDER_') else g
            for g in genes]

# ── Score named drug library ───────────────────────────────────────────────

def score_drug_library(disease, W, v_dis, v_hea, gene_names,
                       drug_mols_by_lib, libraries, top_k, is_neuro):
    abbrev  = DISEASE_ABBREV.get(disease, disease[:3].upper())
    results = {}

    for lib in libraries:
        entries = drug_mols_by_lib.get(lib, [])
        if not entries:
            continue

        label = LIBRARY_LABELS.get(lib, lib)
        print(f"\n  Scoring {len(entries)} {label} compounds "
              f"[{scoring_note(lib,is_neuro)}]...")

        rows = []
        for mol, entry in entries:
            try:
                props = mol_base_props(mol)
                R, otci, p_nodes, p_edges = compute_R_and_pathway(
                    mol, W, v_dis, v_hea, gene_names)
                sc = composite_score(R, props['qed'], props['bbb_pass'],
                                     props['ro5'], otci, lib, is_neuro)
                rows.append({
                    'candidate_id':   f"{abbrev}-{entry['name'][:12]}",
                    'name':           entry['name'],
                    'library':        lib,
                    'library_label':  label,
                    'indication':     entry.get('disease',''),
                    'target_class':   entry.get('target',''),
                    'disease':        disease,
                    'scm_targets':    DISEASE_TARGETS.get(disease,''),
                    'smiles':         entry['smiles'],
                    'R_score':        round(R,4),
                    'OTCI':           round(otci,4),
                    'score':          sc,
                    'scoring_note':   scoring_note(lib,is_neuro),
                    'pathway_nodes':  p_nodes,
                    'pathway_edges':  p_edges,
                    **props,
                })
            except:
                continue

        if not rows:
            continue

        df = (pd.DataFrame(rows)
                .drop_duplicates('name')
                .sort_values('score', ascending=False)
                .reset_index(drop=True))

        lkey = lib.replace('_compounds','').replace('_','')
        df.head(top_k).to_csv(
            results_dir/f'{disease}_{lkey}_top{top_k}.csv', index=False)
        df.to_csv(
            results_dir/f'{disease}_{lkey}_all.csv', index=False)

        results[lib] = df

        # Print top 10
        print(f"\n  ── {label.upper()} — {disease.upper()} ──")
        print(f"  {'Name':25s} {'R':7s} {'Score':7s} {'BBB':4s} "
              f"{'MW':6s} {'Pathway nodes (top 2)'}")
        print(f"  {'-'*90}")
        for _, r in df.head(10).iterrows():
            b   = '+' if r['bbb_pass'] else '-'
            pn  = str(r['pathway_nodes'])[:35]
            print(f"  {r['name'][:25]:25s} {r['R_score']:7.4f} "
                  f"{r['score']:7.4f} {b:>3}  {r['mw']:6.1f}  {pn}")

        print(f"\n  n={len(df)} | R: {df['R_score'].min():.4f}–"
              f"{df['R_score'].max():.4f} "
              f"(mean={df['R_score'].mean():.4f})")

    return results

# ── Score ZINC molecules ───────────────────────────────────────────────────

def score_zinc(disease, W, v_dis, v_hea, gene_names, n_zinc, top_k):
    abbrev = DISEASE_ABBREV.get(disease, disease[:3].upper())
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
        if len(mols) >= n_zinc:
            break

    print(f"\n  ── ZINC NOVEL — {disease.upper()} (n={len(mols)}) ──")
    print(f"  Scoring: 0.50*R + 0.20*QED + 0.15*BBB + 0.10*Ro5 - 0.05*OTCI")

    rows = []
    for i, (mol, smi) in enumerate(mols):
        try:
            props = mol_base_props(mol)
            R, otci, p_nodes, p_edges = compute_R_and_pathway(
                mol, W, v_dis, v_hea, gene_names)
            sc = composite_score(R, props['qed'], props['bbb_pass'],
                                  props['ro5'], otci, 'zinc', False)
            rows.append({
                'candidate_id':  f"{abbrev}-Z{i+1:04d}",
                'name':          f"Novel-{abbrev}-{i+1:04d}",
                'source':        'ZINC250k',
                'smiles':        smi,
                'disease':       disease,
                'scm_targets':   DISEASE_TARGETS.get(disease,''),
                'R_score':       round(R,4),
                'OTCI':          round(otci,4),
                'score':         sc,
                'scoring_note':  '0.50R+0.20QED+0.15BBB+0.10Ro5-0.05OTCI',
                'pathway_nodes': p_nodes,
                'pathway_edges': p_edges,
                **props,
            })
        except:
            continue

    if not rows:
        return None

    df = (pd.DataFrame(rows)
            .sort_values('score', ascending=False)
            .reset_index(drop=True))

    df.head(top_k).to_csv(
        results_dir/f'{disease}_zinc_top{top_k}.csv', index=False)
    df.to_csv(
        results_dir/f'{disease}_zinc_all.csv', index=False)

    print(f"  Scored: {len(df)} | R: {df['R_score'].min():.4f}–"
          f"{df['R_score'].max():.4f} (std={df['R_score'].std():.4f})")
    print(f"  BBB: {df['bbb_pass'].sum()}/{len(df)} "
          f"({100*df['bbb_pass'].mean():.1f}%)")
    print(f"\n  Top 10:")
    print(f"  {'ID':14s} {'R':7s} {'Score':7s} {'BBB':3s} "
          f"{'MW':6s}  {'Pathway nodes (top 2)'}")
    print(f"  {'-'*75}")
    for _, r in df.head(10).iterrows():
        b  = '+' if r['bbb_pass'] else '-'
        pn = str(r['pathway_nodes'])[:30]
        print(f"  {r['candidate_id']:14s} {r['R_score']:7.4f} "
              f"{r['score']:7.4f} {b:>3}  {r['mw']:6.1f}  {pn}")
    return df

# ── Load drug library ──────────────────────────────────────────────────────

print("\nLoading drug library index...")
drug_index = json.load(open('drug_library_index.json'))
drug_mols_by_lib = defaultdict(list)
seen = defaultdict(set)
for entry in drug_index:
    lib = entry['library']
    smi = entry['smiles']
    if smi in seen[lib]:
        continue
    seen[lib].add(smi)
    mol = Chem.MolFromSmiles(smi)
    if mol:
        drug_mols_by_lib[lib].append((mol, entry))

print("Drug library loaded (deduplicated):")
for lib, entries in drug_mols_by_lib.items():
    print(f"  {LIBRARY_LABELS.get(lib,lib):20s}: {len(entries)}")

# ── Main loop ──────────────────────────────────────────────────────────────

diseases = ALL_DISEASES if args.all else [args.disease]
summary  = []

for disease in diseases:
    if not (Path('models')/f'causaldiff_{disease}_best.pt').exists():
        print(f"\nNo model for {disease} — skipping")
        continue

    print(f"\n{'='*65}")
    print(f"DISEASE: {disease.upper()}")
    is_neuro = disease in NEURO_DISEASES
    print(f"Type: {'NEUROLOGICAL' if is_neuro else 'NON-NEUROLOGICAL'}")
    print(f"{'='*65}")

    if not (data_dir/f'W_{disease}.npy').exists():
        print("No SCM data — skipping")
        continue

    W       = np.load(data_dir/f'W_{disease}.npy')
    v_dis   = np.load(data_dir/f'v_disease_{disease}.npy')
    v_hea   = np.load(data_dir/f'v_healthy_{disease}.npy')
    n_edg   = int(np.sum(np.abs(W)>0.05))
    gene_names = load_gene_names(disease)

    print(f"SCM: {W.shape[0]} nodes, {n_edg} edges")
    print(f"Targets: {DISEASE_TARGETS.get(disease,'')}")
    print(f"Top SCM genes: {', '.join(gene_names[:5])}")

    row = {'disease':disease, 'is_neuro':is_neuro, 'scm_edges':n_edg,
           'targets':DISEASE_TARGETS.get(disease,'')}

    # Score drug libraries
    libs_to_score = MODE_TO_LIBS.get(args.mode, [])
    if libs_to_score:
        drug_results = score_drug_library(
            disease, W, v_dis, v_hea, gene_names,
            drug_mols_by_lib, libs_to_score, args.top_k, is_neuro)

        for lib, df in drug_results.items():
            key = lib.replace('_compounds','').replace('_','')
            if len(df) > 0:
                top = df.iloc[0]
                row[f'{key}_top_drug']    = top['name']
                row[f'{key}_top_R']       = top['R_score']
                row[f'{key}_top_score']   = top['score']
                row[f'{key}_top_logBB']   = top['logbb']
                row[f'{key}_top_BBB']     = '+' if top['bbb_pass'] else '-'
                row[f'{key}_top_pathway'] = top['pathway_nodes']
                row[f'{key}_top_edges']   = top['pathway_edges']

    # Score ZINC
    if args.mode in ('zinc','all'):
        zinc_df = score_zinc(disease, W, v_dis, v_hea, gene_names,
                             args.n_zinc, args.top_k)
        if zinc_df is not None:
            top = zinc_df.iloc[0]
            row['zinc_top_R']       = top['R_score']
            row['zinc_top_score']   = top['score']
            row['zinc_mean_R']      = round(zinc_df['R_score'].mean(),4)
            row['zinc_R_std']       = round(zinc_df['R_score'].std(),4)
            row['zinc_bbb_pct']     = round(100*zinc_df['bbb_pass'].mean(),1)
            row['zinc_top_pathway'] = top['pathway_nodes']

    summary.append(row)

# ── Summary ────────────────────────────────────────────────────────────────
if summary:
    pd.DataFrame(summary).to_csv(
        results_dir/'summary_all_diseases.csv', index=False)

    print(f"\n{'='*75}")
    print("CROSS-DISEASE SUMMARY WITH PATHWAY ANNOTATION")
    print(f"{'='*75}")

    libs = MODE_TO_LIBS.get(args.mode, [])
    for lib in libs:
        key   = lib.replace('_compounds','').replace('_','')
        label = LIBRARY_LABELS.get(lib, lib)
        if not any(f'{key}_top_drug' in r for r in summary):
            continue
        print(f"\n{label.upper()}:")
        print(f"{'Disease':15s} {'Top Drug':25s} {'R':>7s} "
              f"{'Score':>7s} {'Pathway nodes'}")
        print(f"{'-'*85}")
        for r in summary:
            if f'{key}_top_drug' in r:
                print(f"{r['disease']:15s} "
                      f"{r[f'{key}_top_drug'][:25]:25s} "
                      f"{r[f'{key}_top_R']:>7.4f} "
                      f"{r[f'{key}_top_score']:>7.4f}  "
                      f"{str(r.get(f'{key}_top_pathway',''))[:35]}")

    if args.mode in ('zinc','all'):
        print(f"\nZINC NOVEL CANDIDATES:")
        print(f"{'Disease':15s} {'Top R':>7s} {'Score':>7s} "
              f"{'Mean R':>7s} {'BBB%':>6s}  {'Top pathway nodes'}")
        print(f"{'-'*80}")
        for r in summary:
            if 'zinc_top_R' in r:
                print(f"{r['disease']:15s} {r['zinc_top_R']:>7.4f} "
                      f"{r['zinc_top_score']:>7.4f} "
                      f"{r['zinc_mean_R']:>7.4f} "
                      f"{r['zinc_bbb_pct']:>5.1f}%  "
                      f"{str(r.get('zinc_top_pathway',''))[:35]}")

    print(f"\nAll results saved to results/")
    print("Each CSV now includes 'pathway_nodes' and 'pathway_edges' columns.")
