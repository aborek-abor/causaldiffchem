"""
CausalDiffChem — Step 2: logBB Predictor
Trains a Random Forest on the B3DB dataset (n=7,807) using RDKit descriptors.
This is the BBB permeability model referenced in the paper (Section 2.5).

Usage:
    python 02_logbb_predictor.py --data data/B3DB_classification.tsv --save models/logbb_rf.pkl
    python 02_logbb_predictor.py --predict "CCOc1ccc2nc(S(N)(=O)=O)sc2c1"   # single SMILES

Outputs:
    models/logbb_rf.pkl        — trained RandomForest model
    models/logbb_scaler.pkl    — feature scaler
    models/logbb_features.json — feature names
"""

import argparse
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.rdMolDescriptors import (
    CalcTPSA, CalcNumRings, CalcNumAromaticRings,
    CalcNumHBD, CalcNumHBA, CalcNumRotatableBonds
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, roc_auc_score, matthews_corrcoef,
    mean_absolute_error, r2_score
)


# ── Feature engineering ───────────────────────────────────────────────────────

FEATURE_NAMES = [
    'MW', 'LogP', 'TPSA', 'NumHBD', 'NumHBA', 'NumRotBonds',
    'NumRings', 'NumAromaticRings', 'NumHeavyAtoms', 'NumHeteroatoms',
    'FractionCSP3', 'MolMR',                          # molar refractivity
    'PEOE_VSA1', 'PEOE_VSA2', 'PEOE_VSA3',           # partial charge VSA
    'SMR_VSA1',  'SMR_VSA2',                          # molar refractivity VSA
    'SlogP_VSA1', 'SlogP_VSA2',                       # logP VSA
    'BertzCT',                                         # topological complexity
    'Chi0', 'Chi1', 'Chi0n', 'Chi1n',                 # connectivity indices
    'HallKierAlpha',                                   # Kier alpha
    'Kappa1', 'Kappa2', 'Kappa3',                     # kappa shape indices
    'NHOH_count', 'NO_count',
    'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
    'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles',
    'RingCount',
    'MaxAbsEStateIndex', 'MaxEStateIndex', 'MinAbsEStateIndex', 'MinEStateIndex',
    'MaxAbsPartialCharge', 'MaxPartialCharge',
    'MinAbsPartialCharge', 'MinPartialCharge',
    # Morgan fingerprint bits (ECFP4, radius=2, 64 bits)
    *[f'ECFP4_{i}' for i in range(64)],
]


def mol_to_features(smi: str) -> np.ndarray | None:
    """
    Convert SMILES to a 107-dimensional feature vector.
    Returns None if the SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    try:
        feats = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            CalcTPSA(mol),
            CalcNumHBD(mol),
            CalcNumHBA(mol),
            CalcNumRotatableBonds(mol),
            CalcNumRings(mol),
            CalcNumAromaticRings(mol),
            mol.GetNumHeavyAtoms(),
            sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (6, 1)),  # heteroatoms
            Descriptors.FractionCSP3(mol),
            Descriptors.MolMR(mol),
            Descriptors.PEOE_VSA1(mol),
            Descriptors.PEOE_VSA2(mol),
            Descriptors.PEOE_VSA3(mol),
            Descriptors.SMR_VSA1(mol),
            Descriptors.SMR_VSA2(mol),
            Descriptors.SlogP_VSA1(mol),
            Descriptors.SlogP_VSA2(mol),
            Descriptors.BertzCT(mol),
            Descriptors.Chi0(mol),
            Descriptors.Chi1(mol),
            Descriptors.Chi0n(mol),
            Descriptors.Chi1n(mol),
            Descriptors.HallKierAlpha(mol),
            Descriptors.Kappa1(mol),
            Descriptors.Kappa2(mol),
            Descriptors.Kappa3(mol),
            Descriptors.NHOHCount(mol),
            Descriptors.NOCount(mol),
            Descriptors.NumAliphaticCarbocycles(mol),
            Descriptors.NumAliphaticHeterocycles(mol),
            Descriptors.NumSaturatedCarbocycles(mol),
            Descriptors.NumSaturatedHeterocycles(mol),
            Descriptors.RingCount(mol),
            Descriptors.MaxAbsEStateIndex(mol),
            Descriptors.MaxEStateIndex(mol),
            Descriptors.MinAbsEStateIndex(mol),
            Descriptors.MinEStateIndex(mol),
            Descriptors.MaxAbsPartialCharge(mol),
            Descriptors.MaxPartialCharge(mol),
            Descriptors.MinAbsPartialCharge(mol),
            Descriptors.MinPartialCharge(mol),
        ]

        # ECFP4 fingerprint (64-bit folded)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=64)
        feats.extend(list(fp))

        arr = np.array(feats, dtype=np.float32)
        # replace inf/nan with 0
        arr = np.where(np.isfinite(arr), arr, 0.0)
        return arr

    except Exception:
        return None


# ── Data loading ──────────────────────────────────────────────────────────────

def load_b3db(tsv_path: str):
    """
    Load B3DB_classification.tsv.
    Returns X (features), y_cls (BBB+/BBB- as 1/0), y_reg (logBB floats).
    """
    df = pd.read_csv(tsv_path, sep='\t')
    print(f"B3DB: {len(df)} compounds loaded")
    print(f"  Columns: {list(df.columns[:6])}")
    print(f"  BBB+ count: {(df['BBB+/BBB-'] == 'BBB+').sum()}  "
          f"BBB- count: {(df['BBB+/BBB-'] == 'BBB-').sum()}")

    X, y_cls, y_reg, valid_smiles = [], [], [], []

    for _, row in df.iterrows():
        smi = str(row['SMILES'])
        feat = mol_to_features(smi)
        if feat is None:
            continue

        label = 1 if str(row['BBB+/BBB-']).strip() == 'BBB+' else 0
        try:
            logbb = float(row['logBB'])
        except (ValueError, KeyError):
            logbb = np.nan

        X.append(feat)
        y_cls.append(label)
        y_reg.append(logbb)
        valid_smiles.append(smi)

    X = np.array(X)
    y_cls = np.array(y_cls)
    y_reg = np.array(y_reg)

    print(f"  Valid after featurisation: {len(X)}")
    return X, y_cls, y_reg, valid_smiles


# ── Training ───────────────────────────────────────────────────────────────────

def train_classifier(X, y, n_estimators=300, cv=5):
    """Train BBB+/BBB- Random Forest classifier with 5-fold CV."""
    print(f"\n── Training RF Classifier (n_estimators={n_estimators}, CV={cv}) ──")

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced',
    )

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    auc_scores = cross_val_score(clf, X_sc, y, cv=skf, scoring='roc_auc', n_jobs=-1)
    acc_scores  = cross_val_score(clf, X_sc, y, cv=skf, scoring='accuracy', n_jobs=-1)

    print(f"  CV ROC-AUC : {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")
    print(f"  CV Accuracy: {acc_scores.mean():.3f} ± {acc_scores.std():.3f}")

    clf.fit(X_sc, y)   # refit on full data

    # feature importance top-10
    importances = clf.feature_importances_
    top10 = np.argsort(importances)[::-1][:10]
    print(f"\n  Top-10 features by importance:")
    for rank, idx in enumerate(top10):
        fname = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"feat_{idx}"
        print(f"    {rank+1:2d}. {fname:30s} {importances[idx]:.4f}")

    return clf, scaler


def train_regressor(X, y_reg, n_estimators=300, cv=5):
    """Train logBB regression RF on compounds with known logBB values."""
    mask = np.isfinite(y_reg)
    X_r, y_r = X[mask], y_reg[mask]
    print(f"\n── Training RF Regressor (n_compounds_with_logBB={mask.sum()}, CV={cv}) ──")

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_r)

    reg = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    mae_scores = cross_val_score(reg, X_sc, y_r, cv=kf, scoring='neg_mean_absolute_error')
    r2_scores  = cross_val_score(reg, X_sc, y_r, cv=kf, scoring='r2')

    print(f"  CV MAE : {-mae_scores.mean():.3f} ± {mae_scores.std():.3f}")
    print(f"  CV R²  : {r2_scores.mean():.3f} ± {r2_scores.std():.3f}")

    reg.fit(X_sc, y_r)
    return reg, scaler


# ── Prediction ─────────────────────────────────────────────────────────────────

def predict_single(smi: str, clf, clf_scaler, reg=None, reg_scaler=None):
    """Predict BBB class and logBB for a single SMILES."""
    feat = mol_to_features(smi)
    if feat is None:
        return {'error': 'Invalid SMILES'}

    feat_2d = feat.reshape(1, -1)
    X_cls = clf_scaler.transform(feat_2d)
    prob  = clf.predict_proba(X_cls)[0][1]
    cls   = 'BBB+' if prob >= 0.5 else 'BBB-'

    result = {'smiles': smi, 'BBB_class': cls, 'BBB+_probability': round(float(prob), 3)}

    if reg is not None:
        X_reg = reg_scaler.transform(feat_2d)
        logbb = float(reg.predict(X_reg)[0])
        result['logBB'] = round(logbb, 3)
        result['logBB_interpretation'] = (
            'CNS penetrant (logBB > −1)' if logbb > -1
            else 'Unlikely CNS penetrant (logBB ≤ −1)'
        )

    return result


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train logBB predictor on B3DB')
    parser.add_argument('--data',    type=str, default='data/B3DB_classification.tsv')
    parser.add_argument('--save',    type=str, default='models/logbb_rf.pkl')
    parser.add_argument('--predict', type=str, default=None, help='Predict a single SMILES')
    parser.add_argument('--trees',   type=int, default=300)
    parser.add_argument('--cv',      type=int, default=5)
    args = parser.parse_args()

    Path('models').mkdir(exist_ok=True)

    print("=" * 60)
    print("CausalDiffChem — logBB Predictor Training")
    print("=" * 60)

    # Load
    X, y_cls, y_reg, smiles = load_b3db(args.data)

    # Train classifier
    clf, clf_scaler = train_classifier(X, y_cls, n_estimators=args.trees, cv=args.cv)

    # Train regressor (on subset with continuous logBB)
    reg, reg_scaler = train_regressor(X, y_reg, n_estimators=args.trees, cv=args.cv)

    # Save all artefacts
    model_bundle = {
        'classifier':   clf,
        'clf_scaler':   clf_scaler,
        'regressor':    reg,
        'reg_scaler':   reg_scaler,
        'feature_names': FEATURE_NAMES,
        'n_features':   X.shape[1],
        'n_train_cls':  len(y_cls),
        'n_train_reg':  int(np.isfinite(y_reg).sum()),
    }
    save_path = args.save
    with open(save_path, 'wb') as f:
        pickle.dump(model_bundle, f)
    print(f"\n  Model bundle saved to {save_path}")

    # Optionally predict a single compound
    if args.predict:
        result = predict_single(args.predict, clf, clf_scaler, reg, reg_scaler)
        print(f"\n  Prediction for '{args.predict}':")
        for k, v in result.items():
            print(f"    {k}: {v}")
    else:
        # Demo: predict on a few known CNS drugs
        test_cases = [
            ("CNS penetrant — caffeine",         "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
            ("CNS penetrant — ibuprofen",         "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
            ("CNS penetrant — diazepam",          "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21"),
            ("Non-penetrant — metformin",         "CN(C)C(=N)NC(=N)N"),
            ("Target compound — dasatinib-like",  "CC1=C(C=CC(=C1)NC2=NC=CC(=N2)NC3=CC(=CC(=C3)Cl)Cl)C"),
        ]
        print("\n  ── Demo predictions ──")
        for name, smi in test_cases:
            r = predict_single(smi, clf, clf_scaler, reg, reg_scaler)
            print(f"  {name}")
            print(f"    → {r.get('BBB_class')}  P(BBB+)={r.get('BBB+_probability')}  "
                  f"logBB={r.get('logBB', 'n/a')}  "
                  f"{r.get('logBB_interpretation','')}")
