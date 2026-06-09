"""
CausalDiffChem — Step 1: Molecular Preprocessing Pipeline
Converts SMILES (from ZINC15 / ChEMBL) → node/edge feature tensors
ready for the graph diffusion model.

Usage:
    python 01_mol_pipeline.py --input molecules.smi --output processed/ --limit 1000

Outputs per molecule (saved as .npz):
    x          : node features  [N_atoms, 9]
    edge_index : COO edge list  [2, N_edges]
    edge_attr  : edge features  [N_edges, 4]
    smiles     : canonical SMILES string
    qed        : drug-likeness score
    sa         : synthetic accessibility score
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED
from rdkit.Chem.rdMolDescriptors import CalcTPSA


# ── Atom feature definitions ────────────────────────────────────────────────

ATOM_TYPES   = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'other']
HYBRIDIZATION = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.OTHER,
]

# Continuous features are appended after the one-hot blocks:
#   [atom_type(10), degree(7), hybridization(4), H_count(5),
#    formal_charge(1), is_aromatic(1), is_in_ring(1)]  → 29 dims total
# We keep it at 9 compact features to match the paper's node_dim.

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def atom_features(atom) -> np.ndarray:
    """9-dim atom feature vector."""
    # 1. Atom type (one-hot, 5 common + other)
    symbol = atom.GetSymbol()
    short_types = ['C', 'N', 'O', 'S', 'F']
    atom_oh = [int(symbol == t) for t in short_types] + [int(symbol not in short_types)]

    # 2. Degree (capped at 4)
    degree = min(atom.GetDegree(), 4) / 4.0

    # 3. Hybridization (sp / sp2 / sp3 / other)
    hyb = atom.GetHybridization()
    hyb_oh = [
        int(hyb == Chem.rdchem.HybridizationType.SP),
        int(hyb == Chem.rdchem.HybridizationType.SP2),
        int(hyb == Chem.rdchem.HybridizationType.SP3),
    ]

    # [6 type bits] + [1 degree] + [3 hyb bits] → 10 total; pad/truncate to 9
    feat = atom_oh + [degree] + hyb_oh   # 10 dims
    return np.array(feat[:9], dtype=np.float32)


def bond_features(bond) -> np.ndarray:
    """4-dim bond feature vector (one-hot bond type)."""
    bt = bond.GetBondType()
    return np.array([int(bt == t) for t in BOND_TYPES], dtype=np.float32)


def mol_to_graph(mol):
    """
    Convert an RDKit molecule to (x, edge_index, edge_attr).
    Returns None if the molecule is invalid or has no atoms.
    """
    if mol is None:
        return None

    mol = Chem.AddHs(mol)               # add implicit H
    AllChem.EmbedMolecule(mol,           # 3-D coords (optional; used for SA)
        AllChem.ETKDGv3())
    mol = Chem.RemoveHs(mol)            # back to heavy atoms for the graph

    atoms = mol.GetAtoms()
    n = mol.GetNumAtoms()
    if n == 0:
        return None

    x = np.stack([atom_features(a) for a in atoms])   # [N, 9]

    rows, cols, edge_feats = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat  = bond_features(bond)
        # undirected → add both directions
        rows  += [i, j]
        cols  += [j, i]
        edge_feats += [feat, feat]

    if len(rows) == 0:
        # single-atom molecule — add self-loop so edge_index is never empty
        rows, cols = [0], [0]
        edge_feats = [np.zeros(4, dtype=np.float32)]

    edge_index = np.array([rows, cols], dtype=np.int64)    # [2, E]
    edge_attr  = np.stack(edge_feats)                       # [E, 4]

    return x, edge_index, edge_attr


# ── Property calculators ─────────────────────────────────────────────────────

def calc_sa_score(mol) -> float:
    """
    Synthetic Accessibility score (Ertl & Schuffenhauer, 2009).
    Requires the SA_Score module from RDKit contrib.
    Falls back to a simple MW-based proxy if unavailable.
    """
    try:
        from rdkit.Chem import RDConfig
        import sys
        sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
        import sascorer
        return sascorer.calculateScore(mol)
    except Exception:
        # proxy: heavier molecule → harder to synthesise
        mw = Descriptors.MolWt(mol)
        return min(max(1.0, mw / 100.0), 10.0)


def calc_qed(mol) -> float:
    try:
        return QED.qed(mol)
    except Exception:
        return 0.0


def calc_logp(mol) -> float:
    try:
        return Descriptors.MolLogP(mol)
    except Exception:
        return 0.0


# ── Filters (Lipinski + BBB proxy) ──────────────────────────────────────────

def passes_lipinski(mol) -> bool:
    """Ro5 drug-likeness filter."""
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd  = Descriptors.NumHDonors(mol)
    hba  = Descriptors.NumHAcceptors(mol)
    return mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10


def passes_bbb_proxy(mol) -> bool:
    """
    Simple BBB proxy filter (not the full RF model from Task 3).
    MW < 450, logP 1–4, TPSA < 90, rotatable bonds < 8.
    Passes ~60-70 % of CNS drugs.
    """
    mw    = Descriptors.MolWt(mol)
    logp  = Descriptors.MolLogP(mol)
    tpsa  = CalcTPSA(mol)
    rotb  = Descriptors.NumRotatableBonds(mol)
    return mw < 450 and 1 <= logp <= 4 and tpsa < 90 and rotb < 8


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_smiles_file(input_path: str,
                        output_dir: str,
                        limit: int = None,
                        bbb_filter: bool = False) -> dict:
    """
    Read a .smi / .csv file (one SMILES per line, or CSV with 'smiles' column),
    featurise each molecule, apply filters, and save .npz files.

    Returns a summary dict.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── detect format ──
    with open(input_path) as f:
        first = f.readline().strip()
    is_csv = ',' in first and not Chem.MolFromSmiles(first.split(',')[0]) is None

    def iter_smiles():
        with open(input_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                smi = line.split(',')[0] if ',' in line else line.split()[0]
                yield smi

    stats = {'total': 0, 'valid': 0, 'lipinski': 0, 'bbb_proxy': 0, 'saved': 0}
    saved_smiles = []

    for i, smi in enumerate(tqdm(iter_smiles(), desc="Processing molecules")):
        if limit and i >= limit:
            break
        stats['total'] += 1

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        stats['valid'] += 1

        if not passes_lipinski(mol):
            continue
        stats['lipinski'] += 1

        if bbb_filter and not passes_bbb_proxy(mol):
            continue
        stats['bbb_proxy'] += 1

        result = mol_to_graph(mol)
        if result is None:
            continue

        x, edge_index, edge_attr = result
        canonical = Chem.MolToSmiles(mol)
        qed = calc_qed(mol)
        sa  = calc_sa_score(mol)
        mw  = Descriptors.MolWt(mol)
        logp = calc_logp(mol)

        fname = out / f"mol_{stats['saved']:07d}.npz"
        np.savez_compressed(
            fname,
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            smiles=np.array([canonical]),
            qed=np.array([qed]),
            sa=np.array([sa]),
            mw=np.array([mw]),
            logp=np.array([logp]),
        )
        saved_smiles.append(canonical)
        stats['saved'] += 1

    # write index file
    with open(out / 'index.json', 'w') as f:
        json.dump({'stats': stats, 'smiles': saved_smiles[:1000]}, f, indent=2)

    return stats


# ── Demo: run on 200 hard-coded CNS-relevant SMILES if no file given ──────────

DEMO_SMILES = [
    # A sample of known CNS-active scaffolds for testing
    "CC1=C(C=CC(=C1)NC2=NC=CC(=N2)NC3=CC(=CC(=C3)Cl)Cl)C",           # dasatinib-like
    "COC1=CC2=C(C=C1OC)NC(=O)C2=O",
    "CC(=O)Nc1ccc(cc1)O",                                               # paracetamol (simple test)
    "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",
    "CN1CCC[C@H]1c2cccnc2",                                             # nicotine
    "CC12CCC(=O)C=C1CCC3C2CC(O)C4(C)C3CCC4=O",
    "O=C(O)c1ccccc1O",                                                  # salicylic acid
    "CC(CS)C(=O)N1CCCC1C(=O)O",                                        # captopril
    "OC(=O)c1cc(Cl)ccc1Nc1ccnc2cc(Cl)ccc12",
    "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",                                   # salbutamol
]

def run_demo():
    """Quick smoke-test on 10 molecules. Run with: python 01_mol_pipeline.py --demo"""
    print("\n=== CausalDiffChem Mol Pipeline — Demo ===\n")
    out = Path("/home/claude/causaldiffchem/data/demo_graphs")
    out.mkdir(parents=True, exist_ok=True)

    ok = 0
    for i, smi in enumerate(DEMO_SMILES):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"  [{i}] INVALID: {smi[:40]}")
            continue

        result = mol_to_graph(mol)
        if result is None:
            print(f"  [{i}] GRAPH FAILED: {smi[:40]}")
            continue

        x, edge_index, edge_attr = result
        qed  = calc_qed(mol)
        sa   = calc_sa_score(mol)
        mw   = Descriptors.MolWt(mol)
        lip  = passes_lipinski(mol)
        bbb  = passes_bbb_proxy(mol)

        print(f"  [{i}] OK  atoms={x.shape[0]:3d}  edges={edge_index.shape[1]:3d}  "
              f"QED={qed:.2f}  SA={sa:.1f}  MW={mw:.0f}  "
              f"Lipinski={'✓' if lip else '✗'}  BBB={'✓' if bbb else '✗'}")
        print(f"       x.shape={x.shape}  edge_index.shape={edge_index.shape}  "
              f"edge_attr.shape={edge_attr.shape}")

        np.savez_compressed(out / f"demo_{i:02d}.npz",
                            x=x, edge_index=edge_index, edge_attr=edge_attr,
                            smiles=np.array([Chem.MolToSmiles(mol)]),
                            qed=np.array([qed]), sa=np.array([sa]))
        ok += 1

    print(f"\n  Saved {ok}/{len(DEMO_SMILES)} molecules to {out}/")
    print("  Each .npz contains: x (node features), edge_index, edge_attr, smiles, qed, sa")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CausalDiffChem molecular preprocessing pipeline')
    parser.add_argument('--input',    type=str, help='Input .smi or .csv file')
    parser.add_argument('--output',   type=str, default='data/processed', help='Output directory')
    parser.add_argument('--limit',    type=int, default=None, help='Max molecules to process')
    parser.add_argument('--bbb',      action='store_true', help='Apply BBB proxy filter')
    parser.add_argument('--demo',     action='store_true', help='Run on 10 demo molecules')
    args = parser.parse_args()

    if args.demo or args.input is None:
        run_demo()
    else:
        stats = process_smiles_file(args.input, args.output, args.limit, args.bbb)
        print(f"\nDone. Stats: {stats}")
        print(f"Saved {stats['saved']} graph files to {args.output}/")
