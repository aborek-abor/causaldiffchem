"""
process_drug_libraries.py
==========================
Run this from your causaldiffchem/ folder.
It reads the 3 SDF files you uploaded and converts them
directly into graph .npz files in your graphs/ folder.

Usage:
    py -3.11 process_drug_libraries.py

Requires:
    - L4200-Targetmol-FDA-approved_Drug_Library-1729cpds__1_.sdf
    - L3400-Targetmol-Clinical_Compound_Library-3480cpds.sdf
    - L1000-Targetmol-Approved_Drug_Library-2863cpds.sdf
    All three SDF files must be in the same folder as this script.

Output:
    - Adds graph .npz files to graphs/ folder
    - Saves drug_library_index.json with metadata
"""

import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, QED
from rdkit.Chem.rdMolDescriptors import CalcTPSA, CalcNumRotatableBonds
import numpy as np
from pathlib import Path
import json

RDLogger.DisableLog('rdApp.*')

# ── Atom/bond feature functions ────────────────────────────────────────────

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

def atom_feat(atom):
    s = atom.GetSymbol()
    t = ['C', 'N', 'O', 'S', 'F']
    oh = [int(s == x) for x in t] + [int(s not in t)]
    h = atom.GetHybridization()
    H = Chem.rdchem.HybridizationType
    return np.array(
        (oh + [min(atom.GetDegree(), 4) / 4.0] +
         [int(h == H.SP), int(h == H.SP2), int(h == H.SP3)])[:9],
        dtype=np.float32
    )

def bond_feat(b):
    bt = b.GetBondType()
    return np.array([int(bt == t) for t in BOND_TYPES], dtype=np.float32)

def bbb_ok(m):
    return (Descriptors.MolWt(m) < 450 and
            1 <= Descriptors.MolLogP(m) <= 4 and
            CalcTPSA(m) < 90 and
            CalcNumRotatableBonds(m) < 8)

# ── Setup paths ────────────────────────────────────────────────────────────

graph_dir = Path('graphs')
graph_dir.mkdir(exist_ok=True)

# Find how many graphs already exist
existing = len(list(graph_dir.glob('mol_*.npz')))
print(f"Existing graphs in graphs/: {existing:,}")

# SDF library files
libraries = {
    'fda_approved':       'L4200-Targetmol-FDA-approved_Drug_Library-1729cpds__1_.sdf',
    'approved_drugs':     'L1000-Targetmol-Approved_Drug_Library-2863cpds.sdf',
    'clinical_compounds': 'L3400-Targetmol-Clinical_Compound_Library-3480cpds.sdf',
}

# Check files exist
print("\nChecking SDF files:")
missing = []
for name, path in libraries.items():
    if Path(path).exists():
        size = Path(path).stat().st_size / 1e6
        print(f"  {name}: {path} ({size:.1f} MB) ✓")
    else:
        print(f"  {name}: {path} — NOT FOUND ✗")
        missing.append(path)

if missing:
    print(f"\nERROR: {len(missing)} SDF file(s) not found.")
    print("Make sure these files are in your causaldiffchem/ folder:")
    for f in missing:
        print(f"  {f}")
    exit(1)

# ── Process each library ───────────────────────────────────────────────────

all_metadata = []
total_added = 0

for lib_name, sdf_path in libraries.items():
    print(f"\nProcessing {lib_name}...")
    suppl = Chem.SDMolSupplier(sdf_path, sanitize=True, removeHs=True)

    lib_saved = 0
    lib_skipped = 0

    for mol in suppl:
        if mol is None:
            lib_skipped += 1
            continue

        try:
            # Get properties
            props = {}
            for key in ['Name', 'ID', 'CAS', 'Disease', 'Target',
                        'Indication', 'Approved status']:
                try:
                    props[key] = mol.GetProp(key) if mol.HasProp(key) else ''
                except Exception:
                    props[key] = ''

            # Build graph
            x = np.stack([atom_feat(a) for a in mol.GetAtoms()])
            rows, cols, ef = [], [], []
            for b in mol.GetBonds():
                i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                fe = bond_feat(b)
                rows += [i, j]
                cols += [j, i]
                ef   += [fe, fe]
            if not rows:
                rows, cols, ef = [0], [0], [np.zeros(4, dtype=np.float32)]

            # Properties
            qed  = QED.qed(mol)
            mw   = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = CalcTPSA(mol)
            bbb  = bbb_ok(mol)
            smi  = Chem.MolToSmiles(mol)

            # Save graph
            idx   = existing + total_added
            fname = graph_dir / f'mol_{idx:07d}.npz'
            np.savez_compressed(
                fname,
                x          = x,
                edge_index = np.array([rows, cols], dtype=np.int64),
                edge_attr  = np.stack(ef),
                smiles     = np.array([smi]),
                qed        = np.array([qed]),
                mw         = np.array([mw]),
                logp       = np.array([logp]),
                bbb_pass   = np.array([bbb]),
            )

            # Metadata
            all_metadata.append({
                'file_id':  idx,
                'library':  lib_name,
                'filename': fname.name,
                'smiles':   smi,
                'name':     props.get('Name', ''),
                'id':       props.get('ID', ''),
                'disease':  props.get('Disease', ''),
                'target':   props.get('Target', ''),
                'mw':       round(mw, 1),
                'qed':      round(qed, 3),
                'logp':     round(logp, 2),
                'bbb_pass': bool(bbb),
            })

            total_added += 1
            lib_saved   += 1

        except Exception:
            lib_skipped += 1
            continue

    print(f"  Saved:   {lib_saved:,}")
    print(f"  Skipped: {lib_skipped:,}")

# ── Save metadata index ────────────────────────────────────────────────────

index_path = Path('drug_library_index.json')
with open(index_path, 'w') as f:
    json.dump(all_metadata, f, indent=2)

# ── Summary ────────────────────────────────────────────────────────────────

bbb_count = sum(1 for m in all_metadata if m['bbb_pass'])
cns_count = sum(1 for m in all_metadata if any(
    x in m['disease'].lower()
    for x in ['nervous', 'neuro', 'brain', 'alzheimer', 'psychiatric']
))

from collections import Counter
lib_counts = Counter(m['library'] for m in all_metadata)

print(f"\n{'='*55}")
print(f"DRUG LIBRARY PROCESSING COMPLETE")
print(f"{'='*55}")
print(f"  FDA Approved:        {lib_counts['fda_approved']:,}")
print(f"  Approved Drugs:      {lib_counts['approved_drugs']:,}")
print(f"  Clinical Compounds:  {lib_counts['clinical_compounds']:,}")
print(f"  Total added:         {total_added:,}")
print(f"  BBB-permissive:      {bbb_count:,} ({100*bbb_count/max(total_added,1):.1f}%)")
print(f"  CNS-relevant:        {cns_count:,}")
print(f"\nTotal graphs in graphs/: {existing + total_added:,}")
print(f"Metadata saved to:       drug_library_index.json")

print(f"\nSample drugs added:")
bbb_drugs = [m for m in all_metadata if m['bbb_pass']]
for m in bbb_drugs[:8]:
    print(f"  {m['name']:25s}  QED={m['qed']:.2f}  "
          f"MW={m['mw']:.0f}  {m['disease'][:35]}")
