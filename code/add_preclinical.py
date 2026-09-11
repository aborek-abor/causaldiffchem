"""
add_preclinical.py
==================
Processes the preclinical SDF file and adds it to drug_library_index.json

Usage:
    py -3.11 add_preclinical.py
"""

import warnings; warnings.filterwarnings('ignore')
import json
import numpy as np
from pathlib import Path
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, QED, AllChem
from rdkit.Chem.rdMolDescriptors import CalcTPSA, CalcNumRotatableBonds
import joblib

RDLogger.DisableLog('rdApp.*')

# ── Load existing index ────────────────────────────────────────────────────
idx_path = Path('drug_library_index.json')
existing = json.load(open(idx_path))
print(f"Existing index: {len(existing)} entries")

# Find next file_id
max_id = max(e['file_id'] for e in existing)
print(f"Max existing file_id: {max_id}")

# ── Find preclinical SDF ───────────────────────────────────────────────────
sdf_files = list(Path('.').glob('*Preclinical*.sdf')) + \
            list(Path('.').glob('*preclinical*.sdf')) + \
            list(Path('.').glob('L3410*.sdf'))

if not sdf_files:
    print("ERROR: Preclinical SDF not found")
    exit(1)

sdf_path = sdf_files[0]
print(f"Found: {sdf_path.name}")

# ── Load BBB model ─────────────────────────────────────────────────────────
bbb_bundle = joblib.load('logbb_rf.pkl') if Path('logbb_rf.pkl').exists() else None

def predict_bbb_simple(mol):
    logp = Descriptors.MolLogP(mol)
    mw   = Descriptors.MolWt(mol)
    tpsa = CalcTPSA(mol)
    rotb = CalcNumRotatableBonds(mol)
    hbd  = Descriptors.NumHDonors(mol)
    return (1<=logp<=4 and mw<450 and tpsa<90 and rotb<8 and hbd<=3)

# ── Parse SDF ─────────────────────────────────────────────────────────────
print("Parsing preclinical SDF...")
supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=True)

new_entries = []
file_id = max_id + 1
skipped = 0

for mol in supplier:
    if mol is None:
        skipped += 1
        continue
    try:
        # Get molecule name
        name = mol.GetProp('_Name') if mol.HasProp('_Name') else ''
        if not name or name.strip() == '':
            # Try other name fields
            for prop in ['Name', 'Compound Name', 'GENERIC_NAME', 'name']:
                if mol.HasProp(prop):
                    name = mol.GetProp(prop)
                    break
        if not name:
            name = f'Preclinical-{file_id}'

        # Get other properties
        disease_ind = ''
        target = ''
        for prop in mol.GetPropsAsDict():
            pval = str(mol.GetProp(prop))
            prop_lower = prop.lower()
            if any(x in prop_lower for x in ['disease', 'indication', 'therapeutic']):
                disease_ind = pval[:50]
            if any(x in prop_lower for x in ['target', 'mechanism', 'pathway']):
                target = pval[:50]

        smi = Chem.MolToSmiles(mol)
        mw   = round(Descriptors.MolWt(mol), 1)
        logp = round(Descriptors.MolLogP(mol), 3)
        qed  = round(QED.qed(mol), 3)
        bbb  = predict_bbb_simple(mol)

        entry = {
            'file_id':  file_id,
            'library':  'preclinical_compounds',
            'smiles':   smi,
            'name':     name.strip(),
            'disease':  disease_ind,
            'target':   target,
            'mw':       mw,
            'qed':      qed,
            'logp':     logp,
            'bbb_pass': bbb,
        }
        new_entries.append(entry)
        file_id += 1

    except Exception as e:
        skipped += 1
        continue

print(f"Parsed {len(new_entries)} preclinical compounds ({skipped} skipped)")

# Show sample
print("\nSample entries:")
for e in new_entries[:3]:
    print(f"  {e['name']:30s} MW={e['mw']:6.1f} QED={e['qed']:.3f} "
          f"BBB={'Y' if e['bbb_pass'] else 'N'} {e['disease'][:30]}")

# ── Save graphs for preclinical compounds ─────────────────────────────────
print("\nSaving molecular graphs...")
graphs_dir = Path('graphs')
graphs_dir.mkdir(exist_ok=True)

supplier2 = Chem.SDMolSupplier(str(sdf_path), removeHs=True)
saved = 0

for i, (mol, entry) in enumerate(zip(supplier2, new_entries)):
    if mol is None:
        continue
    try:
        from rdkit.Chem import rdmolops
        adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
        atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
        fid = entry['file_id']
        np.savez_compressed(
            graphs_dir / f'mol_{fid:06d}.npz',
            smiles=np.array([entry['smiles']]),
            atoms=np.array(atoms),
            adj=adj,
            mw=np.array([entry['mw']]),
            qed=np.array([entry['qed']]),
            logp=np.array([entry['logp']]),
            bbb_pass=np.array([entry['bbb_pass']]),
            source=np.array(['preclinical_compounds']),
        )
        saved += 1
    except:
        continue

print(f"Saved {saved} graph files to graphs/")

# ── Update index ───────────────────────────────────────────────────────────
updated_index = existing + new_entries
json.dump(updated_index, open(idx_path, 'w'), indent=2)
print(f"\nUpdated drug_library_index.json:")
from collections import Counter
libs = Counter(e['library'] for e in updated_index)
for lib, n in libs.items():
    print(f"  {lib}: {n}")
print(f"  TOTAL: {len(updated_index)}")
print("\nDone. Now run:")
print("  py -3.11 generate_candidates.py --disease ms_blood --mode drugs --top_k 20")
