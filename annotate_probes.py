"""
annotate_probes.py (v3)
========================
Uses mygene.info API to map probe IDs to gene symbols.
Works for all Affymetrix and Illumina platforms.
No downloads required — uses REST API.

Usage:
    py -3.11 annotate_probes.py
"""

import warnings; warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import urllib.request
import urllib.parse
import json
import time
from pathlib import Path

data_dir = Path('data')
expr_dir = Path('expression')

# ── mygene.info query ──────────────────────────────────────────────────────

def query_mygene(probe_ids, scopes, species='human', batch_size=500):
    """
    Query mygene.info to map probe IDs to gene symbols.
    scopes: comma-separated list of ID types to search
    e.g. 'reporter,affymetrix,illumina,alias'
    """
    base_url = 'https://mygene.info/v3/query'
    mapping = {}
    total = len(probe_ids)

    for i in range(0, total, batch_size):
        batch = probe_ids[i:i+batch_size]
        print(f"    Querying batch {i//batch_size+1}/"
              f"{(total+batch_size-1)//batch_size} "
              f"({len(batch)} probes)...")

        params = {
            'q':       ','.join(str(p) for p in batch),
            'scopes':  scopes,
            'fields':  'symbol,name',
            'species': species,
            'size':    batch_size,
        }
        data = urllib.parse.urlencode(params).encode()

        try:
            req = urllib.request.Request(
                base_url,
                data=data,
                method='POST',
                headers={
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'User-Agent': 'CausalDiffChem/1.0'
                }
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                results = json.loads(r.read().decode())

            for item in results:
                if 'notfound' not in item and 'symbol' in item:
                    query = item.get('query','')
                    symbol = item.get('symbol','')
                    if query and symbol:
                        mapping[query] = symbol

            time.sleep(0.3)  # be polite to the API

        except Exception as e:
            print(f"    Batch error: {e}")
            time.sleep(1)

    return mapping

def query_affymetrix(probe_ids, platform):
    """Map Affymetrix probe IDs using mygene.info."""
    print(f"  Querying mygene.info for {len(probe_ids)} Affymetrix probes...")
    # Try affymetrix scope first, then reporter
    mapping = query_mygene(probe_ids,
                           scopes='reporter,affymetrix,alias,symbol')
    print(f"  Mapped {len(mapping)} probes")
    return mapping

def query_illumina_numeric(probe_ids):
    """
    Map Illumina numeric Array_Address_Id probes.
    These are harder - try multiple approaches.
    """
    print(f"  Querying mygene.info for {len(probe_ids)} Illumina probes...")
    # Numeric Illumina IDs - try as reporter IDs
    mapping = query_mygene(probe_ids,
                           scopes='reporter,illumina,alias,symbol')
    print(f"  Initial mapping: {len(mapping)} probes")

    # For unmapped probes, try with ILMN_ prefix
    unmapped = [p for p in probe_ids if p not in mapping]
    if unmapped:
        print(f"  Trying {len(unmapped)} probes with ILMN_ prefix...")
        ilmn_ids = [f'ILMN_{p}' for p in unmapped]
        m2 = query_mygene(ilmn_ids,
                          scopes='reporter,illumina,alias,symbol')
        # Map back to original IDs
        for ilmn, gene in m2.items():
            orig = ilmn.replace('ILMN_','')
            mapping[orig] = gene
        print(f"  After ILMN_ prefix: {len(mapping)} total")

    return mapping

def update_scm_genes(disease, mapping_dict):
    genes_file = data_dir / f'{disease}_scm_genes.txt'
    if not genes_file.exists():
        return 0
    probes = open(genes_file).read().strip().split('\n')
    mapped = []
    for p in probes:
        g = mapping_dict.get(p.strip(), '')
        mapped.append(g if g not in ('','nan','---') else p.strip())
    open(genes_file,'w').write('\n'.join(mapped))
    n = sum(1 for p,g in zip(probes,mapped) if p.strip()!=g)
    print(f"    SCM: {n}/{len(probes)} annotated → {', '.join(mapped[:5])}")
    return n

def annotate_expression(disease, mapping_dict):
    for name in [f'{disease}_expression.csv',
                 f'{disease}_real_expression.csv']:
        p = expr_dir/name
        if p.exists():
            df = pd.read_csv(p)
            gene_cols = [c for c in df.columns if c != 'label']
            new_cols = {}
            for c in gene_cols:
                g = mapping_dict.get(c.strip(),'')
                if g not in ('','nan','---'):
                    new_cols[c] = g
            df = df.rename(columns=new_cols)
            # Deduplicate
            seen = {}; final = []
            for col in df.columns:
                if col=='label': final.append(col); continue
                if col in seen: seen[col]+=1; final.append(f"{col}_{seen[col]}")
                else: seen[col]=0; final.append(col)
            df.columns = final
            out = expr_dir/f'{disease}_annotated.csv'
            df.to_csv(out, index=False)
            n = len(new_cols)
            print(f"    Expression: {n}/{len(gene_cols)} annotated → {out.name}")
            return

# ── Disease configurations ─────────────────────────────────────────────────

DISEASE_CONFIGS = {
    'ad': {
        'platform': 'Affymetrix HG U219',
        'method': 'affymetrix',
    },
    'breast_cancer': {
        'platform': 'Affymetrix HG U133A',
        'method': 'affymetrix',
    },
    'breast_cancer2': {
        'platform': 'Affymetrix HG U133A',
        'method': 'affymetrix',
    },
    'dengue': {
        'platform': 'Affymetrix HG U133 Plus 2.0',
        'method': 'affymetrix',
    },
    'sarcoidosis2': {
        'platform': 'Illumina HumanHT-12',
        'method': 'illumina_numeric',
    },
}

# ── Main ───────────────────────────────────────────────────────────────────
print("="*60)
print("PROBE ANNOTATION v3 — using mygene.info API")
print("="*60)

results = {}

for disease, config in DISEASE_CONFIGS.items():
    print(f"\n{'─'*50}")
    print(f"Disease: {disease} ({config['platform']})")

    genes_file = data_dir / f'{disease}_scm_genes.txt'
    if not genes_file.exists():
        print(f"  No SCM genes file — skipping")
        continue

    # Get all probe IDs from SCM genes file
    scm_probes = open(genes_file).read().strip().split('\n')
    scm_probes = [p.strip() for p in scm_probes]

    # Also get probe IDs from expression file
    all_probes = set(scm_probes)
    for name in [f'{disease}_expression.csv', f'{disease}_real_expression.csv']:
        p = expr_dir/name
        if p.exists():
            df = pd.read_csv(p, nrows=0)
            expr_probes = [c for c in df.columns if c != 'label']
            all_probes.update(expr_probes[:200])  # sample 200 for speed
            break

    all_probes = list(all_probes)
    print(f"  Probes to annotate: {len(all_probes)} "
          f"(SCM: {len(scm_probes)}, expr sample: {len(all_probes)-len(scm_probes)})")
    print(f"  Sample probes: {scm_probes[:3]}")

    # Query mygene.info
    if config['method'] == 'affymetrix':
        mapping = query_affymetrix(all_probes, config['platform'])
    else:
        mapping = query_illumina_numeric(all_probes)

    if not mapping:
        print(f"  No mappings found — skipping")
        continue

    # Save mapping
    pd.DataFrame(list(mapping.items()),
                 columns=['probe_id','gene_symbol']
                 ).to_csv(data_dir/f'{disease}_probe_to_gene.csv', index=False)

    # Update SCM genes
    n = update_scm_genes(disease, mapping)
    # Annotate expression
    annotate_expression(disease, mapping)
    results[disease] = n

# ── Final summary ──────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("FINAL SUMMARY")
print(f"{'='*60}")
for f in sorted(data_dir.glob('*_scm_genes.txt')):
    disease = f.stem.replace('_scm_genes','')
    genes = open(f).read().strip().split('\n')
    top3 = ', '.join(genes[:3])
    status = 'ANNOTATED' if disease in results else 'pre-annotated'
    print(f"  {disease:22s} [{status}]: {top3}")

print("\nNext steps:")
print("  py -3.11 proper_notears.py --all")
print("  py -3.11 generate_candidates.py --all --mode all")
