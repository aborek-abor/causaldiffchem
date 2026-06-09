"""
annotate_ad_dengue.py
======================
Annotates AD probes from ADNI file and Dengue probes from GPL570 soft file.

Usage:
    py -3.11 annotate_ad_dengue.py
"""

import warnings; warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import gzip, io, urllib.request, urllib.parse, json, time
from pathlib import Path

data_dir = Path('data')
expr_dir = Path('expression')

def update_scm_genes(disease, mapping_dict):
    genes_file = data_dir / f'{disease}_scm_genes.txt'
    if not genes_file.exists():
        print(f"  No SCM genes file for {disease}"); return 0
    probes = open(genes_file).read().strip().split('\n')
    mapped = []
    for p in probes:
        g = mapping_dict.get(p.strip(),'')
        mapped.append(g if g not in ('','nan','---') else p.strip())
    open(genes_file,'w').write('\n'.join(mapped))
    n = sum(1 for p,g in zip(probes,mapped) if p.strip()!=g)
    print(f"  SCM: {n}/{len(probes)} annotated → {', '.join(mapped[:5])}")
    return n

def annotate_expression(disease, mapping_dict):
    for name in [f'{disease}_real_expression.csv', f'{disease}_expression.csv']:
        p = expr_dir/name
        if p.exists():
            df = pd.read_csv(p)
            gene_cols = [c for c in df.columns if c != 'label']
            new_cols = {c: mapping_dict.get(c.strip(),c)
                       for c in gene_cols
                       if mapping_dict.get(c.strip(),'') not in ('','nan','---')}
            df = df.rename(columns=new_cols)
            seen = {}; final = []
            for col in df.columns:
                if col=='label': final.append(col); continue
                if col in seen: seen[col]+=1; final.append(f"{col}_{seen[col]}")
                else: seen[col]=0; final.append(col)
            df.columns = final
            out = expr_dir/f'{disease}_annotated.csv'
            df.to_csv(out, index=False)
            n = len(new_cols)
            print(f"  Expression: {n}/{len(gene_cols)} annotated → {out.name}")
            return

# ── Step 1: AD annotation from ADNI CSV ───────────────────────────────────
print("="*60)
print("Step 1: AD — extracting probe mapping from ADNI CSV")
print("="*60)

print("Reading ADNI_Gene_Expression_Profile.csv (takes ~1 min)...")
raw = pd.read_csv('ADNI_Gene_Expression_Profile.csv',
                   header=0, low_memory=False, index_col=0)

probe_names  = list(raw.index[8:])
col_headers  = raw.iloc[7].tolist()

try:
    sym_col = col_headers.index('Symbol')
    symbols = raw.iloc[8:, sym_col].values
    ad_mapping = {}
    for probe, sym in zip(probe_names, symbols):
        s = str(sym).strip()
        if s not in ('nan','','---'):
            ad_mapping[probe] = s
    print(f"Mapped {len(ad_mapping)} AD probes to gene symbols")
    print(f"Sample: {list(ad_mapping.items())[:3]}")

    # Save mapping
    pd.DataFrame(list(ad_mapping.items()),
                 columns=['probe_id','gene_symbol']
                 ).to_csv(data_dir/'ad_probe_to_gene.csv', index=False)

    update_scm_genes('ad', ad_mapping)
    annotate_expression('ad', ad_mapping)

except Exception as e:
    print(f"AD annotation failed: {e}")

# ── Step 2: Dengue — download GPL570 soft file ────────────────────────────
print("\n"+"="*60)
print("Step 2: Dengue — downloading GPL570 annotation")
print("="*60)

dengue_mapping = {}

# Try soft file format (different from annot.gz)
soft_urls = [
    'https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL570nnn/GPL570/soft/GPL570_family.soft.gz',
    'https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL570nnn/GPL570/soft/GPL570.soft.gz',
]

for url in soft_urls:
    try:
        print(f"Trying: {url[:70]}...")
        req = urllib.request.Request(url,
              headers={'User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        print(f"Downloaded {len(data)//1024} KB")

        with gzip.open(io.BytesIO(data), 'rt',
                       encoding='utf-8', errors='replace') as f:
            lines = []
            in_table = False
            for line in f:
                if '!platform_table_begin' in line:
                    in_table = True; continue
                if '!platform_table_end' in line:
                    break
                if in_table:
                    lines.append(line)

        if lines:
            df = pd.read_csv(io.StringIO(''.join(lines)),
                            sep='\t', low_memory=False)
            print(f"Columns: {list(df.columns[:8])}")

            # Find gene symbol column
            gene_c = None
            for want in ['Gene Symbol','Gene symbol','Symbol','GENE_SYMBOL']:
                for col in df.columns:
                    if col.strip().lower() == want.lower():
                        gene_c = col; break
                if gene_c: break

            if gene_c:
                probe_c = df.columns[0]  # ID is always first
                mapping = dict(zip(
                    df[probe_c].astype(str).str.strip(),
                    df[gene_c].astype(str).str.strip()
                ))
                # Clean up
                dengue_mapping = {k:v.split('//')[0].strip()
                                  for k,v in mapping.items()
                                  if v not in ('nan','','---')}
                print(f"Mapped {len(dengue_mapping)} Dengue probes")
                break
    except Exception as e:
        print(f"Failed: {e}")

# If soft file fails, try mygene with stripped PM suffix
if not dengue_mapping:
    print("\nSoft file failed — trying mygene.info with stripped probe IDs...")
    genes_file = data_dir/'dengue_scm_genes.txt'
    probes = open(genes_file).read().strip().split('\n')

    # Strip _PM_ suffix: 224588_PM_at -> 224588_at
    stripped_map = {}
    for p in probes:
        stripped = p.replace('_PM_s_at','_s_at').replace('_PM_x_at','_x_at').replace('_PM_at','_at')
        stripped_map[stripped] = p  # stripped -> original

    params = {
        'q':      ','.join(stripped_map.keys()),
        'scopes': 'reporter,affymetrix,alias,symbol',
        'fields': 'symbol',
        'species':'human',
        'size':   len(stripped_map),
    }
    try:
        data = urllib.parse.urlencode(params).encode()
        req = urllib.request.Request('https://mygene.info/v3/query',
            data=data, method='POST',
            headers={'Content-Type':'application/x-www-form-urlencoded'})
        with urllib.request.urlopen(req, timeout=30) as r:
            results = json.loads(r.read())
        for item in results:
            if 'symbol' in item:
                orig = stripped_map.get(item['query'], item['query'])
                dengue_mapping[orig] = item['symbol']
        print(f"mygene mapped {len(dengue_mapping)} Dengue probes")
    except Exception as e:
        print(f"mygene failed: {e}")

if dengue_mapping:
    pd.DataFrame(list(dengue_mapping.items()),
                 columns=['probe_id','gene_symbol']
                 ).to_csv(data_dir/'dengue_probe_to_gene.csv', index=False)
    update_scm_genes('dengue', dengue_mapping)
    annotate_expression('dengue', dengue_mapping)
else:
    print("Could not annotate Dengue probes — will remain as probe IDs")

# ── Final summary ──────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("FINAL SUMMARY")
print(f"{'='*60}")
for f in sorted(data_dir.glob('*_scm_genes.txt')):
    disease = f.stem.replace('_scm_genes','')
    genes = open(f).read().strip().split('\n')
    top3 = ', '.join(genes[:3])
    print(f"  {disease:22s}: {top3}")
