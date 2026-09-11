"""
fix_expression_normalisation.py
================================
Fixes all disease expression files by z-score normalising them.
Detects raw intensities AND log2-normalised-but-not-centred data.

A properly z-scored v_disease vector has |mean| < 1.5.
Log2 normalised (not z-scored) has mean ~ 4-8.
Raw intensities have mean > 100.

Usage:
    py -3.11 fix_expression_normalisation.py
    py -3.11 fix_expression_normalisation.py --dry_run
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--dry_run', action='store_true')
args = parser.parse_args()

DATA_DIR = Path('data')
EXPR_DIR = Path('expression')

# Properly z-scored data has |mean| < 1.5
# Log2 normalised has mean ~ 4-8, raw has mean > 100
Z_SCORE_MEAN_THRESHOLD = 1.5

diseases = sorted(set(
    f.stem.replace('v_disease_', '')
    for f in DATA_DIR.glob('v_disease_*.npy')
))

print(f"Found {len(diseases)} disease vectors to check\n")
print(f"  {'Disease':22s}  {'Status':12s}  {'Mean':>8s}  {'Max':>8s}  Action")
print(f"  {'-'*75}")

to_fix     = []
already_ok = []
no_file    = []

for disease in diseases:
    v_dis_file = DATA_DIR / f'v_disease_{disease}.npy'
    if not v_dis_file.exists():
        continue

    v_dis    = np.load(v_dis_file)
    mean_val = float(np.abs(v_dis).mean())
    max_val  = float(np.abs(v_dis).max())

    needs_fix = mean_val > Z_SCORE_MEAN_THRESHOLD

    # Prefer annotated CSV (gene symbols) over probe-ID CSV
    annotated = EXPR_DIR / f'{disease}_annotated.csv'
    standard  = EXPR_DIR / f'{disease}_expression.csv'

    if annotated.exists():
        src_file  = annotated
        src_label = 'annotated'
    elif standard.exists():
        src_file  = standard
        src_label = 'expression'
    else:
        src_file  = None
        src_label = 'MISSING'

    if mean_val > 100:
        status = 'RAW'
    elif needs_fix:
        status = 'LOG2'
    else:
        status = 'OK'

    action = f'fix from {src_label}' if needs_fix and src_file else \
             'no source file'        if needs_fix else 'skip'

    print(f"  {disease:22s}  {status:12s}  {mean_val:8.2f}  {max_val:8.2f}  {action}")

    if needs_fix:
        if src_file:
            to_fix.append((disease, src_file))
        else:
            no_file.append(disease)
    else:
        already_ok.append(disease)

print()
print(f"  Summary: {len(already_ok)} already OK  |  "
      f"{len(to_fix)} need fixing  |  "
      f"{len(no_file)} missing source file")

if no_file:
    print(f"\n  WARNING: no source file for: {no_file}")

if args.dry_run:
    print("\n[DRY RUN] No changes made.")
    raise SystemExit(0)

if not to_fix:
    print("\nAll expression vectors correctly normalised. Nothing to do.")
    raise SystemExit(0)

# ── Fix each disease ──────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("FIXING")
print(f"{'='*65}")

fixed  = []
failed = []

for disease, src_file in to_fix:
    print(f"\n  [{disease}]  source: {src_file.name}")
    try:
        df = pd.read_csv(src_file)

        drop_cols = {'label', 'Unnamed: 0', 'disease_subtype'}
        gene_cols = [c for c in df.columns if c not in drop_cols]

        if 'label' not in df.columns:
            print(f"    ERROR: no label column — skipping")
            failed.append(disease)
            continue

        label = df['label'].values
        expr  = df[gene_cols].astype(float)

        before_mean = expr.values.mean()
        before_std  = expr.values.std()

        # Z-score per gene across all samples
        expr_norm = (expr - expr.mean()) / expr.std().replace(0, 1)

        after_mean = expr_norm.values.mean()
        after_std  = expr_norm.values.std()

        print(f"    Before: mean={before_mean:.2f}  std={before_std:.2f}")
        print(f"    After : mean={after_mean:.4f}  std={after_std:.4f}")

        # Save normalised expression file
        target = EXPR_DIR / f'{disease}_expression.csv'
        df_out = expr_norm.copy()
        if 'Unnamed: 0' in df.columns:
            df_out.insert(0, 'Unnamed: 0', df['Unnamed: 0'])
        if 'disease_subtype' in df.columns:
            df_out['disease_subtype'] = df['disease_subtype']
        df_out['label'] = label
        df_out.to_csv(target, index=False)
        print(f"    Saved: {target}")

        # Regenerate v_disease and v_healthy from SCM gene list
        scm_file = DATA_DIR / f'{disease}_scm_genes.txt'
        if not scm_file.exists():
            print(f"    WARNING: no SCM gene list — v_disease/v_healthy not updated")
            fixed.append(disease)
            continue

        scm_genes = [
            g.replace('SEX_CONFOUNDER_', '') if g.startswith('SEX_CONFOUNDER_') else g
            for g in scm_file.read_text().strip().split('\n')
        ]

        available = set(expr_norm.columns)
        matched   = [g for g in scm_genes if g in available]
        missing   = [g for g in scm_genes if g not in available]

        if missing:
            print(f"    WARNING: {len(missing)} SCM genes not found: {missing[:5]}")
        if not matched:
            print(f"    ERROR: no SCM genes matched — cannot update vectors")
            failed.append(disease)
            continue

        dis_mask  = label == 'disease'
        hea_mask  = label == 'control'
        expr_scm  = expr_norm[matched]

        v_dis_new = expr_scm[dis_mask].mean(axis=0).values.astype(np.float32)
        v_hea_new = expr_scm[hea_mask].mean(axis=0).values.astype(np.float32)

        # Pad to 128 dims to match delta_v format
        if len(v_dis_new) < 128:
            v_dis_new = np.pad(v_dis_new, (0, 128 - len(v_dis_new)))
            v_hea_new = np.pad(v_hea_new, (0, 128 - len(v_hea_new)))

        np.save(DATA_DIR / f'v_disease_{disease}.npy', v_dis_new)
        np.save(DATA_DIR / f'v_healthy_{disease}.npy', v_hea_new)

        gap = v_hea_new[:len(matched)] - v_dis_new[:len(matched)]
        print(f"    v_disease/v_healthy updated — "
              f"gap range: {gap.min():+.3f} to {gap.max():+.3f}  "
              f"mean|gap|: {np.abs(gap).mean():.3f}")

        fixed.append(disease)

    except Exception as e:
        print(f"    ERROR: {e}")
        failed.append(disease)

# ── Report ────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("COMPLETE")
print(f"{'='*65}")
print(f"Fixed  : {len(fixed)}   — {fixed}")
if failed:
    print(f"Failed : {len(failed)}   — {failed}")

print(f"""
Next steps for each fixed disease:

  1. Rerun NOTEARS (rebuilds W matrix on z-scored data):
       py -3.11 proper_notears.py --disease <name>

  2. Retrain model:
       py -3.11 train_causal.py --disease <name> --epochs 150 --batch_size 64

  3. Rescore everything once all diseases are retrained:
       py -3.11 score_by_correction.py --all --mode all --top_k 20 --n_zinc 1000
""")
