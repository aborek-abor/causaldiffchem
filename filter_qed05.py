import pandas as pd
from pathlib import Path

diseases = ['tuberculosis', 'malaria2']
modes    = ['approveddrugs', 'clinical']

for d in diseases:
    for m in modes:
        f = Path(f'results/{d}_{m}_top20.csv')
        if not f.exists():
            print(f"MISSING: {f}"); continue
        df = pd.read_csv(f)
        clean = df[(df['mw'] <= 900) & (df['qed'] >= 0.5)].head(10)
        print(f"\n=== {d.upper()} | {m} | MW<=900 & QED>=0.5 ===")
        for _, r in clean.iterrows():
            name = str(r['name'])[:30]
            bbb  = '+' if r['bbb_pass'] else '-'
            print(f"  {name:30s} R={r['R_score']} Score={r['score']} "
                  f"MW={r['mw']} QED={r['qed']} BBB={bbb} logBB={r['logbb']}")
