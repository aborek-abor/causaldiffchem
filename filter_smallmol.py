import pandas as pd
from pathlib import Path

diseases = ['tuberculosis','malaria2','dengue','leishmaniasis2',
            'sarcoidosis2','breast_cancer','breast_cancer2']
modes    = ['approveddrugs','clinical']

for d in diseases:
    for m in modes:
        f = Path(f'results/{d}_{m}_top20.csv')
        if not f.exists():
            continue
        df = pd.read_csv(f)
        # Small molecules only: MW <= 900, QED >= 0.1, BBB pass or non-neuro
        sm = df[(df['mw'] <= 900) & (df['qed'] >= 0.1)].head(10)
        print(f"\n=== {d.upper()} | {m} | small molecules (MW<=900, QED>=0.1) ===")
        for _, r in sm.iterrows():
            name = str(r['name'])[:28]
            print(f"  {name:28s} R={r['R_score']} Score={r['score']} "
                  f"MW={r['mw']} QED={r['qed']} BBB={'+' if r['bbb_pass'] else '-'} "
                  f"logBB={r['logbb']}")
