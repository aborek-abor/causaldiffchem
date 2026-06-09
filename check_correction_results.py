import pandas as pd
from pathlib import Path

diseases = ['ms_blood','hiv','ad','tuberculosis','dengue',
            'sarcoidosis2','leishmaniasis2','malaria2',
            'breast_cancer','breast_cancer2']
modes = ['approveddrugs','clinical','preclinical','zinc']

for d in diseases:
    print(f"\n=== {d.upper()} ===")
    for m in modes:
        f = Path(f'results/{d}_{m}_correction_top20.csv')
        if not f.exists():
            print(f"  [{m}] MISSING"); continue
        df = pd.read_csv(f).head(10)
        name_col = 'name' if 'name' in df.columns else 'candidate_id'
        print(f"  [{m}]")
        for _, row in df.iterrows():
            name  = str(row[name_col])[:28]
            bbb   = '+' if row['bbb_pass'] else '-'
            nodes = str(row.get('nodes_toward_H','')).split(';')[0][:28]
            edge  = str(row.get('top_causal_edge',''))[:35]
            print(f"    {name:28s} NetCorr={row['net_correction']} "
                  f"R={row['R_score']} %H={row['pct_nodes_H']} "
                  f"BBB={bbb} MW={row['mw']} | {nodes} | {edge}")
