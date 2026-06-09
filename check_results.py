import pandas as pd
from pathlib import Path

diseases = ['ms_blood','hiv','ad','tuberculosis','dengue',
            'sarcoidosis2','leishmaniasis2','malaria2',
            'breast_cancer','breast_cancer2']
modes = ['approveddrugs','clinical','preclinical','zinc']

for d in diseases:
    print(f"\n=== {d.upper()} ===")
    for m in modes:
        f = Path(f'results/{d}_{m}_top20.csv')
        if not f.exists():
            print(f"  [{m}] MISSING"); continue
        df = pd.read_csv(f).head(10)
        name_col  = 'name' if 'name' in df.columns else 'candidate_id'
        score_col = 'score' if 'score' in df.columns else 'R_score'
        print(f"  [{m}]")
        for _, row in df.iterrows():
            name    = str(row[name_col])[:28]
            bbb     = '+' if row['bbb_pass'] else '-'
            pathway = str(row.get('pathway_nodes','')).split(';')[0][:30]
            edges   = str(row.get('pathway_edges','')).split(';')[0][:35]
            print(f"    {name:28s} R={row['R_score']} Score={row[score_col]} "
                  f"BBB={bbb} MW={row['mw']} | Node: {pathway} | Edge: {edges}")
