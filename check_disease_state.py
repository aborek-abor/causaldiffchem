import numpy as np
from pathlib import Path

ALL_DISEASES = [
    'ms_blood','hiv','ad','tuberculosis','dengue',
    'sarcoidosis2','leishmaniasis2','malaria2',
    'breast_cancer','breast_cancer2',
]

data_dir = Path('data')

for disease in ALL_DISEASES:
    v_dis_f = data_dir/f'v_disease_{disease}.npy'
    v_hea_f = data_dir/f'v_healthy_{disease}.npy'
    gene_f  = data_dir/f'{disease}_scm_genes.txt'

    if not v_dis_f.exists():
        print(f"\n{disease.upper()}: missing v_disease"); continue
    if not v_hea_f.exists():
        print(f"\n{disease.upper()}: missing v_healthy"); continue
    if not gene_f.exists():
        print(f"\n{disease.upper()}: missing scm_genes"); continue

    v_dis  = np.load(v_dis_f)
    v_hea  = np.load(v_hea_f)
    genes  = open(gene_f).read().strip().split('\n')
    genes  = [g.replace('SEX_CONFOUNDER_','')+'*'
              if g.startswith('SEX_CONFOUNDER_') else g
              for g in genes]

    n = min(len(genes), len(v_dis), len(v_hea), 20)

    print(f"\n{'='*65}")
    print(f"DISEASE: {disease.upper()}")
    print(f"{'='*65}")
    print(f"{'Gene':<22} {'Disease':>10} {'Healthy':>10} {'Diff':>9} {'Direction'}")
    print(f"{'-'*65}")

    # Sort by absolute difference
    diffs = [(genes[i], float(v_dis[i]), float(v_hea[i]),
              float(v_dis[i]-v_hea[i])) for i in range(n)]
    diffs.sort(key=lambda x: abs(x[3]), reverse=True)

    for gene, d, h, diff in diffs:
        direction = 'UP in disease' if diff > 0 else 'DOWN in disease'
        print(f"{gene:<22} {d:>10.3f} {h:>10.3f} {diff:>9.3f}  {direction}")
