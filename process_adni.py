"""
process_adni.py (v5) - Fixed: uses all visits for label lookup
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize

data_dir = Path('data')
expr_dir = Path('expression')
data_dir.mkdir(exist_ok=True)
expr_dir.mkdir(exist_ok=True)

print("Loading diagnosis from DXSUM...")
diag_df = pd.read_csv('DXSUM_03Jun2026.csv')

# Build label per subject using ALL visits
# Priority: if ever AD -> disease, elif ever CN -> control
ptid_label = {}
for ptid, grp in diag_df[diag_df['DIAGNOSIS'].isin([1.0, 3.0])].groupby('PTID'):
    diags = grp['DIAGNOSIS'].tolist()
    if 3.0 in diags:
        ptid_label[str(ptid).strip()] = 'disease'
    elif 1.0 in diags:
        ptid_label[str(ptid).strip()] = 'control'

cn = sum(1 for v in ptid_label.values() if v=='control')
ad = sum(1 for v in ptid_label.values() if v=='disease')
print(f"Total subjects — CN: {cn}  AD: {ad}")

print("\nLoading expression matrix (1-2 min)...")
raw = pd.read_csv('ADNI_Gene_Expression_Profile.csv',
                   header=0, low_memory=False, index_col=0)
print(f"Shape: {raw.shape}")

sample_ids = raw.iloc[1].astype(str).str.strip().values
sample_ids_clean = ['UNKNOWN' if s in ['nan',''] else s for s in sample_ids]

probe_data = raw.iloc[7:].copy()
probe_names = list(probe_data.index)
print(f"Probes: {len(probe_names)}  Samples: {len(sample_ids_clean)}")
print("Converting to numeric...")
expr_matrix = probe_data.apply(pd.to_numeric, errors='coerce').fillna(0)

print("\nMatching IDs...")
labels = [ptid_label.get(sid, 'exclude') for sid in sample_ids_clean]
n_ad = labels.count('disease')
n_cn = labels.count('control')
print(f"Matched — AD: {n_ad}  CN: {n_cn}")

keep_cols = [i for i,l in enumerate(labels) if l in ['disease','control']]
keep_labels = [labels[i] for i in keep_cols]

expr_kept = expr_matrix.iloc[:, keep_cols].T.values
disease_idx = [i for i,l in enumerate(keep_labels) if l=='disease']
control_idx = [i for i,l in enumerate(keep_labels) if l=='control']
disease_arr = expr_kept[disease_idx]
control_arr = expr_kept[control_idx]
print(f"Disease matrix: {disease_arr.shape}")
print(f"Control matrix: {control_arr.shape}")

# Top 500 variable probes
var = disease_arr.var(0)
top_idx = np.argsort(var)[::-1][:500]
top_probes = [probe_names[i] for i in top_idx]
disease_top = disease_arr[:, top_idx]
control_top = control_arr[:, top_idx]

d_rows = pd.DataFrame(disease_top, columns=top_probes)
d_rows['label'] = 'disease'
c_rows = pd.DataFrame(control_top, columns=top_probes)
c_rows['label'] = 'control'
out_df = pd.concat([d_rows, c_rows], ignore_index=True)
out_df.to_csv(expr_dir/'ad_real_expression.csv', index=False)
print(f"Saved expression/ad_real_expression.csv  {out_df.shape}")

v_disease = disease_top.mean(0)
v_healthy = control_top.mean(0)
delta_v   = v_healthy - v_disease
norm = np.linalg.norm(delta_v)
delta_v_norm = delta_v / norm if norm > 0 else delta_v

def notears_fast(X, lambda1=0.03, max_iter=300, h_tol=1e-7):
    n,d = X.shape
    X = (X-X.mean(0))/(X.std(0)+1e-8)
    W = np.zeros((d,d)); rho,alpha,h_prev = 1.0,0.0,np.inf
    def h(W): return float(np.sum(W*W))
    def obj(wf):
        W_=wf.reshape(d,d); np.fill_diagonal(W_,0)
        return 0.5/n*np.sum((X-X@W_)**2)+0.5*rho*h(W_)**2+alpha*h(W_)+lambda1*np.sum(np.abs(W_))
    for it in range(max_iter):
        res=minimize(obj,W.flatten(),method='L-BFGS-B',options={'maxiter':200,'ftol':1e-14})
        W=res.x.reshape(d,d); np.fill_diagonal(W,0); hn=h(W)
        if it%50==0: print(f"  iter {it}: h={hn:.4f}  loss={res.fun:.4f}")
        if hn>0.25*h_prev: rho=min(rho*5,1e10)
        alpha+=rho*hn
        if hn<=h_tol:
            print(f"  Converged at iter {it}  h={hn:.2e}")
            break
        h_prev=hn
    return W

top20_idx = np.argsort(var)[::-1][:20]
print(f"\nRunning NOTEARS on top 20 genes ({len(disease_idx)} disease samples)...")
W = notears_fast(disease_arr[:, top20_idx])
n_edges = int(np.sum(np.abs(W) > 0.05))
print(f"SCM: 20 nodes, {n_edges} edges")

np.save(data_dir/'W_ad_real.npy',         W)
np.save(data_dir/'delta_v_ad_real.npy',   delta_v_norm)
np.save(data_dir/'v_disease_ad_real.npy', v_disease)
np.save(data_dir/'v_healthy_ad_real.npy', v_healthy)

top5_up   = [top_probes[i] for i in np.argsort(delta_v)[::-1][:5]]
top5_down = [top_probes[i] for i in np.argsort(delta_v)[:5]]

print(f"\n{'='*60}")
print(f"REAL ADNI DATA PROCESSED")
print(f"{'='*60}")
print(f"  AD samples:  {len(disease_idx)}")
print(f"  CN samples:  {len(control_idx)}")
print(f"  Probes:      500")
print(f"  SCM edges:   {n_edges}")
print(f"  Top UP:      {', '.join(top5_up[:3])}")
print(f"  Top DOWN:    {', '.join(top5_down[:3])}")
print(f"\nSwitch to real data and retrain:")
print(f"  copy data\\W_ad_real.npy data\\W_ad.npy")
print(f"  copy data\\delta_v_ad_real.npy data\\delta_v_ad.npy")
print(f"  copy data\\v_disease_ad_real.npy data\\v_disease_ad.npy")
print(f"  copy data\\v_healthy_ad_real.npy data\\v_healthy_ad.npy")
print(f"  py -3.11 train_causal.py --disease ad --epochs 150 --batch_size 128")
