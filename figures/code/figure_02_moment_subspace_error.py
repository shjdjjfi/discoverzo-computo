from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'output'
OUT.mkdir(parents=True, exist_ok=True)
df=pd.read_csv(DATA/'figure_02_moment_subspace_error.csv')
fig,ax=plt.subplots(figsize=(6.6,4.3))
for method,g in df.groupby('method'):
 g=g.sort_values('n_anchors'); ax.plot(g.n_anchors,g['mean'],marker='o',label=method)
ax.set_xscale('log',base=2); ax.set_yscale('log'); ax.set_xlabel('Number of anchors'); ax.set_ylabel('Principal-angle error'); ax.set_title('Finite-sample eigenvector recovery'); ax.legend(); ax.grid(True,which='both',alpha=.25); fig.tight_layout(); fig.savefig(OUT/'figure_02_moment_subspace_error.png',dpi=180)
