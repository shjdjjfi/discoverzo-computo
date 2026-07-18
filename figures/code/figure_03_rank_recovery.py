from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'output'
OUT.mkdir(parents=True, exist_ok=True)
df=pd.read_csv(DATA/'figure_03_rank_recovery.csv')
fig,ax=plt.subplots(figsize=(6.6,4.3))
for r,g in df.groupby('r'):
 ax.plot(g.d,g.rank_recovery,marker='o',label=f'r={r}')
ax.set_ylim(-.03,1.03); ax.set_xlabel('Ambient dimension d'); ax.set_ylabel('Exact rank-recovery frequency'); ax.set_title('Adaptive rank selection at N=20d anchors'); ax.legend(); ax.grid(True,alpha=.25); fig.tight_layout(); fig.savefig(OUT/'figure_03_rank_recovery.png',dpi=180)
