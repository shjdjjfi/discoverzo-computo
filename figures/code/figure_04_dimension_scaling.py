from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'output'
OUT.mkdir(parents=True, exist_ok=True)
df=pd.read_csv(DATA/'figure_04_dimension_scaling.csv')
fig,ax=plt.subplots(figsize=(6.6,4.3))
for r,g in df.groupby('r'):
 ax.plot(g.d,g['mean'],marker='o',label=f'r={r}')
ax.set_xlabel('Ambient dimension d'); ax.set_ylabel('Mean principal-angle error'); ax.set_title('Discovery error under linear anchor scaling'); ax.legend(); ax.grid(True,alpha=.25); fig.tight_layout(); fig.savefig(OUT/'figure_04_dimension_scaling.png',dpi=180)
