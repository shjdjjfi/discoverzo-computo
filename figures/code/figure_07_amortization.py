from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'output'
OUT.mkdir(parents=True, exist_ok=True)
df=pd.read_csv(DATA/'figure_07_amortization.csv')
fig,ax=plt.subplots(figsize=(6.8,4.3))
for method,g in df.groupby('method'):
 ax.plot(g.tasks,g.median_mean_regret,marker='o',label=method)
ax.set_yscale('log'); ax.set_xlabel('Number of related optimization tasks'); ax.set_ylabel('Median mean regret'); ax.set_title('Amortizing one-time structure discovery'); ax.legend(fontsize=8); ax.grid(True,which='both',alpha=.25); fig.tight_layout(); fig.savefig(OUT/'figure_07_amortization.png',dpi=180)
