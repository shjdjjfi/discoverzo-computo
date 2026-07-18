from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'output'
OUT.mkdir(parents=True, exist_ok=True)
df=pd.read_csv(DATA/'figure_05_optimization_heatmap.csv')
order=['oracle_subspace','cross_block_known','cross_block_adaptive','cross_sparse_known','random_subspace','full_zo']; names=['sphere','ellipsoid','rosenbrock','rastrigin']
p=df.pivot(index='benchmark',columns='method',values='median_regret'); mat=np.log10(p.reindex(index=names,columns=order).to_numpy())
fig,ax=plt.subplots(figsize=(9,4.2)); im=ax.imshow(mat,aspect='auto'); ax.set_xticks(np.arange(len(order)),order,rotation=30,ha='right'); ax.set_yticks(np.arange(len(names)),names); ax.set_title('Median log10 regret on exact embedded benchmarks'); fig.colorbar(im,ax=ax,label='log10 regret'); fig.tight_layout(); fig.savefig(OUT/'figure_05_optimization_heatmap.png',dpi=180)
