"""Feature correlation heatmap on BTC segment 0 (training rows). Run from the repo root."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from src.experiments.common import btc_sessions, prepare_data

F = Path(__file__).resolve().parents[1] / "figures"
s = btc_sessions(4)[0]
d = prepare_data(s)
X = d.tab_feat[: d.train_end]
corr = np.corrcoef(X, rowvar=False)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"], "font.size": 7})
fig, ax = plt.subplots(figsize=(6.2, 5.2))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(d.feat_names)), d.feat_names, rotation=90, fontsize=6)
ax.set_yticks(range(len(d.feat_names)), d.feat_names, fontsize=6)
fig.colorbar(im, ax=ax, shrink=0.75, label="Pearson correlation")
ax.set_title("Correlation of the 27 features (BTC segment 0, training rows)")
fig.savefig(F / "feature_corr.pdf", bbox_inches="tight")
iu = np.triu_indices_from(corr, 1)
pairs = sorted(zip(np.abs(corr[iu]), np.array(d.feat_names)[iu[0]], np.array(d.feat_names)[iu[1]]), reverse=True)[:8]
for p in pairs: print(round(p[0], 3), p[1], p[2])
print("median |corr|", np.median(np.abs(corr[iu])))
