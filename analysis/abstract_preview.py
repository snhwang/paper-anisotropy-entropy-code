"""Preview figure for the ISMRM 2027 abstract: FA against the eigenvalue entropy, one panel, large type.

ISMRM asks for one preview figure, with no caption, legible at the width of a phone. This draws the
point of Figure 2 in a single panel. At order 2 every tissue voxel lies on the curve
H_2 = ln(3 - 2 FA^2). At order 1 (Shannon) the voxels fill a band between the linear and planar
tensors.

Data and selection are those of fa_vs_eigen_entropy.py: the four HCP-Aging subjects (first four rows
of the b = 1500 manifest), cached b = 1500 eigenvalues (b1500_tensor.py), clipped at zero, mean
diffusivity 0.4 to 1.5e-3 mm^2/s, FA below 0.999. Entropies are in natural-log units, in [0, ln 3].
5,000 voxels are drawn per subject, with a fixed seed.

Writes figures/abstract_preview.png in this repository (not the manuscript's figures folder).
Usage: python analysis/abstract_preview.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from b1500_tensor import b1500_eigenvalues
from paths import MANIFEST

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures" / "abstract_preview.png"
PER_SUBJECT = 5_000
rng = np.random.default_rng(20261027)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
# order 1 in teal, not red: red means linear tensors in Figure 1 (figure review, 2026-10-27)
TEAL = "#117a65"


def fa_h1_h2(lam):
    lam = np.clip(lam, 0, None); p = lam / lam.sum(-1, keepdims=True)
    dev = lam - lam.mean(-1, keepdims=True)
    fa = np.sqrt(1.5 * (dev ** 2).sum(-1) / (lam ** 2).sum(-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        h1 = -np.where(p > 0, p * np.log(p), 0.0).sum(-1)
    return fa, h1, -np.log((p ** 2).sum(-1))


pts = []
for _, r in pd.read_csv(MANIFEST, sep="\t").head(4).iterrows():
    lam, mask = b1500_eigenvalues(r["session_id"])
    md = np.clip(lam, 0, None).mean(-1)
    ok = mask & np.all(np.isfinite(lam), -1) & (np.clip(lam, 0, None).sum(-1) > 0) & (md > 0.4e-3) & (md < 1.5e-3)
    fa, h1, h2 = fa_h1_h2(lam[ok])
    keep = fa < 0.999
    fa, h1, h2 = fa[keep], h1[keep], h2[keep]
    pick = rng.choice(fa.size, min(PER_SUBJECT, fa.size), replace=False)
    pts.append(np.stack([fa[pick], h1[pick], h2[pick]], 1))
P = np.concatenate(pts)

fig, ax = plt.subplots(figsize=(3.2, 2.7), layout="constrained")
ax.scatter(P[:, 0], P[:, 1], s=1.2, color=TEAL, alpha=0.35, linewidths=0, rasterized=True)
ax.scatter(P[:, 0], P[:, 2], s=1.2, color=INK, alpha=0.6, linewidths=0, rasterized=True)
ax.text(0.30, 0.80, "order 2:\none curve", fontsize=11, color=INK, ha="right", va="top")
ax.text(0.80, 1.00, "order 1:\na band", fontsize=11, color=TEAL, ha="left", va="bottom")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.25)
ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
ax.tick_params(labelsize=10, colors=INK2)
ax.set_xlabel("FA", fontsize=12); ax.set_ylabel("eigenvalue entropy", fontsize=12)
ax.set_title("Entropy adds nothing to FA\nat the same order", loc="left", fontsize=12)
ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, dpi=300)
print(f"written -> {OUT} ({len(P)} voxels)")
