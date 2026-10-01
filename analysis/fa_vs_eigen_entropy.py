"""FA against the entropy of the normalized tensor eigenvalues, at order 2 and at order 1.

FA^2 = (3/2)(1 - PR_lambda/3) with PR_lambda = exp(H_2(p_lambda)), so at order 2 FA is an exact
monotone function of the eigenvalue entropy:  H~_2 = ln(3 - 2 FA^2) / ln 3.  At order 1 (Shannon)
it is not. At fixed FA the Shannon eigenvalue entropy depends on the tensor's shape, and the
feasible region on the 2-simplex is bounded exactly by
  upper:  linear (prolate) tensors, lambda = (1, t, t)          mode = +1
  lower:  planar (oblate) tensors,  lambda = (1, 1, t)          mode = -1,  FA <= 1/sqrt(2)
          one eigenvalue zero,      lambda = (1, t, 0)                      FA >= 1/sqrt(2)
(checked numerically on 3 million simplex points before writing this script). The tensor mode
(Ennis & Kindlmann 2006) therefore sets the position inside the band.

Data: tissue voxels of the four HCP-Aging subjects of the paper (first four rows of the b=1500
manifest), eigenvalues from a DIPY weighted least-squares fit to the b = 0 and b = 1500 volumes
(b1500_tensor.py), clipped at zero, mean diffusivity 0.4 to 1.5e-3 mm^2/s.
Units: every entropy and every interval width in this script and its CSV is NORMALIZED (divided by ln 3).
Multiply by ln 3 for natural-log units.
Writes figures/fa_vs_eigen_entropy.{png,pdf} and analysis/fa_vs_eigen_entropy.csv (per subject:
identity error at order 2, share of voxels inside the order-1 envelope, rank correlation of mode
with the position inside the band).
"""
import os
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from b1500_tensor import b1500_eigenvalues
from paths import MANIFEST, FIGURES

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
LN3 = np.log(3.0); PER_SUBJECT = 25_000
rng = np.random.default_rng(20260916)

# colours: diverging blue (planar) <-> red (linear) with a neutral gray midpoint, validated
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, MID, RED = "#1c5cab", "#a3a19a", "#c73a39"
CMAP = LinearSegmentedColormap.from_list("mode", [BLUE, MID, RED])


def descriptors(lam):
    lam = np.clip(lam, 0, None); s = lam.sum(-1, keepdims=True); p = lam / s
    m = lam.mean(-1, keepdims=True); dev = lam - m
    fa = np.sqrt(1.5 * (dev ** 2).sum(-1) / (lam ** 2).sum(-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        h1 = -np.where(p > 0, p * np.log(p), 0.0).sum(-1) / LN3
        nrm = np.sqrt((dev ** 2).sum(-1))
        mode = np.where(nrm > 0, 3 * np.sqrt(6) * np.prod(dev, -1) / nrm ** 3, 0.0)
    h2 = -np.log((p ** 2).sum(-1)) / LN3
    return fa, h1, h2, np.clip(mode, -1, 1)


def family(kind, n=2001):
    t = np.linspace(0, 1, n); one = np.ones_like(t)
    lam = {"linear": np.stack([one, t, t], 1), "planar": np.stack([one, one, t], 1), "edge": np.stack([one, t, np.zeros_like(t)], 1)}[kind]
    fa, h1, h2, _ = descriptors(lam); o = np.argsort(fa)
    return fa[o], h1[o], h2[o]


LIN, PLA, EDG = family("linear"), family("planar"), family("edge")


def bounds(fa):
    up = np.interp(fa, LIN[0], LIN[1])
    lo = np.where(fa <= 1 / np.sqrt(2), np.interp(fa, PLA[0], PLA[1]), np.interp(fa, EDG[0], EDG[1]))
    return lo, up


man = pd.read_csv(MANIFEST, sep="\t").head(4)
rows, pts = [], []
for k, (_, r) in enumerate(man.iterrows()):
    lam, mask = b1500_eigenvalues(r["session_id"])
    md = np.clip(lam, 0, None).mean(-1)
    assert 1e-4 < np.median(md[mask]) < 3e-3, "eigenvalues are not in mm^2/s"
    ok = mask & np.all(np.isfinite(lam), -1) & (np.clip(lam, 0, None).sum(-1) > 0) & (md > 0.4e-3) & (md < 1.5e-3)
    fa, h1, h2, mode = descriptors(lam[ok])
    keep = fa < 0.999
    fa, h1, h2, mode = fa[keep], h1[keep], h2[keep], mode[keep]
    ident = float(np.max(np.abs(h2 - np.log(3 - 2 * fa ** 2) / LN3)))
    lo, up = bounds(fa); tol = 1e-6
    inside = float(np.mean((h1 >= lo - tol) & (h1 <= up + tol)))
    width = up - lo; pos = np.where(width > 1e-4, (h1 - lo) / np.maximum(width, 1e-12), np.nan)
    wide = np.isfinite(pos)
    rho = float(stats.spearmanr(mode[wide], pos[wide])[0])
    rows.append(dict(subject=f"S{k+1}", n_tissue=int(fa.size), median_MD_brain=float(np.median(md[mask])), identity_order2_max_abs_err=ident, frac_inside_order1_envelope=inside,
                     rho_mode_vs_band_position=rho, n_band_wider_than_1e4=int(wide.sum()), median_FA=float(np.median(fa)),
                     frac_FA_above_0p707=float(np.mean(fa > 1 / np.sqrt(2))), median_band_width_at_voxel_FA=float(np.median(width)),
                     rho_FA_H1=float(stats.spearmanr(fa, h1)[0]), frac_FA_above_0p5=float(np.mean(fa > 0.5))))
    print(f"S{k+1}: n={fa.size}, order-2 identity max err {ident:.1e}, inside order-1 envelope {100*inside:.3f}%, "
          f"rho(mode, band position) {rho:+.3f}, median FA {np.median(fa):.3f}, FA>0.707 {100*np.mean(fa > 0.7071):.1f}%")
    pick = rng.choice(fa.size, min(PER_SUBJECT, fa.size), replace=False)
    pts.append(np.stack([fa[pick], h1[pick], h2[pick], mode[pick]], 1))
df = pd.DataFrame(rows); df.to_csv(HERE / "fa_vs_eigen_entropy.csv", index=False)
P = np.concatenate(pts); P = P[rng.permutation(len(P))]
# the figure is drawn on the natural-log scale (entropies in [0, ln 3]); the CSV stays normalized
P[:, 1:3] *= LN3
LIN_P, PLA_P, EDG_P = [(c[0], c[1] * LN3) for c in (LIN, PLA, EDG)]

# ---------------------------------------------------------------- figure
# drawn at its printed size (5.5 in wide) so font sizes are the printed sizes; include without scaling
plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.6), sharex=True, sharey=True, layout="constrained")
for ax in axes:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02 * LN3)
    ax.grid(True, color=GRID, linewidth=0.4); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.set_xlabel("FA")

ax = axes[0]
f = np.linspace(0, 1, 400)
ax.plot(f, np.log(3 - 2 * f ** 2), color=INK, linewidth=0.7, zorder=1, label=r"$H_2 = \ln(3-2\,\mathrm{FA}^2)$")
sc = ax.scatter(P[:, 0], P[:, 2], c=P[:, 3], cmap=CMAP, vmin=-1, vmax=1, s=1.2, alpha=0.6, linewidths=0, rasterized=True, zorder=2)
ax.set_ylabel(r"eigenvalue entropy $H_\alpha(\lambda)$")
ax.set_title(r"A. Order 2: one curve", loc="left")
ax.legend(loc="lower left", frameon=False)


def draw_b(a, s_pt, lw):
    a.plot(LIN_P[0], LIN_P[1], color=RED, linewidth=lw, zorder=1, label=r"linear, $\lambda_2=\lambda_3$")
    a.plot(PLA_P[0], PLA_P[1], color=BLUE, linewidth=lw, zorder=1, label=r"planar, $\lambda_1=\lambda_2$")
    a.plot(EDG_P[0], EDG_P[1], color=INK2, linewidth=lw, linestyle=(0, (4, 2)), zorder=1, label=r"one eigenvalue zero")
    a.scatter(P[:, 0], P[:, 1], c=P[:, 3], cmap=CMAP, vmin=-1, vmax=1, s=s_pt, alpha=0.6, linewidths=0, rasterized=True, zorder=2)


ax = axes[1]
draw_b(ax, 1.0, 0.8)
ax.set_title(r"B. Order 1: a band set by the mode", loc="left")
ax.legend(loc="lower left", frameon=False, handlelength=1.6)
ins = ax.inset_axes([0.07, 0.355, 0.42, 0.355])
draw_b(ins, 1.6, 0.8)
# zoom about 2x on both axes, where the order-1 band opens and the mode gradient across it is visible
ins.set_xlim(0.50, 0.70); ins.set_ylim(0.82, 1.01)
ins.tick_params(labelsize=6, colors=INK2, width=0.5, length=2); ins.grid(True, color=GRID, linewidth=0.35); ins.set_axisbelow(True)
for sp in ins.spines.values(): sp.set_edgecolor(INK2); sp.set_linewidth(0.5)
ax.indicate_inset_zoom(ins, edgecolor=INK2, alpha=0.6, linewidth=0.5)

cb = fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.02, ticks=[-1, 0, 1])
cb.ax.set_yticklabels(["planar", "0", "linear"]); cb.set_label("tensor mode", color=INK); cb.outline.set_edgecolor(GRID)
cb.solids.set_alpha(1)
fig.savefig(FIGURES / "fa_vs_eigen_entropy.png", dpi=300)
fig.savefig(FIGURES / "fa_vs_eigen_entropy.pdf")
print("written -> figures/fa_vs_eigen_entropy.{png,pdf}, analysis/fa_vs_eigen_entropy.csv")
