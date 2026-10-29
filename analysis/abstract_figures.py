"""Figures 1 and 2 of the ISMRM 2027 abstract, the paper's Figures 2 and 3 revised for the abstract.

The paper's own figures (fa_vs_eigen_entropy.py, build_reflections_maps.py) are left unchanged,
because the manuscript is under review. This script draws abstract versions from the same data and
computations, with the changes of the 27 October 2026 figure review (ISMRM2027/figure_review_2026-10-27.md):

Figure 1, FA against the eigenvalue entropy (paper Figure 2)
  - Panel B carries a second mark at the tissue median FA, 0.2, with the width of the order-1 band
    there, computed from the exact bounds (as in identity_checks.py, about 0.001). The FA 0.7 marks stay.
  - Panel B's y axis reads "order-1 (Shannon) eigenvalue entropy".
  The helpers (descriptors, the family curves, the exact bounds) are copied from fa_vs_eigen_entropy.py,
  which cannot be imported because it draws the paper figure when loaded. The data and selection are
  the same: four HCP-Aging subjects, cached b = 1500 eigenvalues, MD 0.4 to 1.5e-3 mm^2/s, FA < 0.999,
  25,000 voxels per subject with the same seed.
Figure 2, anisotropy maps and their order-2 entropy twins (paper Figure 3)
  - The order-1 column is dropped, leaving FA | PR_lambda/3 and CV_D^2 | PR/N.
  - CV_D^2 is capped at 0.25 (0.5 in the paper).
  - Titles name the participation ratio and read "(left map, reversed)". The suptitle states the result.
  The maps come from build_reflections_maps.py's functions (same participant and slice). The two
  order-2 identities are asserted here, and the paper's check CSV is not rewritten.

Writes figures/abstract_fig1.png and figures/abstract_fig2.png in this repository.
Usage: python analysis/abstract_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from b1500_tensor import b1500_eigenvalues
from build_reflections_maps import directional_maps, eigen_maps, load, panel
from paths import MANIFEST

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"
LN3 = np.log(3.0); PER_SUBJECT = 25_000
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, MID, RED = "#1c5cab", "#a3a19a", "#c73a39"
CMAP = LinearSegmentedColormap.from_list("mode", [BLUE, MID, RED])
BAND = "#ebe6dc"
F_MARK, F_TISSUE = 0.7, 0.2


# ---------------------------------------------------------------- helpers copied from fa_vs_eigen_entropy.py
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
    lam = {"linear": np.stack([one, t, t], 1), "planar": np.stack([one, one, t], 1),
           "edge": np.stack([one, t, np.zeros_like(t)], 1)}[kind]
    fa, h1, h2, _ = descriptors(lam); o = np.argsort(fa)
    return fa[o], h1[o], h2[o]


def family_h1(fa, kind):
    f2 = np.clip(fa, 0.0, 1.0) ** 2; one = np.ones_like(f2)
    if kind == "linear":
        t = 2 * (1 - f2) / (2 + np.sqrt(np.clip(4 - 4 * (2 * f2 - 1) * (f2 - 1), 0, None)))
        lam = np.stack([one, t, t], -1)
    elif kind == "planar":
        t = np.clip(2 * (1 - 2 * f2) / (2 + np.sqrt(np.clip(4 - 4 * (f2 - 1) * (2 * f2 - 1), 0, None))), 0, 1)
        lam = np.stack([one, one, t], -1)
    else:
        t = np.clip(2 * (1 - f2) / (1 + np.sqrt(np.clip(1 - 4 * (f2 - 1) ** 2, 0, None))), 0, 1)
        lam = np.stack([one, t, np.zeros_like(t)], -1)
    return descriptors(lam)[1]


def bounds(fa):
    up = family_h1(fa, "linear")
    lo = np.where(fa <= 1 / np.sqrt(2), family_h1(fa, "planar"), family_h1(fa, "edge"))
    return lo, up


# ---------------------------------------------------------------- Figure 1
rng = np.random.default_rng(20260916)
pts = []
for _, r in pd.read_csv(MANIFEST, sep="\t").head(4).iterrows():
    lam, mask = b1500_eigenvalues(r["session_id"])
    md = np.clip(lam, 0, None).mean(-1)
    ok = mask & np.all(np.isfinite(lam), -1) & (np.clip(lam, 0, None).sum(-1) > 0) & (md > 0.4e-3) & (md < 1.5e-3)
    fa, h1, h2, mode = descriptors(lam[ok])
    keep = fa < 0.999
    fa, h1, h2, mode = fa[keep], h1[keep], h2[keep], mode[keep]
    pick = rng.choice(fa.size, min(PER_SUBJECT, fa.size), replace=False)
    pts.append(np.stack([fa[pick], h1[pick] * LN3, h2[pick] * LN3, mode[pick]], 1))
P = np.concatenate(pts); P = P[rng.permutation(len(P))]
LIN, PLA, EDG = [(c[0], c[1] * LN3) for c in (family("linear"), family("planar"), family("edge"))]

plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3), sharex=True, layout="constrained")
for ax in axes:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02 * LN3)
    ax.grid(True, color=GRID, linewidth=0.4); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("FA (order 2)")


def note(a, x, y, text, dx=-0.04, dy=-0.13, ha="right"):
    a.annotate(text, xy=(x, y), xytext=(x + dx, y + dy), ha=ha, va="top", fontsize=6.5, color=INK,
               arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5, shrinkA=1, shrinkB=2), zorder=5)


ax = axes[0]
f = np.linspace(0, 1, 400)
ax.plot(f, np.log(3 - 2 * f ** 2), color=INK, linewidth=0.6, zorder=1, label=r"$H_2 = \ln(3-2\,\mathrm{FA}^2)$")
sc = ax.scatter(P[:, 0], P[:, 2], c=P[:, 3], cmap=CMAP, vmin=-1, vmax=1, s=0.3, alpha=0.9, linewidths=0,
                rasterized=True, zorder=2)
h2_mark = np.log(3 - 2 * F_MARK ** 2)
ax.plot([F_MARK], [h2_mark], marker="o", ms=3.2, mfc="white", mec=INK, mew=0.8, zorder=4)
note(ax, F_MARK, h2_mark, rf"one value of $H_2$ at FA {F_MARK}")
ax.set_ylabel(r"order-2 eigenvalue entropy $H_2$")
ax.set_title("A. Same order: one curve", loc="left")
ax.legend(loc="lower left", frameon=False)

ax = axes[1]
fg = np.linspace(0, 1, 801)
lo_g, up_g = bounds(fg)
ax.fill_between(fg, lo_g * LN3, up_g * LN3, color=BAND, linewidth=0, zorder=0.5)
ax.plot(LIN[0], LIN[1], color=RED, linewidth=0.8, zorder=1, label=r"linear, $\lambda_2=\lambda_3$")
ax.plot(PLA[0], PLA[1], color=BLUE, linewidth=0.8, zorder=1, label=r"planar, $\lambda_1=\lambda_2$")
ax.plot(EDG[0], EDG[1], color=INK2, linewidth=0.8, linestyle=(0, (4, 2)), zorder=1, label="one eigenvalue zero")
ax.scatter(P[:, 0], P[:, 1], c=P[:, 3], cmap=CMAP, vmin=-1, vmax=1, s=1.0, alpha=0.6, linewidths=0,
           rasterized=True, zorder=2)
lo_m, up_m = (v[0] * LN3 for v in bounds(np.array([F_MARK])))
ax.annotate("", xy=(F_MARK, up_m), xytext=(F_MARK, lo_m), zorder=4,
            arrowprops=dict(arrowstyle="|-|,widthA=0.2,widthB=0.2", color=INK, lw=0.8, shrinkA=0, shrinkB=0))
note(ax, F_MARK, lo_m, rf"range of $H_1$ at FA {F_MARK}")
# the tissue mark: band width at FA 0.2 in the plotted natural-log units (identity_checks.py: 0.0011)
lo_t, up_t = (v[0] for v in bounds(np.array([F_TISSUE])))
width = (up_t - lo_t) * LN3
ax.plot([F_TISSUE], [up_t * LN3], marker="o", ms=3.2, mfc="white", mec=INK, mew=0.8, zorder=4)
note(ax, F_TISSUE, up_t * LN3, f"tissue median FA {F_TISSUE},\nrange {width:.3f}", dx=-0.15, dy=-0.30, ha="left")
ax.set_ylabel("order-1 (Shannon)\neigenvalue entropy $H_1$")
ax.set_title("B. Different orders: a band", loc="left")
ax.legend(loc="lower left", frameon=False, handlelength=1.6)
cb = fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.02, ticks=[-1, 0, 1])
cb.ax.set_yticklabels(["planar", "0", "linear"]); cb.set_label("tensor mode", color=INK); cb.outline.set_edgecolor(GRID)
cb.solids.set_alpha(1)
OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "abstract_fig1.png", dpi=300)
print(f"Figure 1: band width at FA {F_TISSUE} is {width:.4f} (natural log)")

# ---------------------------------------------------------------- Figure 2
bval, img, mask, lam = load()
dm = directional_maps(bval, img, mask); em = eigen_maps(lam, mask); del img
ok = dm["ok"]; ok2 = em["ok"] & np.isfinite(em["FA"])
assert np.nanmax(np.abs(dm["PRN"][ok] - 1 / (1 + dm["Jquad"][ok]))) < 1e-9, "PR/N identity failed"
assert np.nanmax(np.abs(em["FA"][ok2] ** 2 - 1.5 * (1 - em["PRl3"][ok2]))) < 1e-6, "FA identity failed"
z = 60   # the paper's slice (build_reflections_maps.SLICE_Z)
bg = np.nan_to_num(em["FA"][:, :, z]); okd = dm["ok"][:, :, z]; oke = em["ok"][:, :, z] & np.isfinite(em["FA"][:, :, z])
fig, axes = plt.subplots(2, 2, figsize=(4.4, 4.0), layout="constrained")
specs = [
    (em["FA"][:, :, z], oke, 0.0, 0.9, "viridis", "FA"),
    (em["PRl3"][:, :, z], oke, 0.45, 1.0, "viridis", "$\\mathrm{PR}_{\\lambda}/3$, participation ratio\n(left map, reversed)"),
    (dm["Jquad"][:, :, z], okd, 0.0, 0.25, "magma", "$\\mathrm{CV}_D^{2}$"),
    (dm["PRN"][:, :, z], okd, 0.65, 1.0, "magma", "$\\mathrm{PR}/N$, participation ratio\n(left map, reversed)"),
]
for ax, (mp, okm, lo, hi, cmap, title) in zip(axes.flat, specs):
    panel(ax, bg, mp, okm, lo, hi, cmap, title)
axes[0, 0].set_ylabel("tensor eigenvalues ($K=3$)", fontsize=7)
axes[1, 0].set_ylabel(f"directional diffusivities ($N={dm['N']}$)", fontsize=7)
fig.suptitle("Each anisotropy map has an entropy twin with reversed contrast", fontsize=8)
fig.savefig(OUT / "abstract_fig2.png", dpi=300)
print("written -> figures/abstract_fig1.png, figures/abstract_fig2.png")
