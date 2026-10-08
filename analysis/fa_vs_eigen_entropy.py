"""Figure 2: FA against the entropy of the normalized tensor eigenvalues, at order 2 and at order 1.

FA^2 = (3/2)(1 - PR_lambda/3) with PR_lambda = exp(H_2(p_lambda)), so at order 2 FA is an exact
monotone function of the eigenvalue entropy:  H~_2 = ln(3 - 2 FA^2) / ln 3.  At order 1 (Shannon)
it is not. At fixed FA the Shannon eigenvalue entropy depends on the tensor's shape, and the
feasible region on the 2-simplex is bounded exactly by
  upper:  linear (prolate) tensors, lambda = (1, t, t)          mode = +1
  lower:  planar (oblate) tensors,  lambda = (1, 1, t)          mode = -1,  FA <= 1/sqrt(2)
          one eigenvalue zero,      lambda = (1, t, 0)                      FA >= 1/sqrt(2)
(checked in identity_checks.py on 800,000 random eigenvalue triples). The tensor mode
(Ennis & Kindlmann 2006) therefore sets the position inside the band.

Data: tissue voxels of the four HCP-Aging subjects of the paper (first four rows of the b=1500
manifest), eigenvalues from a DIPY weighted least-squares fit to the b = 0 and b = 1500 volumes
(b1500_tensor.py), clipped at zero, mean diffusivity 0.4 to 1.5e-3 mm^2/s.
Units: every entropy and every interval width in this script and its CSV is NORMALIZED (divided by ln 3).
Multiply by ln 3 for natural-log units.
Panels: A the order-2 entropy against FA, one curve, with one value marked; B the order-1 entropy
against FA, with the band between the bounds shaded and its range marked at one FA.
Writes figures/fa_vs_eigen_entropy.{png,pdf} and analysis/fa_vs_eigen_entropy.csv (per subject:
identity error at order 2, share of voxels inside the order-1 envelope, rank correlation of mode
with the position in the band below FA = 1/sqrt(2)).
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


def family_h1(fa, kind):
    """Normalized Shannon entropy of the family tensor with the given FA, by exact inversion of FA(t):
    linear (1,t,t) has FA^2 = (1-t)^2/(1+2t^2), planar (1,1,t) has FA^2 = (1-t)^2/(2+t^2), and the
    zero-eigenvalue edge (1,t,0) has FA^2 = (t^2-t+1)/(1+t^2). Each is a quadratic in t, solved in its
    numerically stable form. Exact bounds keep the position in the band well defined at very low FA,
    where the band is narrower than the error of interpolating the curves."""
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


def band_position(fa, h1, lo, up):
    below = (fa < 1 / np.sqrt(2)) & (up - lo > 0)
    return np.where(below, (h1 - lo) / np.where(below, up - lo, 1.0), np.nan)


# the exact bounds must reproduce the plotted family curves
for _kind, _curve in (("linear", LIN), ("planar", PLA), ("edge", EDG)):
    _sel = (_curve[0] <= 1 / np.sqrt(2)) if _kind == "planar" else (_curve[0] >= 1 / np.sqrt(2)) if _kind == "edge" else np.ones_like(_curve[0], bool)
    assert np.max(np.abs(family_h1(_curve[0][_sel], _kind) - _curve[1][_sel])) < 1e-9, f"exact {_kind} bound disagrees with its curve"


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
    # position in the band, 0 at the planar bound and 1 at the linear bound, below FA = 1/sqrt(2)
    # (above it the lower bound is the zero-eigenvalue edge and the position means something else)
    width = up - lo; pos = band_position(fa, h1, lo, up)
    wide = np.isfinite(pos)
    rho = float(stats.spearmanr(mode[wide], pos[wide])[0])
    rows.append(dict(subject=f"S{k+1}", n_tissue=int(fa.size), median_MD_brain=float(np.median(md[mask])), identity_order2_max_abs_err=ident, frac_inside_order1_envelope=inside,
                     rho_mode_vs_band_position=rho, n_FA_below_0p707=int(wide.sum()), median_FA=float(np.median(fa)),
                     frac_FA_above_0p707=float(np.mean(fa > 1 / np.sqrt(2))), median_band_width_at_voxel_FA=float(np.median(width)),
                     rho_FA_H1=float(stats.spearmanr(fa, h1)[0]), frac_FA_above_0p5=float(np.mean(fa > 0.5))))
    print(f"S{k+1}: n={fa.size}, order-2 identity max err {ident:.1e}, inside order-1 envelope {100*inside:.3f}%, "
          f"rho(mode, band position) {rho:+.3f}, median FA {np.median(fa):.3f}, FA>0.707 {100*np.mean(fa > 0.7071):.1f}%")
    pick = rng.choice(fa.size, min(PER_SUBJECT, fa.size), replace=False)
    pts.append(np.stack([fa[pick], h1[pick], h2[pick], mode[pick], pos[pick]], 1))
df = pd.DataFrame(rows); df.to_csv(HERE / "fa_vs_eigen_entropy.csv", index=False)
P = np.concatenate(pts); P = P[rng.permutation(len(P))]
# the figure is drawn on the natural-log scale (entropies in [0, ln 3]); the CSV stays normalized
P[:, 1:3] *= LN3
LIN_P, PLA_P, EDG_P = [(c[0], c[1] * LN3) for c in (LIN, PLA, EDG)]

# ---------------------------------------------------------------- figure
# drawn at its printed size (6.8 in wide, the CAS text width is 6.84 in) so font sizes are the printed sizes; include without scaling
plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.2), sharex=True, layout="constrained")
for ax in axes:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02 * LN3)
    ax.grid(True, color=GRID, linewidth=0.4); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.set_xlabel("FA (order 2)")
F_MARK = 0.7      # FA at which A marks its single entropy value and B the range of entropies
BAND = "#ebe6dc"  # fill of the order-1 band


def note(a, x, y, text):
    a.annotate(text, xy=(x, y), xytext=(x - 0.04, y - 0.13), ha="right", va="top", fontsize=6.5, color=INK,
               arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5, shrinkA=1, shrinkB=2), zorder=5)


ax = axes[0]
f = np.linspace(0, 1, 400)
ax.plot(f, np.log(3 - 2 * f ** 2), color=INK, linewidth=0.6, zorder=1, label=r"$H_2 = \ln(3-2\,\mathrm{FA}^2)$")
sc = ax.scatter(P[:, 0], P[:, 2], c=P[:, 3], cmap=CMAP, vmin=-1, vmax=1, s=0.3, alpha=0.9, linewidths=0, rasterized=True, zorder=2)
h2_mark = np.log(3 - 2 * F_MARK ** 2)
ax.plot([F_MARK], [h2_mark], marker="o", ms=3.2, mfc="white", mec=INK, mew=0.8, zorder=4)
note(ax, F_MARK, h2_mark, rf"one value of $H_2$ at FA {F_MARK}")
ax.set_ylabel(r"order-2 eigenvalue entropy $H_2$")
ax.set_title(r"A. Same order: one curve", loc="left")
ax.legend(loc="lower left", frameon=False)


def draw_b(a, s_pt, lw):
    a.plot(LIN_P[0], LIN_P[1], color=RED, linewidth=lw, zorder=1, label=r"linear, $\lambda_2=\lambda_3$")
    a.plot(PLA_P[0], PLA_P[1], color=BLUE, linewidth=lw, zorder=1, label=r"planar, $\lambda_1=\lambda_2$")
    a.plot(EDG_P[0], EDG_P[1], color=INK2, linewidth=lw, linestyle=(0, (4, 2)), zorder=1, label=r"one eigenvalue zero")
    a.scatter(P[:, 0], P[:, 1], c=P[:, 3], cmap=CMAP, vmin=-1, vmax=1, s=s_pt, alpha=0.6, linewidths=0, rasterized=True, zorder=2)


ax = axes[1]
fg = np.linspace(0, 1, 801)
lo_g, up_g = bounds(fg)
ax.fill_between(fg, lo_g * LN3, up_g * LN3, color=BAND, linewidth=0, zorder=0.5)
draw_b(ax, 1.0, 0.8)
lo_m, up_m = (v[0] * LN3 for v in bounds(np.array([F_MARK])))
ax.annotate("", xy=(F_MARK, up_m), xytext=(F_MARK, lo_m), zorder=4,
            arrowprops=dict(arrowstyle="|-|,widthA=0.2,widthB=0.2", color=INK, lw=0.8, shrinkA=0, shrinkB=0))
note(ax, F_MARK, lo_m, rf"range of $H_1$ at FA {F_MARK}")
ax.set_ylabel(r"order-1 eigenvalue entropy $H_1$")
ax.set_title(r"B. Different orders: a band", loc="left")
ax.legend(loc="lower left", frameon=False, handlelength=1.6)

cb = fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.02, ticks=[-1, 0, 1])
cb.ax.set_yticklabels(["planar", "0", "linear"]); cb.set_label("tensor mode", color=INK); cb.outline.set_edgecolor(GRID)
cb.solids.set_alpha(1)
fig.savefig(FIGURES / "fa_vs_eigen_entropy.png", dpi=300)
fig.savefig(FIGURES / "fa_vs_eigen_entropy.pdf", dpi=600)  # resolution of the rasterized scatter layers
print("written -> figures/fa_vs_eigen_entropy.{png,pdf}, analysis/fa_vs_eigen_entropy.csv")
