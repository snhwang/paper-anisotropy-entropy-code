"""Figure 1: three diffusion tensors read as distributions of their normalized eigenvalues.

An isotropic tensor, a prolate (linear) tensor and an oblate (planar) tensor. The prolate and oblate
tensors are chosen with the same FA (0.6), so they share the order-2 entropy H_2 = ln(3 - 2 FA^2) and
the participation ratio PR_lambda = exp(H_2), but their Shannon entropies H_1 differ because their
shapes (tensor modes) differ. Top row, diffusion ellipsoids with semi-axes proportional to the
eigenvalues, with simple Lambertian shading. Bottom row, the normalized eigenvalues
p_i = lambda_i / sum_j lambda_j against the uniform value 1/3. Entropies in natural-log units.

Writes tensor_concept.{png,pdf} to the figures folder of paths.py and analysis/tensor_concept.csv,
read by identity_checks.py.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from paths import FIGURES

HERE = Path(__file__).resolve().parent
FA_PAIR = 0.6

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, MID, RED = "#1c5cab", "#a3a19a", "#c73a39"   # the tensor-mode colours of the FA-entropy figure


def fa(l):
    l = np.asarray(l, float); m = l.mean()
    return float(np.sqrt(1.5 * ((l - m) ** 2).sum() / (l ** 2).sum()))


def mode(l):
    l = np.asarray(l, float); d = l - l.mean(); n = np.sqrt((d ** 2).sum())
    return float(3 * np.sqrt(6) * np.prod(d) / n ** 3) if n > 0 else 0.0


def family(kind, target):
    """Eigenvalues (1, t, t) or (1, 1, t) with the given FA, scaled to mean 1."""
    shape = (lambda t: [1, t, t]) if kind == "prolate" else (lambda t: [1, 1, t])
    t = brentq(lambda t: fa(shape(t)) - target, 1e-9, 1 - 1e-9)
    l = np.array(shape(t), float)
    return l / l.mean()


TENSORS = [
    ("isotropic", np.array([1.0, 1.0, 1.0]), MID),
    ("prolate (linear)", family("prolate", FA_PAIR), RED),
    ("oblate (planar)", family("oblate", FA_PAIR), BLUE),
]

rows = []
for name, lam, _ in TENSORS:
    p = lam / lam.sum()
    h2 = float(-np.log((p ** 2).sum())); h1 = float(-(p * np.log(p)).sum())
    rows.append(dict(tensor=name, l1=lam[0], l2=lam[1], l3=lam[2], p1=p[0], p2=p[1], p3=p[2],
                     FA=fa(lam), mode=mode(lam), H2=h2, PR=float(np.exp(h2)), H1=h1))
    assert abs(h2 - np.log(3 - 2 * fa(lam) ** 2)) < 1e-12, "H2 = ln(3 - 2 FA^2) failed"
df = pd.DataFrame(rows)
df.to_csv(HERE / "tensor_concept.csv", index=False)
print(df.round(4).to_string(index=False))

# ---------------------------------------------------------------- figure
plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig = plt.figure(figsize=(6.8, 2.9))
gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], left=0.07, right=0.99, top=0.93, bottom=0.27, hspace=0.12, wspace=0.32)
light = np.array([0.35, 0.25, 1.0]); light /= np.linalg.norm(light)
u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 120), np.linspace(0, np.pi, 60))
SX, SY, SZ = np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)
scale = max(l.max() for _, l, _ in TENSORS)

for k, (name, lam, colour) in enumerate(TENSORS):
    # ellipsoid: first eigenvector along x, so the prolate cigar lies across the view and the oblate disc faces it
    a, b, c = lam / scale
    X, Y, Z = a * SX, b * SY, c * SZ
    nx, ny, nz = X / a ** 2, Y / b ** 2, Z / c ** 2
    nn = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    lamb = np.clip(0.45 + 0.55 * (nx * light[0] + ny * light[1] + nz * light[2]) / nn, 0.3, 1.0)
    rgb = np.array(matplotlib.colors.to_rgb(colour))
    cols = np.ones(X.shape + (4,)); cols[..., :3] = rgb[None, None, :] * lamb[..., None]
    ax = fig.add_subplot(gs[0, k], projection="3d")
    ax.plot_surface(X, Y, Z, facecolors=cols, rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False, rasterized=True)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
    lim = 0.6
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.view_init(elev=30, azim=-55)
    ax.set_title(name, fontsize=8, pad=-2)

    r = df.iloc[k]
    bx = fig.add_subplot(gs[1, k])
    bx.bar([0, 1, 2], [r.p1, r.p2, r.p3], width=0.62, color=colour, edgecolor="none")
    bx.axhline(1 / 3, color=INK2, lw=0.7, ls=(0, (3, 2)))
    bx.set_xticks([0, 1, 2]); bx.set_xticklabels([r"$p_1$", r"$p_2$", r"$p_3$"])
    bx.set_ylim(0, 0.65); bx.set_yticks([0, 1 / 3, 0.6]); bx.set_yticklabels(["0", "1/3", "0.6"])
    for sp in ("top", "right"): bx.spines[sp].set_visible(False)
    if k == 0:
        bx.set_ylabel("normalized eigenvalue")
    stats = (f"FA = {r.FA:.2f}\n"
             f"$H_2$ = {r.H2:.3f},  $\\mathrm{{PR}}_\\lambda$ = {r.PR:.2f}\n"
             f"$H_1$ = {r.H1:.3f}")
    bx.text(0.5, -0.30, stats, transform=bx.transAxes, ha="center", va="top", fontsize=7, linespacing=1.45)

FIGURES.mkdir(exist_ok=True)
fig.savefig(FIGURES / "tensor_concept.png", dpi=300)
fig.savefig(FIGURES / "tensor_concept.pdf", dpi=600)
print(f"written -> {FIGURES}/tensor_concept.{{png,pdf}}, analysis/tensor_concept.csv")
