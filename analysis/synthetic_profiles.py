"""Table 1 (Section 4.3): anisotropy indices and entropies of noise-free synthetic voxels.
At order 2 the normalized entropy is a function of CV_D alone, so the two columns carry the
same information. An entropy of another order reads the same divergence from isotropy
differently and can order the voxels differently.

For each synthetic profile D_i = sum_k f_k g_i^T D_k g_i on N=93 near-uniform
directions we compute, from the angular distribution p_i = D_i / sum_j D_j:
  CV_D^2 (= J_quad = chi^2 divergence from isotropy), J_ln, J_harm,
  PR/N = 1/(1+CV^2), normalized Renyi entropies H~_alpha (alpha = 0.5,1,2,4,inf),
  GFA of the profile (Tuch), and FA of the least-squares single-tensor fit;
plus signal-domain (b=1500) H~_1^(S), PR^(S)/N, J_ln^(S).
The exact identity ln N - H_alpha = ln(1+J_alpha)/(alpha-1) is asserted.
Writes analysis/synthetic_profiles.csv and figures/synthetic_profiles.{png,pdf}.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
N = 93
B = 1500.0
ALPHAS = [0.5, 1.0, 2.0, 4.0, np.inf]
ALPHA_CURVE = np.concatenate([np.geomspace(0.25, 16, 40)])


def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n); theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)


def unit(v):
    v = np.asarray(v, float); return v / np.linalg.norm(v)


def prolate(axis, lpar, lperp):
    a = unit(axis)[:, None]; return lperp * np.eye(3) + (lpar - lperp) * (a @ a.T)


def oblate(axis, lplane, lsmall):
    a = unit(axis)[:, None]; return lsmall * (a @ a.T) + lplane * (np.eye(3) - a @ a.T)


def profile(G, tensors, fracs, b=B):
    """Measured ADC profile of a multi-tensor voxel at b: the SIGNAL is the
    fraction-weighted mixture S_i = sum_k f_k exp(-b g_i^T D_k g_i), and the
    per-direction apparent diffusivity is D_i = -ln(S_i)/b. For a single tensor
    this is exactly g^T D g; for crossings it is not a linear mixture of the
    component profiles, which is why a crossing's measured profile stays
    non-uniform even when its single-tensor fit is isotropic."""
    S = sum(f * np.exp(-b * np.einsum("ij,jk,ik->i", G, T, G)) for T, f in zip(tensors, fracs))
    return -np.log(S) / b


def renyi_norm(p, a):
    n = p.shape[-1]; pc = np.maximum(p, 1e-300)
    if np.isinf(a):
        return -np.log(pc.max()) / np.log(n)
    if abs(a - 1) < 1e-12:
        return -(pc * np.log(pc)).sum() / np.log(n)
    return np.log((pc ** a).sum()) / (1 - a) / np.log(n)


def fa_of_tensor(T):
    lam = np.linalg.eigvalsh(T); m = lam.mean()
    return np.sqrt(1.5 * ((lam - m) ** 2).sum() / (lam ** 2).sum())


def fit_tensor(G, D):
    X = np.column_stack([G[:, 0] ** 2, G[:, 1] ** 2, G[:, 2] ** 2,
                         2 * G[:, 0] * G[:, 1], 2 * G[:, 0] * G[:, 2], 2 * G[:, 1] * G[:, 2]])
    t, *_ = np.linalg.lstsq(X, D, rcond=None)
    return np.array([[t[0], t[3], t[4]], [t[3], t[1], t[5]], [t[4], t[5], t[2]]])


def metrics(D, G):
    p = D / D.sum(); n = len(D); Db = D.mean(); cv2 = D.var() / Db ** 2
    out = dict(CV_D=np.sqrt(cv2), J_quad=cv2, J_ln=np.log(Db) - np.log(D).mean(),
               J_harm=(1 / D).mean() * Db - 1, PR_over_N=1 / (n * (p ** 2).sum()),
               GFA_profile=np.sqrt(n * ((D - Db) ** 2).sum() / ((n - 1) * (D ** 2).sum())),
               FA_fit=fa_of_tensor(fit_tensor(G, D)))
    for a in ALPHAS:
        out[f"Hn_{'inf' if np.isinf(a) else a:g}" if not np.isinf(a) else "Hn_inf"] = renyi_norm(p, a)
    # exact identity checks
    for a in (0.5, 2.0, 4.0):
        J = (D ** a).mean() / Db ** a - 1
        assert abs((np.log(n) - renyi_norm(p, a) * np.log(n)) - np.log(1 + J) / (a - 1)) < 1e-9
    assert abs((np.log(n) - renyi_norm(p, 1.0) * np.log(n)) - ((p * np.log(D)).sum() - np.log(Db))) < 1e-9
    assert abs(out["PR_over_N"] - 1 / (1 + cv2)) < 1e-12
    # tensor FA is the same construction on the eigenvalue shares (manuscript Eq. FA):
    # FA^2 = (3/2) CV_lambda^2/(1+CV_lambda^2) = (3/2)(1 - PR_lambda/3), CV_lambda^2 = chi^2(p_lambda||u_3),
    # and the perplexity exp(H_alpha(p_lambda)) lies in [1, 3].
    lam = np.linalg.eigvalsh(fit_tensor(G, D)); lam = np.clip(lam, 1e-12, None)
    pl = lam / lam.sum(); cvl2 = lam.var() / lam.mean() ** 2; PRl = 1 / (pl ** 2).sum()
    assert abs(out["FA_fit"] ** 2 - 1.5 * cvl2 / (1 + cvl2)) < 1e-9, "FA identity (CV form) failed"
    assert abs(out["FA_fit"] ** 2 - 1.5 * (1 - PRl / 3)) < 1e-9, "FA identity (PR form) failed"
    assert abs((3 * (pl ** 2).sum() - 1) - cvl2) < 1e-9, "chi^2 of eigenvalue shares != CV_lambda^2"
    for a in (0.5, 1.0, 2.0, 4.0):
        assert 1 - 1e-9 <= np.exp(renyi_norm(pl, a) * np.log(3)) <= 3 + 1e-9, "eigenvalue perplexity out of [1,3]"
    # weighted generalization (non-uniform angular sampling): with solid-angle-like
    # weights w_i, p_i = w_i D_i / sum_j w_j D_j and the reference is w itself;
    # D_alpha(p||w) = ln(1 + J_alpha^(w)) / (alpha-1) with weighted power means.
    w = np.random.default_rng(1).uniform(0.5, 1.5, size=n); w /= w.sum()
    pw = w * D / (w * D).sum(); Dw = (w * D).sum()
    for a in (0.5, 2.0, 3.0, 4.0):
        div = np.log((pw ** a * w ** (1 - a)).sum()) / (a - 1)
        Jw = (w * D ** a).sum() / Dw ** a - 1
        assert abs(div - np.log(1 + Jw) / (a - 1)) < 1e-9, "weighted identity failed"
    kl = (pw * np.log(pw / w)).sum()
    assert abs(kl - ((pw * np.log(D)).sum() - np.log(Dw))) < 1e-9, "weighted Shannon limit failed"
    # signal domain
    S = np.exp(-B * D); q = S / S.sum()
    out["Hn1_S"] = renyi_norm(q, 1.0); out["PR_S_over_N"] = 1 / (n * (q ** 2).sum())
    out["J_ln_S"] = np.log(S.mean()) - np.log(S).mean()
    out["_curve"] = np.array([renyi_norm(p, a) for a in ALPHA_CURVE])
    return out


G = fibonacci_sphere(N)
e3 = 1e-3
c60 = unit([np.cos(np.pi / 3), np.sin(np.pi / 3), 0])
CASES = {
    "isotropic":            ([prolate([0, 0, 1], 0.8 * e3, 0.8 * e3)], [1.0]),
    "single fiber, strong": ([prolate([0, 0, 1], 1.7 * e3, 0.3 * e3)], [1.0]),
    "single fiber, weak":   ([prolate([0, 0, 1], 1.0 * e3, 0.7 * e3)], [1.0]),
    "planar (oblate)":      ([oblate([0, 0, 1], 1.2 * e3, 0.3 * e3)], [1.0]),
    "crossing 90 deg":      ([prolate([1, 0, 0], 1.7 * e3, 0.3 * e3), prolate([0, 1, 0], 1.7 * e3, 0.3 * e3)], [0.5, 0.5]),
    "crossing 60 deg":      ([prolate([1, 0, 0], 1.7 * e3, 0.3 * e3), prolate(c60, 1.7 * e3, 0.3 * e3)], [0.5, 0.5]),
    "three-way crossing":   ([prolate([1, 0, 0], 1.7 * e3, 0.3 * e3), prolate([0, 1, 0], 1.7 * e3, 0.3 * e3),
                              prolate([0, 0, 1], 1.7 * e3, 0.3 * e3)], [1 / 3] * 3),
}

rows, curves = [], {}
for name, (tens, fr) in CASES.items():
    m = metrics(profile(G, tens, fr), G); curves[name] = m.pop("_curve"); rows.append(dict(case=name, **m))
df = pd.DataFrame(rows)
df.to_csv(HERE / "synthetic_profiles.csv", index=False)

cols = ["FA_fit", "GFA_profile", "CV_D", "J_ln", "PR_over_N", "Hn_0.5", "Hn_1", "Hn_2", "Hn_4", "Hn_inf", "Hn1_S", "J_ln_S"]
print(df.set_index("case")[cols].round(3).to_string())

# figure: (A) normalized Renyi spectrum per case, above (B) anisotropy indices vs entropy deficits
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 10,
                     "xtick.labelsize": 11, "ytick.labelsize": 11})
fig, ax = plt.subplots(2, 1, figsize=(7.5, 10))
for name, c in curves.items():
    ax[0].plot(ALPHA_CURVE, c, label=name, linewidth=1.8)
ax[0].set_xscale("log"); ax[0].set_xlabel("Rényi order α"); ax[0].set_ylabel(r"$\tilde H_\alpha$ (diffusivity domain)")
ticks = [0.25, 0.5, 1, 2, 4, 8, 16]
ax[0].set_xticks(ticks); ax[0].set_xticklabels(["1/4", "1/2", "1", "2", "4", "8", "16"]); ax[0].minorticks_off()
ax[0].axvline(1, color="0.7", linewidth=0.8, linestyle=":"); ax[0].axvline(2, color="0.7", linewidth=0.8, linestyle=":")
ax[0].set_title("A. Normalized Rényi spectrum", loc="left"); ax[0].legend(frameon=False)
# B: one quantity in one unit, the divergence from isotropy, ln K - H_alpha, at three
# orders of the profile and at order 2 of the tensor eigenvalues (K = 3). Every column is
# given by the identity in closed form; heights are comparable, the rise with alpha is the
# order dependence, and the tensor's zero on the three-way crossing sits beside a nonzero
# profile divergence.
div = pd.DataFrame({
    "case": df["case"],
    r"profile, $\alpha=1/2$ ($-2\ln\mathrm{BC}$)": (1 - df["Hn_0.5"]) * np.log(N),
    r"profile, $\alpha=1$ (Shannon deficit)": (1 - df["Hn_1"]) * np.log(N),
    r"profile, $\alpha=2$ ($\ln(1+\mathrm{CV}_D^2)$)": np.log1p(df["CV_D"] ** 2),
    r"tensor, $\alpha=2$ ($\ln(1+\mathrm{CV}_\lambda^2)$, from FA)": np.log1p(df["FA_fit"] ** 2 / (1.5 - df["FA_fit"] ** 2)),
})
div.to_csv(HERE / "synthetic_profiles_divergences.csv", index=False)
idx = np.arange(len(df)); w = 0.2
for k, col in enumerate(div.columns[1:]):
    ax[1].bar(idx + (k - 1.5) * w, div[col].values, w, label=col)
ax[1].set_xticks(idx); ax[1].set_xticklabels(df["case"], rotation=30, ha="right")
ax[1].set_ylabel(r"divergence from isotropy, $\ln K - H_\alpha$")
ax[1].set_title("B. One divergence, read at three orders and on two distributions", loc="left"); ax[1].legend(frameon=False, ncol=1, fontsize=9)
fig.tight_layout()
(ROOT / "figures").mkdir(exist_ok=True)
fig.savefig(ROOT / "figures" / "synthetic_profiles.png", dpi=200); fig.savefig(ROOT / "figures" / "synthetic_profiles.pdf")
print("\nwritten -> analysis/synthetic_profiles.csv, figures/synthetic_profiles.{png,pdf}")
