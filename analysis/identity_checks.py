"""Verify every mathematical statement and every quoted number in main.tex.

Each check prints [OK  ] or [FAIL] and the script exits non-zero if anything fails.
Part A needs nothing but numpy and scipy (exact algebra on random values).
Part B reads the CSV outputs of the scripts in this folder:
  synthetic_profiles.csv        synthetic_profiles.py      (Table 1)
  tensor_concept.csv            tensor_concept.py          (Figure 1, three tensors as distributions)
  direction_count.csv           direction_count.py         (entropies against the number of directions)
  order1_link.csv               order1_link.py             (order-1 profile against eigenvalue deficit)
  reflections_maps_check.csv    build_reflections_maps.py  (Figure 3, voxelwise identities)
  fa_vs_eigen_entropy.csv       fa_vs_eigen_entropy.py     (tensor interval in tissue)
  renyi_order_test.csv          renyi_order_test.py        (orders 1 and 2 in tissue)
  acquisition_counts.csv        acquisition_counts.py      (Section 5 acquisition facts)
The last four need the HCP-Aging processing sessions (DTI_OUTPUT_DIR) to regenerate, and their CSVs
are kept here so the checks run without the data. Part A also computes the Discussion's worked
numbers (scale of the order-2 deficit, region averages, effective numbers) directly.
Usage:  python identity_checks.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
FAILS = []


def check(label, value, expected, tol, fmt="{:.4g}"):
    ok = abs(value - expected) <= tol
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label:64s} {fmt.format(value):>12s}  (expected {fmt.format(expected)})")
    if not ok:
        FAILS.append(label)


def check_true(label, cond):
    print(f"  [{'OK  ' if cond else 'FAIL'}] {label}")
    if not cond:
        FAILS.append(label)


rng = np.random.default_rng(20260928)


def renyi(p, a):
    p = np.asarray(p, float)
    if a == 1:
        q = p[p > 0]; return float(-(q * np.log(q)).sum())
    if np.isinf(a):
        return float(-np.log(p.max()))
    return float(np.log((p ** a).sum()) / (1 - a))


def fa(lam):
    lam = np.asarray(lam, float); m = lam.mean()
    return float(np.sqrt(1.5 * ((lam - m) ** 2).sum() / (lam ** 2).sum()))


# =============================================================================================
print("A. Exact algebra")
# A1. the identity, Eq. 2, at every order for arbitrary positive values: every form of the labeled equation
# (entropy deficit = Renyi divergence from uniform = ln(1 + J)/(alpha - 1) = alpha/(alpha - 1) ln(M_alpha/M_1))
# and the derivation line before it, sum p^alpha = K^(1 - alpha) (1 + J_alpha)
worst = worst_sum = 0.0
for _ in range(2000):
    K = int(rng.integers(2, 120)); x = rng.uniform(0.05, 5.0, K) ** rng.uniform(0.5, 3.0); p = x / x.sum(); u = np.full(K, 1.0 / K)
    for a in (-2.0, -1.0, 0.5, 2.0, 3.0, 7.0):
        J = (x ** a).mean() / x.mean() ** a - 1
        lhs = np.log(K) - renyi(p, a)
        divergence = np.log((p ** a * u ** (1 - a)).sum()) / (a - 1)
        worst = max(worst, abs(lhs - divergence), abs(lhs - np.log1p(J) / (a - 1)),
                    abs(lhs - a / (a - 1) * np.log(((x ** a).mean()) ** (1 / a) / x.mean())))
        worst_sum = max(worst_sum, abs((p ** a).sum() / (K ** (1 - a) * (1 + J)) - 1))
    worst = max(worst, abs(np.log(K) - renyi(p, 1) - ((p * np.log(x)).sum() - np.log(x.mean()))))
check("Eq. 2 identity, every form, all orders (2000 random sets), max abs error", worst, 0.0, 1e-10, "{:.1e}")
check("Eq. 2 derivation: sum p^alpha = K^(1-alpha)(1 + J_alpha), max rel error", worst_sum, 0.0, 1e-10, "{:.1e}")

# A2. anchors
x = rng.uniform(0.1, 3, 40); p = x / x.sum(); K = x.size; cv2 = x.var() / x.mean() ** 2
check("order 2: PR/K = 1/(1+CV^2)", (1 / (p ** 2).sum()) / K - 1 / (1 + cv2), 0.0, 1e-12, "{:.1e}")
check("order 2: CV^2 = chi^2(p||u)", cv2 - (K * (p ** 2).sum() - 1), 0.0, 1e-12, "{:.1e}")
check("order 1/2: ln K - H_1/2 = -2 ln BC(p,u)", (np.log(K) - renyi(p, 0.5)) + 2 * np.log(np.sqrt(p / K).sum()), 0.0, 1e-12, "{:.1e}")
check("order inf: ln K - H_inf = ln(max x / mean x)", (np.log(K) - renyi(p, np.inf)) - np.log(x.max() / x.mean()), 0.0, 1e-12, "{:.1e}")
klu = (np.log(1 / K) - np.log(p)).mean()
check("uniform log gap ln(mean) - mean(ln) = KL(u||p)", (np.log(x.mean()) - np.log(x).mean()) - klu, 0.0, 1e-12, "{:.1e}")
check("Shannon deficit = Jensen gap of x ln x, scaled by mean", (np.log(K) - renyi(p, 1)) - ((x * np.log(x)).mean() - x.mean() * np.log(x.mean())) / x.mean(), 0.0, 1e-12, "{:.1e}")

# A3. signs: gaps have the sign of alpha(alpha-1); divergence >= 0 only for alpha > 0
bad = 0
for _ in range(3000):
    x = rng.uniform(0.2, 3, 7); p = x / x.sum()
    for a in (-2.0, -0.5, 0.5, 2.0, 4.0):
        J = (x ** a).mean() / x.mean() ** a - 1; D = np.log(7) - renyi(p, a)
        bad += int(np.sign(J) != np.sign(a * (a - 1))) + int((a > 0) and D < -1e-12) + int((a < 0) and D > 1e-12)
check("sign violations (gap sign, divergence sign by order)", bad, 0, 0, "{:.0f}")

# A4. angle form: theta between x and (1,...,1)
x = rng.uniform(0.1, 3, 3); th = np.arccos(x.sum() / np.sqrt(3 * (x ** 2).sum())); p = x / x.sum()
check("PR/K = cos^2 theta", (1 / (p ** 2).sum()) / 3 - np.cos(th) ** 2, 0.0, 1e-12, "{:.1e}")
check("CV = tan theta", np.sqrt(x.var() / x.mean() ** 2) - np.tan(th), 0.0, 1e-12, "{:.1e}")
check("FA = sqrt(3/2) sin theta", fa(x) - np.sqrt(1.5) * np.sin(th), 0.0, 1e-12, "{:.1e}")

# A4b. the coefficient of variation of the diffusion (Aja-Fernandez et al. 2018, Eq. 20) equals the GFA of Eq. (7)
r_cvd = np.random.default_rng(2018); worst = 0.0
for _ in range(500):
    N = int(r_cvd.integers(6, 150)); D = r_cvd.uniform(0.1, 3.0, N) * 1e-3; b = 1500.0; logE = -b * D
    V = N / (N - 1) * ((logE ** 2).mean() - logE.mean() ** 2)          # their sample variance
    cvd2 = V / ((-logE) ** 2).mean()
    cv2 = D.var() / D.mean() ** 2
    worst = max(worst, abs(cvd2 - N / (N - 1) * cv2 / (1 + cv2)))
check("CVD^2 of Aja-Fernandez 2018 = GFA^2 of the profile, Eq. (7), max error", worst, 0.0, 1e-12, "{:.1e}")

# A6. tensor: FA order 2, QA order 1, VR reverse divergence, Atkinson
worst = 0.0
for _ in range(2000):
    lam = rng.uniform(0.05, 3.0, 3); p = lam / lam.sum(); cvl2 = lam.var() / lam.mean() ** 2
    D2 = np.log(3) - renyi(p, 2)
    worst = max(worst, abs(D2 - np.log1p(cvl2)), abs(fa(lam) ** 2 - 1.5 * (1 - np.exp(-D2))),
                abs(fa(lam) ** 2 - 1.5 * cvl2 / (1 + cvl2)),
                abs((np.log(3) - renyi(p, 1)) - ((p * np.log(lam)).sum() - np.log(lam.mean()))))
    vr = lam.prod() / lam.mean() ** 3; G01 = np.log(lam.mean()) - np.log(lam).mean()
    worst = max(worst, abs(-np.log(vr) / 3 - G01), abs(vr ** (1 / 3) - np.exp(np.log(lam).mean()) / lam.mean()))
check("tensor orders, FA, QA and VR relations, max abs error", worst, 0.0, 1e-10, "{:.1e}")
worst = 0.0  # Ozarslan et al. 2005, Eq. 11: FA = sqrt((3 - 1/trace(R^2))/2), and trace(R^2) = exp(-H_2)
for _ in range(2000):
    lam = rng.uniform(0.05, 3.0, 3); p = lam / lam.sum(); trR2 = (p ** 2).sum()
    worst = max(worst, abs(fa(lam) - np.sqrt(0.5 * (3 - 1 / trR2))), abs(trR2 - np.exp(-renyi(p, 2))))
check("Ozarslan Eq. 11, FA from trace(R^2), and trace(R^2) = exp(-H_2)", worst, 0.0, 1e-12, "{:.1e}")
r_pa = np.random.default_rng(2009); worst_pa = worst_ga = 0.0
for _ in range(2000):  # Dryden et al. 2009: PA = FA of sqrt(eigenvalues); Cheng et al. 2009: H_1/2 from the geometric anisotropy
    lam = r_pa.uniform(0.05, 3.0, 3); p = lam / lam.sum(); D_half = np.log(3) - renyi(p, 0.5)
    worst_pa = max(worst_pa, abs(fa(np.sqrt(lam)) ** 2 - 1.5 * (1 - np.exp(-D_half))))
    K = int(r_pa.integers(3, 100)); q = r_pa.dirichlet(np.ones(K)); GA = np.arccos(np.sqrt(q / K).sum())
    worst_ga = max(worst_ga, abs(renyi(q, 0.5) - (2 * np.log(np.cos(GA)) + np.log(K))))
check("Procrustes anisotropy (Dryden 2009): PA^2 = (3/2)(1 - exp(-D_1/2))", worst_pa, 0.0, 1e-12, "{:.1e}")
check("Cheng 2009 order 1/2: H_1/2 = 2 ln cos(GA) + ln K, GA = arccos <sqrt p, sqrt u>", worst_ga, 0.0, 1e-12, "{:.1e}")

# A7. fixed-FA interval of the Shannon eigenvalue entropy (Harremoes-Topsoe, K = 3)
P = np.vstack([rng.dirichlet([0.3] * 3, 300_000), rng.dirichlet([1] * 3, 300_000), rng.dirichlet([5] * 3, 200_000)])
def fa_rows(P):
    m = P.mean(1, keepdims=True); return np.sqrt(1.5 * ((P - m) ** 2).sum(1) / (P ** 2).sum(1))
def h1_rows(P):
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.where(P > 0, P * np.log(P), 0.0).sum(1)
t = np.linspace(0, 1, 200_001); one = np.ones_like(t)
fam = {"lin": np.stack([one, t, t], 1), "pla": np.stack([one, one, t], 1), "edg": np.stack([one, t, 0 * t], 1)}
fam = {k: v / v.sum(1, keepdims=True) for k, v in fam.items()}
def h1_at(kind, fq):
    f = fa_rows(fam[kind]); o = np.argsort(f); tq = np.interp(fq, f[o], t[o])
    lam = {"lin": np.stack([np.ones_like(tq), tq, tq], 1), "pla": np.stack([np.ones_like(tq), np.ones_like(tq), tq], 1),
           "edg": np.stack([np.ones_like(tq), tq, 0 * tq], 1)}[kind]
    return h1_rows(lam / lam.sum(1, keepdims=True))
f, h = fa_rows(P), h1_rows(P); keep = f < 0.999; f, h = f[keep], h[keep]
up = h1_at("lin", f)
lo = np.where(f <= 1 / np.sqrt(2), h1_at("pla", np.minimum(f, 1 / np.sqrt(2))), h1_at("edg", np.maximum(f, 1 / np.sqrt(2))))
check_true(f"all {keep.sum():,} random eigenvalue triples inside the linear/planar/zero-eigenvalue interval",
           bool(((h <= up + 1e-9) & (h >= lo - 1e-9)).all()))
w = lambda fq: float(h1_at("lin", np.array([fq]))[0] - h1_at("pla", np.array([fq]))[0])
check("interval width at FA 0.2", w(0.2), 0.0011, 0.00005, "{:.4f}")
check("interval width at FA 0.5", w(0.5), 0.025, 0.0006, "{:.4f}")
small = f < 0.05
check("order-1 member / (FA^2/3) near isotropy (FA < 0.05), median", float(np.median((np.log(3) - h[small]) / (f[small] ** 2 / 3))), 1.0, 0.01, "{:.4f}")

# A8. profile-tensor link: CV_D^2 = (2/5) CV_lambda^2 for a single-tensor profile on the sphere
def fib(n):
    i = np.arange(n) + 0.5; ph = np.arccos(1 - 2 * i / n); th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)], 1)
G = fib(200_000); worst = 0.0
for _ in range(50):
    lam = rng.uniform(0.1, 3.0, 3); Q, _ = np.linalg.qr(rng.normal(size=(3, 3))); T = Q @ np.diag(lam) @ Q.T
    D = np.einsum("ij,jk,ik->i", G, T, G)
    worst = max(worst, abs((D.var() / D.mean() ** 2) / (lam.var() / lam.mean() ** 2) - 0.4))
check("CV_D^2 / CV_lambda^2 on a dense sphere, 50 random tensors, max |ratio - 2/5|", worst, 0.0, 1e-4, "{:.1e}")
Ds = G[:, 0] ** 2  # linear tensor: normalized second moment 3<D_N^2> of the profile against sum rho^2 = 1
check("linear tensor: profile's normalized second moment / eigenvalues' = 3/5 (Ozarslan 2005)", float(3 * np.mean(Ds ** 2)), 0.6, 1e-4, "{:.4f}")

# A9. divergences are not metrics
A, B, C = (rng.dirichlet(np.full(6, 0.7), 20000) for _ in range(3))
def ren2(p, q, a):
    if a == 1:
        return (p * np.log(p / q)).sum(1)
    return np.log((p ** a * q ** (1 - a)).sum(1)) / (a - 1)
check("order 1/2 is symmetric, max |D(p||q) - D(q||p)|", float(np.max(np.abs(ren2(A, B, .5) - ren2(B, A, .5)))), 0.0, 1e-10, "{:.1e}")
check_true("orders 1, 2 and 4 are asymmetric", all(float(np.max(np.abs(ren2(A, B, a) - ren2(B, A, a)))) > 0.01 for a in (1, 2, 4)))
check_true("order 1/2 fails the triangle inequality on some triples", bool((ren2(A, C, .5) > ren2(A, B, .5) + ren2(B, C, .5) + 1e-12).any()))
bc = lambda p, q: np.clip(np.sqrt(p * q).sum(1), 0, 1)
check_true("Bhattacharyya angle and Hellinger distance satisfy it",
           bool(((np.arccos(bc(A, C)) <= np.arccos(bc(A, B)) + np.arccos(bc(B, C)) + 1e-9).all()) and
                ((np.sqrt(1 - bc(A, C)) <= np.sqrt(1 - bc(A, B)) + np.sqrt(1 - bc(B, C)) + 1e-9).all())))

# A10. scale (Discussion): what the monotone transform leaves alone and what it changes
dfa = lambda f: -np.log(1 - 2 * np.asarray(f, float) ** 2 / 3)  # ln 3 - H_2 of the eigenvalues, from Eq. FA
lam = rng.uniform(0.05, 3.0, (2000, 3)); pl = lam / lam.sum(1, keepdims=True)
check("ln 3 - H_2(lambda) = -ln(1 - 2 FA^2/3), max abs error", float(np.max(np.abs(np.log(3) + np.log((pl ** 2).sum(1)) - dfa(fa_rows(lam))))), 0.0, 1e-12, "{:.1e}")
check("order-2 deficit / (2/3 FA^2) at FA 0.01", float(dfa(0.01) / (2 * 0.01 ** 2 / 3)), 1.0, 1e-4, "{:.5f}")
check("order-2 deficit, change from FA 0 to 0.1", float(dfa(0.1) - dfa(0.0)), 0.007, 0.0005, "{:.4f}")
check("order-2 deficit, change from FA 0.5 to 0.8", float(dfa(0.8) - dfa(0.5)), 0.37, 0.005, "{:.3f}")
check("region at FA 0.1 and 0.7 in equal numbers: mean FA", (0.1 + 0.7) / 2, 0.40, 1e-12, "{:.2f}")
check("region at FA 0.1 and 0.7 in equal numbers: mean order-2 deficit", float((dfa(0.1) + dfa(0.7)) / 2), 0.201, 0.0005, "{:.4f}")
check("region at FA 0.45 throughout: order-2 deficit", float(dfa(0.45)), 0.145, 0.0005, "{:.4f}")
from scipy.stats import rankdata
cohen = lambda a, b: (b.mean() - a.mean()) / np.sqrt((a.var() + b.var()) / 2)
def auc(a, b):
    r = rankdata(np.concatenate([a, b])); return (r[a.size:].sum() - b.size * (b.size + 1) / 2) / (a.size * b.size)
worst_d, worst_auc = 0.0, 0.0
for m in (0.20, 0.45):  # two groups 0.02 apart in FA, between-subject SD 0.03
    g1, g2 = np.abs(rng.normal(m, 0.03, 100_000)), np.abs(rng.normal(m + 0.02, 0.03, 100_000))
    worst_d = max(worst_d, abs(cohen(dfa(g1), dfa(g2)) / cohen(g1, g2) - 1)); worst_auc = max(worst_auc, abs(auc(dfa(g1), dfa(g2)) - auc(g1, g2)))
check("group comparison: same AUC on FA and deficit scales, max difference", worst_auc, 0.0, 1e-12, "{:.1e}")
check("group comparison at FA 0.2 and 0.45: standardized effect ratio, max |ratio - 1|", worst_d, 0.0, 0.01, "{:.4f}")
pr = lambda l: float(1 / ((np.asarray(l, float) / np.sum(l)) ** 2).sum())
check_true("PR_lambda = e^H_2 is 1 for a stick, 2 for a flat disc and 3 for a sphere",
           np.allclose([pr([1, 0, 0]), pr([1, 1, 0]), pr([1, 1, 1])], [1, 2, 3]))
check("PR_lambda of the strong fiber (1.7, 0.3, 0.3)", pr([1.7, 0.3, 0.3]), 1.72, 0.005, "{:.3f}")
worst = 0.0
for _ in range(500):  # chain rule of Shannon entropy over any grouping of the directions
    K = int(rng.integers(6, 120)); x = rng.uniform(0.05, 5.0, K); p = x / x.sum(); grp = rng.integers(0, int(rng.integers(2, 6)), K)
    within = between = 0.0
    for g in np.unique(grp):
        m = grp == g; P = p[m].sum(); n = int(m.sum())
        within += P * (np.log(n) - renyi(p[m] / P, 1)); between += P * np.log(P / (n / K))
    worst = max(worst, abs((np.log(K) - renyi(p, 1)) - within - between))
check("order-1 deficit = share-weighted within-group deficits + between-group divergence", worst, 0.0, 1e-12, "{:.1e}")
worst, tried = 0.0, 0
while tried < 500:  # order 2 stays defined, and exact, when some values are negative
    K = int(rng.integers(3, 100)); x = rng.normal(1.0, 0.7, K)
    if x.mean() <= 0 or (x > 0).all():
        continue
    tried += 1; p = x / x.sum(); cv2 = x.var() / x.mean() ** 2
    worst = max(worst, abs(np.log(K) + np.log((p ** 2).sum()) - np.log1p(cv2)))
lneg = np.array([1.5, 0.4, -0.1]); cvn = lneg.var() / lneg.mean() ** 2
worst = max(worst, abs(fa(lneg) ** 2 - 1.5 * cvn / (1 + cvn)))
check("order 2 with negative values: identity and FA relation, max error", worst, 0.0, 1e-12, "{:.1e}")
with np.errstate(invalid="ignore"):
    check_true("order 1 is undefined with a negative share", bool(np.isnan(-(lneg / lneg.sum() * np.log(lneg / lneg.sum())).sum())))

# =============================================================================================
print("\nB. Numbers quoted from the analysis outputs")
s = pd.read_csv(HERE / "synthetic_profiles.csv").set_index("case")
TABLE = {  # case: FA, CV_D, H~1, H~2, H~inf   (as printed in Table 1)
    "single fiber, strong": (0.799, 0.544, 0.968, 0.943, 0.828),
    "single fiber, weak":   (0.213, 0.112, 0.999, 0.997, 0.952),
    "planar (oblate)":      (0.522, 0.298, 0.989, 0.981, 0.937),
    "crossing 90 deg":      (0.412, 0.250, 0.993, 0.987, 0.914),
    "crossing 60 deg":      (0.595, 0.360, 0.986, 0.973, 0.856),
}
worst = 0.0
for case, vals in TABLE.items():
    r = s.loc[case]; got = (r.FA_fit, r.CV_D, r.Hn_1, r.Hn_2, r.Hn_inf)
    worst = max(worst, max(abs(round(g, 3) - v) for g, v in zip(got, vals)))
check("Table 1, max difference from synthetic_profiles.csv (3 d.p.)", worst, 0.0, 1e-9, "{:.3f}")
single = [c for c in TABLE if c.startswith("single") or c.startswith("planar")]
ratios = [s.loc[c].CV_D ** 2 / (s.loc[c].FA_fit ** 2 / (1.5 - s.loc[c].FA_fit ** 2)) for c in single]
check("CV_D^2 / CV_lambda^2, single-tensor voxels on 93 directions, max |ratio - 2/5|", max(abs(r - 0.4) for r in ratios), 0.0, 0.001, "{:.4f}")
cv2s = s.loc["single fiber, strong"].CV_D ** 2
check("second-order form overstates the order-2 member at CV 0.544 (fraction)", cv2s / np.log1p(cv2s) - 1, 0.14, 0.005, "{:.3f}")
check("strong single fiber CV_D", float(s.loc["single fiber, strong"].CV_D), 0.544, 0.0005, "{:.4f}")
tab = s.loc[[c for c in s.index if c not in ("isotropic", "three-way crossing")]]
check("Table 1: H~2 = 1 - ln(1 + CV_D^2)/ln N, max error", float(np.max(np.abs(tab.Hn_2 - (1 - np.log1p(tab.CV_D ** 2) / np.log(93))))), 0.0, 1e-9, "{:.1e}")
check_true("Table 1: CV_D and H~2 rank the voxels in exactly opposite order",
           list(tab.CV_D.sort_values().index) == list(tab.Hn_2.sort_values(ascending=False).index))
pl, cr = tab.loc["planar (oblate)"], tab.loc["crossing 90 deg"]
check_true("Table 1: planar vs 90 deg crossing, larger CV_D and smaller H~2 but larger H~inf",
           bool(pl.CV_D > cr.CV_D and pl.Hn_2 < cr.Hn_2 and pl.Hn_inf > cr.Hn_inf))

check("Figure 2 H_2 = ln(3 - 2 FA^2): tissue identity error (from CSV, normalized)", float(fe_max) if (fe_max := pd.read_csv(HERE / "fa_vs_eigen_entropy.csv").identity_order2_max_abs_err.max()) is not None else 1.0, 0.0, 1e-12, "{:.1e}")

rm = pd.read_csv(HERE / "reflections_maps_check.csv")
check("Figure 3, voxelwise identities, max abs error", float(rm.max_abs_err.max()), 0.0, 1e-14, "{:.1e}")
check("Figure 3, voxels checked, both rows about 387,000", float(abs(rm.n_vox.iloc[:2] - 387000).max()), 0, 1000, "{:.0f}")

fe = pd.read_csv(HERE / "fa_vs_eigen_entropy.csv")
check("tissue voxels inside the interval, min share over subjects", float(fe.frac_inside_order1_envelope.min()), 1.0, 1e-9, "{:.4f}")
check("rho(FA, Shannon eigenvalue entropy), weakest subject", float(fe.rho_FA_H1.abs().min()), 0.9998, 0.00006, "{:.5f}")
check("Sec 3.2 voxels ordered by mode: rho(mode, position), FA < 1/sqrt 2, weakest", float(fe.rho_mode_vs_band_position.min()), 0.9998, 0.00006, "{:.5f}")
check("median tissue FA, lowest subject", float(fe.median_FA.min()), 0.19, 0.005, "{:.3f}")
check("median tissue FA, highest subject", float(fe.median_FA.max()), 0.21, 0.005, "{:.3f}")
check_true("tensors are b = 1500 fits (median brain MD > 0.65e-3; the two-shell default gives about 0.57e-3)",
           bool((fe.median_MD_brain > 0.65e-3).all()))

tc = pd.read_csv(HERE / "tensor_concept.csv").set_index("tensor")
iso, pro, obl = tc.loc["isotropic"], tc.loc["prolate (linear)"], tc.loc["oblate (planar)"]
check("Figure 1: isotropic tensor, H_1 = H_2 = ln 3", max(abs(iso.H1 - np.log(3)), abs(iso.H2 - np.log(3))), 0.0, 1e-12, "{:.1e}")
check("Figure 1: prolate and oblate share FA 0.6", max(abs(pro.FA - 0.6), abs(obl.FA - 0.6)), 0.0, 1e-9, "{:.1e}")
check("Figure 1: same FA gives the same H_2 and PR_lambda", max(abs(pro.H2 - obl.H2), abs(pro.PR - obl.PR)), 0.0, 1e-9, "{:.1e}")
check_true("Figure 1: prolate and oblate differ in H_1 (0.952 against 0.895), modes +1 and -1",
           bool(round(pro.H1, 3) == 0.952 and round(obl.H1, 3) == 0.895 and abs(pro["mode"] - 1) < 1e-9 and abs(obl["mode"] + 1) < 1e-9))

check("Sec 3.1 worked example: prolate p = (0.598, 0.201, 0.201)", max(abs(pro.p1 - 0.598), abs(pro.p2 - 0.201), abs(pro.p3 - 0.201)), 0.0, 0.0005, "{:.4f}")
check("Sec 3.1 worked example: sum p^2 = 0.439", float(pro.p1 ** 2 + pro.p2 ** 2 + pro.p3 ** 2), 0.439, 0.0005, "{:.4f}")
check("Sec 3.1 worked example: H_2 = 0.824, PR = 2.28", max(abs(pro.H2 - 0.824), abs(pro.PR - 2.28) / 10), 0.0, 0.0005, "{:.4f}")
check("Sec 3.1 worked example: FA from PR, sqrt(3/2 (1 - PR/3)) = 0.60", float(np.sqrt(1.5 * (1 - pro.PR / 3))), 0.60, 0.0005, "{:.4f}")

# Sec 4.1: signal-domain against diffusivity-domain entropy deficits, near isotropy
def _fib(n):
    i = np.arange(n) + 0.5; ph = np.arccos(1 - 2 * i / n); th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)], 1)
_G = _fib(93); _b = 1500.0
def _deficits(l):
    Dp = np.einsum("ij,jk,ik->i", _G, np.diag(np.asarray(l) * 1e-3), _G)
    out = []
    for x in (Dp, np.exp(-_b * Dp)):
        q = x / x.sum(); out.append(np.log(len(x)) + (q * np.log(q)).sum())
    return out[0], out[1], (_b * Dp.mean()) ** 2
dD, dS, bd2 = _deficits((0.85, 0.78, 0.77))
check("Sec 4.1 near isotropy: signal/diffusivity deficit ratio vs (b Dbar)^2", float(dS / dD / bd2), 1.0, 0.02, "{:.3f}")

ro_d = pd.read_csv(HERE / "renyi_order_test.csv"); ro_d = ro_d[ro_d.domain == "diffusivity"].set_index("alpha")
check_true("Sec 5: split-half reliability 0.95 to 0.96 for orders 1 to 5",
           bool(all(0.945 <= ro_d.loc[a, "split_half"] < 0.965 for a in (1.0, 2.0, 3.0, 5.0))))
check("Sec 5: split-half reliability at order 10", float(ro_d.loc[10.0, "split_half"]), 0.93, 0.005, "{:.3f}")

ac = pd.read_csv(HERE / "acquisition_counts.csv")
check_true("Sec 5: 93 directions at b = 1500 in all four subjects", bool((ac.n_b1500 == 93).all()) and len(ac) == 4)
check_true("Sec 5: 14 b ~ 0 volumes, interleaved through the series (largest gap under 10% of the volumes)",
           bool((ac.n_b0 == 14).all() and (ac.largest_b0_gap < 0.1 * ac.n_volumes).all() and (ac.last_b0 > 0.8 * ac.n_volumes).all()))
check("Sec 5: near-uniform directions, largest |<g g^T> - I/3|, worst subject", float(ac.second_moment_max_dev.max()), 0.0, 0.02, "{:.4f}")

ro = pd.read_csv(HERE / "renyi_order_test.csv")
r1 = ro[(ro.domain == "diffusivity") & (ro.alpha == 1.0)].iloc[0]
check("order-1 vs order-2 directional entropy, voxelwise correlation", float(r1.corr_with_H2), 0.998, 0.0006, "{:.3f}")

ol = pd.read_csv(HERE / "order1_link.csv").set_index("quantity").value
check("order-1 link: Jensen violations over the eigenvalue simplex", float(ol["Jensen violations, profile deficit > eigenvalue deficit"]), 0, 0, "{:.0f}")
check("order-1 link: lowest profile/eigenvalue deficit ratio", float(ol["ratio min (FA > 0.02)"]), 0.31, 0.005, "{:.3f}")
check_true("order-1 link: lowest ratio at the flat disc (mode -1, FA 1/sqrt 2)",
           abs(ol["ratio min at mode"] + 1) < 1e-6 and abs(ol["ratio min at FA"] - 2 ** -0.5) < 1e-3)
check("order-1 link: highest ratio", float(ol["ratio max (FA > 0.02)"]), 0.42, 0.005, "{:.3f}")
check_true("order-1 link: highest ratio on prolate tensors (mode +1)", abs(ol["ratio max at mode"] - 1) < 1e-6)
check("order-1 link: median ratio near isotropy (FA < 0.05)", float(ol["ratio median, FA < 0.05"]), 0.4, 0.002, "{:.4f}")
check("order-1 link: strong fiber ratio", float(ol["strong fiber (1.7, 0.3, 0.3): ratio"]), 0.42, 0.005, "{:.3f}")
check("order-1 link: stick profile deficit = ln 3 - 2/3 (Ozarslan 2005 infimum)", float(ol["stick (1, 0, 0): profile deficit"]), np.log(3) - 2 / 3, 1e-4, "{:.5f}")

dc = pd.read_csv(HERE / "direction_count.csv").set_index("N")
check_true("direction counts span 30 to 256", dc.index.min() == 30 and dc.index.max() == 256)
check("strong fiber on 30-256 directions: lowest CV_D", float(dc.CV_D.min()), 0.539, 0.0005, "{:.4f}")
check("strong fiber on 30-256 directions: highest CV_D", float(dc.CV_D.max()), 0.544, 0.0005, "{:.4f}")
check("strong fiber on 30-256 directions: lowest order-2 deficit", float(dc.deficit_2.min()), 0.255, 0.0005, "{:.4f}")
check("strong fiber on 30-256 directions: highest order-2 deficit", float(dc.deficit_2.max()), 0.260, 0.0005, "{:.4f}")
check("normalized order-2 entropy, 30 directions", float(dc.loc[30].Hn_2), 0.925, 0.0005, "{:.4f}")
check("normalized order-2 entropy, 256 directions", float(dc.loc[256].Hn_2), 0.953, 0.0005, "{:.4f}")
check_true("1 - H~2 falls by more than a third from 30 to 256 directions",
           (1 - dc.loc[256].Hn_2) / (1 - dc.loc[30].Hn_2) < 2 / 3)
spread = lambda s: float((s.max() - s.min()) / s.max())
check_true("orders 1/2 to 4: deficits vary by under 3% with N, 1 - H~ by over 30%",
           all(spread(dc[f"deficit_{a}"]) < 0.03 and spread(1 - dc[f"Hn_{a}"]) > 0.30 for a in ("0.5", "1", "2", "4")))

print()
if FAILS:
    print(f"FAILED {len(FAILS)}: {FAILS}")
    sys.exit(1)
print("ALL CHECKS PASSED")
