"""Check the ISMRM 2027 abstract's synthetic-profile numbers against synthetic_profiles.csv.

identity_checks.py checks every relation and number of the paper, including the identity in every
form at every order on 2,000 random sets (check A1), the Shannon limit among them, and the single-fiber
and 90-degree crossing rows of Table 1. The abstract adds one number the paper does not quote, the
three-way orthogonal crossing: tensor FA 0.000 while the directional profile keeps CV_D 0.092. This
script checks that row and repeats the Table 1 values the abstract quotes, as they appear in the text.
Exits non-zero on any failure.
Usage: python analysis/abstract_checks.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
s = pd.read_csv(HERE / "synthetic_profiles.csv").set_index("case")
FAILS = []


def one(label, value, quoted, nd):
    ok = round(float(value), nd) == quoted
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label:58s} {round(float(value), nd)}  (quoted {quoted})")
    if not ok:
        FAILS.append(label)


one("strong single fiber, tensor FA", s.loc["single fiber, strong", "FA_fit"], 0.799, 3)
one("strong single fiber, CV_D", s.loc["single fiber, strong", "CV_D"], 0.544, 3)
one("strong single fiber, normalized Shannon entropy", s.loc["single fiber, strong", "Hn_1"], 0.968, 3)
one("90-degree crossing, tensor FA", s.loc["crossing 90 deg", "FA_fit"], 0.412, 3)
one("90-degree crossing, CV_D", s.loc["crossing 90 deg", "CV_D"], 0.250, 3)
one("90-degree crossing, normalized Shannon entropy", s.loc["crossing 90 deg", "Hn_1"], 0.993, 3)
one("three-way orthogonal crossing, tensor FA", s.loc["three-way crossing", "FA_fit"], 0.000, 3)
one("three-way orthogonal crossing, CV_D", s.loc["three-way crossing", "CV_D"], 0.092, 3)

# Meningioma replication (ogm_replication.py). Data sentence: Philips 3 T, b = 1500, 64 directions, one b0,
# 34 preoperative patients, 27 with segmented edema. Results: rho -0.9994 to -0.9998, r 0.994 to 0.997, edema
# deficits |rho| >= 0.995 among CV_D^2, PR/N, H1 and H2, MD |rho| < 0.6. Notes: the old window gives
# 0.989 to 0.991, the identity ceiling keeps a median 58% of each edema, and the 2.0e-3 ceiling keeps 99%
# with |rho| >= 0.993 and MD <= 0.63. Bounds are checked as strict inequalities, not after rounding.


def check(label, ok, got):
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label:58s} {got}")
    if not ok:
        FAILS.append(label)


o = pd.read_csv(HERE / "ogm_replication.csv")
check("34 preoperative patients, Philips 3 T", len(o) == 34 and (o.manufacturer == "Philips").all() and (o.field_T == 3).all(),
      f"{len(o)} {set(o.manufacturer)} {set(o.field_T)} T")
check("all preoperative (no sub-olfNNfu), sub-olf21 (truncated file) absent",
      (~o.subject.str.endswith("fu")).all() and "sub-olf21" not in set(o.subject), sorted(o.subject)[:2])
check("b = 1500, 64 directions, one b0", (o.b == 1500).all() and (o.n_dir == 64).all() and (o.n_b0 == 1).all(),
      f"{set(o.b)} {set(o.n_dir)} {set(o.n_b0)}")
rng = lambda v, nd: (round(float(v.min()), nd), round(float(v.max()), nd))
check("rho(FA, Shannon eigenvalue entropy) -0.9994 to -0.9998", rng(o.rho_FA_H1, 4) == (-0.9998, -0.9994), rng(o.rho_FA_H1, 4))
check("r(order-1, order-2 directional entropy) 0.994 to 0.997", rng(o.r_H1_H2_D, 3) == (0.994, 0.997), rng(o.r_H1_H2_D, 3))
check("same without tumor and edema, 0.994 to 0.997", rng(o.r_H1_H2_D_no_lesion, 3) == (0.994, 0.997), rng(o.r_H1_H2_D_no_lesion, 3))
f4 = o[o.first_four]
check("notes: the first four give r 0.996", len(f4) == 4 and rng(f4.r_H1_H2_D, 3) == (0.996, 0.996), rng(f4.r_H1_H2_D, 3))
check("notes: old window reproduces 0.989 to 0.991 in the first four", rng(f4.r_H1_H2_D_infotheory_rule, 3) == (0.989, 0.991),
      rng(f4.r_H1_H2_D_infotheory_rule, 3))
ep = pd.read_csv(HERE / "ogm_replication_edema.csv"); ec = pd.read_csv(HERE / "ogm_replication_edema_corr.csv")
check("the 27 edema patients are among the 34", set(ep[ep.d_MD.notna()].subject) <= set(o.subject), len(set(ep.subject)))
for rule, lo, md_hi, kept in (("identity", 0.995, 0.6, 0.58), ("csf2", 0.993, 0.63, 0.99)):
    e = ep[(ep.rule == rule) & ep.d_MD.notna()]; c = ec[ec.rule == rule]
    within = c[(c.a != "MD") & (c.b != "MD")].rho.abs(); with_md = c[(c.a == "MD") | (c.b == "MD")].rho.abs()
    check(f"{rule}: 27 patients with segmented edema", len(e) == 27 and (c.n_patients == 27).all(), len(e))
    check(f"{rule}: |rho| >= {lo} among CV_D^2, PR/N, H1, H2", within.min() >= lo, round(within.min(), 4))
    check(f"{rule}: |rho| with MD {'<' if rule == 'identity' else '<='} {md_hi}",
          with_md.max() < md_hi if rule == "identity" else with_md.max() <= md_hi, round(with_md.max(), 4))
    check(f"{rule}: median share of each edema kept {kept:.2f}", round(e.frac_edema_kept.median(), 2) == kept,
          round(e.frac_edema_kept.median(), 3))

# Not in the submitted abstract (Hark's ECHO #01587 kept the shorter PR sentence). Kept for later use, for
# example the ASNR pitfalls exhibit.
# Discussion: PR_lambda = exp(H_2) = 3 - 2 FA^2 on the tensor eigenvalues, from 1 (line) to 3 (isotropy).
# It is the Methods relation FA^2 = (3/2)(1 - exp(-D_2)) with D_2 = ln 3 - H_2. Checked on 2,000 random
# eigenvalue sets and at the line, plane and sphere, then quoted from the fitted FA of one strong fiber
# (1.72) and the 90-degree crossing (2.66).
pr_lam = lambda l: l.sum(-1) ** 2 / (l ** 2).sum(-1)
fa_sq = lambda l: 1.5 * ((l - l.mean(-1, keepdims=True)) ** 2).sum(-1) / (l ** 2).sum(-1)
lam = np.random.default_rng(0).uniform(0.01, 3.0, (2000, 3))
err = np.abs(pr_lam(lam) - (3 - 2 * fa_sq(lam))).max()
check("PR_lambda = exp(H_2) = 3 - 2 FA^2 on 2,000 random sets", err < 1e-12, f"max error {err:.1e}")
ends = np.array([[1.0, 0, 0], [1.0, 1.0, 0], [1.0, 1.0, 1.0]])
check("PR_lambda 1 for a line, 2 for a plane, 3 for isotropy", np.allclose(pr_lam(ends), [1, 2, 3]), pr_lam(ends))
one("one strong fiber, PR_lambda = 3 - 2 FA^2", 3 - 2 * s.loc["single fiber, strong", "FA_fit"] ** 2, 1.72, 2)
one("90-degree crossing, PR_lambda = 3 - 2 FA^2", 3 - 2 * s.loc["crossing 90 deg", "FA_fit"] ** 2, 2.66, 2)
print()
if FAILS:
    print(f"FAILED {len(FAILS)}: {FAILS}"); sys.exit(1)
print("ALL ABSTRACT NUMBERS MATCH")
