"""Does going to higher Renyi order add anything, or just add noise?
For a few subjects, compute the normalized Renyi entropy H~_alpha at
alpha = 1,2,3,5,10 in both domains (diffusivity p~D, signal p~e^{-bD}),
voxelwise in brain, and report:
  (1) REDUNDANCY: correlation of H~_alpha with H~_2 (is higher order new info?)
  (2) NOISE: split-half reliability r(H~_alpha on two disjoint direction halves)
      -- higher alpha weights the single most extreme direction, so it should
      grow less reliable as alpha rises.
"""
import os
from pathlib import Path
import numpy as np, nibabel as nib, pandas as pd

HERE = Path(__file__).resolve().parent                      # this repo's analysis/ (outputs)
from paths import DTI_OUTPUT as OUT, MANIFEST
SHELL = 1500
ALPHAS = [1.0, 2.0, 3.0, 5.0, 10.0]
rng = np.random.default_rng(0)


def renyi_norm(p, alpha):
    # p: (..., n) probabilities over directions; returns H_alpha/ln n
    n = p.shape[-1]
    if abs(alpha - 1.0) < 1e-9:
        H = -np.sum(p * np.log(np.maximum(p, 1e-20)), axis=-1)
    else:
        H = (1.0 / (1.0 - alpha)) * np.log(np.sum(np.maximum(p, 1e-20) ** alpha, axis=-1))
    return H / np.log(n)


def dists(dwi_s, bvals_s, brain):
    """Angular distributions for tissue voxels. Voxels are kept only if no
    direction hit the signal floor/ceiling (i.e. no clipped or zero D_i) and the
    mean diffusivity is in a tissue range (0.4-1.5e-3 mm^2/s), which excludes CSF
    and noise-dominated voxels. Without this guard the alpha=1 diffusivity-domain
    entropy is dominated by a few zero-diffusivity directions in non-tissue voxels
    and its whole-brain correlation with alpha=2 is spurious (-0.4), whereas in
    white matter the two agree at r>0.99 (ROI tables)."""
    b0 = np.mean(dwi_s[..., bvals_s < 50], axis=-1)
    sh = bvals_s >= 50
    Sraw = dwi_s[..., sh] / (b0[..., None] + 1e-10)
    Sp = np.clip(Sraw, 1e-2, 1.0)
    shb = float(np.mean(bvals_s[sh]))
    D = np.clip(-(1.0 / shb) * np.log(Sp), 0.0, 5e-3)
    Dbar = D.mean(axis=-1)
    valid = brain.astype(bool) & np.all((Sraw > 1e-2) & (Sraw < 1.0), axis=-1) & (Dbar > 0.4e-3) & (Dbar < 1.5e-3)
    pD = (D / (np.sum(D, axis=-1, keepdims=True) + 1e-20))[valid]
    pS = (Sp / (np.sum(Sp, axis=-1, keepdims=True) + 1e-20))[valid]
    return pD, pS  # (nvox, n)


def half_reliability(p, alpha):
    n = p.shape[-1]; idx = rng.permutation(n); h1, h2 = idx[:n // 2], idx[n // 2:2 * (n // 2)]
    a = renyi_norm(p[:, h1] / p[:, h1].sum(1, keepdims=True), alpha)
    b = renyi_norm(p[:, h2] / p[:, h2].sum(1, keepdims=True), alpha)
    g = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[g], b[g])[0, 1]


man = pd.read_csv(MANIFEST, sep="\t").head(4)
red = {"D": {a: [] for a in ALPHAS}, "S": {a: [] for a in ALPHAS}}
rel = {"D": {a: [] for a in ALPHAS}, "S": {a: [] for a in ALPHAS}}
for _, r in man.iterrows():
    sess = r["session_id"]
    proc = OUT / sess / "processed"; inp = OUT / sess / "inputs"
    dwi = np.asarray(nib.load(str(proc / "dwi_raw.nii.gz")).dataobj, dtype=np.float32)
    bvals = np.loadtxt(inp / "dwi.bval").astype(float)
    brain = np.asarray(nib.load(str(proc / "mask.nii.gz")).dataobj).astype(bool)
    keep = (np.abs(bvals - min(sorted(set(np.round(bvals[bvals >= 50] / 500) * 500)),
                                key=lambda s: abs(s - SHELL))) < 50) | (bvals < 50)
    pD, pS = dists(dwi[..., keep], bvals[keep], brain)
    for tag, p in (("D", pD), ("S", pS)):
        H = {a: renyi_norm(p, a) for a in ALPHAS}
        for a in ALPHAS:
            g = np.isfinite(H[a]) & np.isfinite(H[2.0])
            red[tag][a].append(np.corrcoef(H[a][g], H[2.0][g])[0, 1])
            rel[tag][a].append(half_reliability(p, a))

print(f"n_dir at b{SHELL}: {int(((np.abs(bvals-1500)<50)).sum())}  (subjects=4)\n")
for tag, name in (("D", "DIFFUSIVITY  p~D"), ("S", "SIGNAL  p~e^-bD")):
    print(f"=== {name} ===")
    print(f"  {'alpha':>6} {'corr with H~_2 (redundancy)':>28} {'split-half reliability (noise)':>32}")
    for a in ALPHAS:
        print(f"  {a:>6.1f} {np.mean(red[tag][a]):>28.3f} {np.mean(rel[tag][a]):>32.3f}")
    print()

# persist (paper rule: no number without a script and a CSV behind it)
rows = [dict(domain={"D": "diffusivity", "S": "signal"}[tag], alpha=a,
             corr_with_H2=float(np.mean(red[tag][a])), split_half=float(np.mean(rel[tag][a])),
             n_subjects=len(red[tag][a]))
        for tag in ("D", "S") for a in ALPHAS]
pd.DataFrame(rows).to_csv(HERE / "renyi_order_test.csv", index=False)
print("written -> analysis/renyi_order_test.csv")
