"""Reflections figure: each anisotropy index next to the entropy it mirrors, on one
representative de-identified HCP-A subject (b = 1500, single axial slice).

Row 1, the directional diffusivities (N = 93 directions):
  J_quad = CV_D^2          (quadratic Jensen gap = chi^2 divergence from isotropy)
  PR/N   = 1/(1 + CV_D^2)  (its exact order-2 mirror, the normalized participation ratio)
  H~_1                     (normalized Shannon entropy of the directional distribution,
                            the order-1 reading of the same divergence)
Row 2, the tensor eigenvalues (K = 3):
  FA                       (the same construction on the normalized eigenvalues)
  PR_lambda/3              (its exact order-2 mirror: FA^2 = (3/2)(1 - PR_lambda/3))
  H~_1(lambda)             (normalized Shannon entropy of the eigenvalue distribution;
                            exp(H_1) in [1, 3] is the effective number of principal
                            directions)
Every identity is asserted voxelwise before plotting. Per-direction diffusivities are
D_i = -ln(S_i / S_0) / b on the b = 1500 volumes, S_0 the mean of the b = 0 volumes,
restricted to the brain mask and to voxels with all S_i > 0. Eigenvalues are clipped at
zero for the shares. No participant identifier appears in the figure.
Reads DTI_OUTPUT_DIR/<session>/{inputs/dwi.bval, processed/dwi_raw.nii.gz,
processed/dwi_raw.bvec, processed/mask.nii.gz}. Eigenvalues come from b1500_tensor.py, a DIPY
weighted least-squares fit to the b = 0 and b = 1500 volumes.
Writes reflections_maps.{png,pdf} to the figures folder of paths.py and
analysis/reflections_maps_check.csv.
"""
import os
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from b1500_tensor import b1500_eigenvalues
from paths import DTI_OUTPUT, MANIFEST, FIGURES

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "analysis"
# the second participant of Figure 1, chosen by position in the manifest (identifiers are not published)
SESSION = DTI_OUTPUT / pd.read_csv(MANIFEST, sep="\t").session_id.iloc[1]
B, TOL, SLICE_Z = 1500.0, 150.0, 60


def load():
    bval = np.loadtxt(SESSION / "inputs/dwi.bval")
    dwi = nib.load(str(SESSION / "processed/dwi_raw.nii.gz"))
    img = np.asarray(dwi.dataobj, dtype=np.float32)
    mask = np.asarray(nib.load(str(SESSION / "processed/mask.nii.gz")).dataobj) > 0
    lam, _ = b1500_eigenvalues(SESSION.name)
    return bval, img, mask, lam


def directional_maps(bval, img, mask):
    b0 = img[..., np.abs(bval) < TOL].mean(-1)
    S = img[..., np.abs(bval - B) < TOL]
    N = S.shape[-1]
    ok = mask & (b0 > 0) & np.all(S > 0, axis=-1)
    D = np.full(S.shape, np.nan, np.float64)
    D[ok] = -np.log(S[ok] / b0[ok][:, None]) / B
    D[ok] = np.clip(D[ok], 1e-6, None)
    Dbar = D.mean(-1)
    cv2 = D.var(-1) / Dbar ** 2
    p = D / D.sum(-1, keepdims=True)
    H1 = -(p * np.log(p)).sum(-1) / np.log(N)
    PRN = 1.0 / (N * (p ** 2).sum(-1))
    return dict(Jquad=cv2, PRN=PRN, H1=H1, N=N, ok=ok)


def eigen_maps(lam, mask):
    lam = np.clip(lam, 0, None)
    ok = mask & (lam.sum(-1) > 0)
    p = np.full(lam.shape, np.nan); p[ok] = lam[ok] / lam[ok].sum(-1, keepdims=True)
    m = lam.mean(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        FA = np.sqrt(1.5 * ((lam - m[..., None]) ** 2).sum(-1) / (lam ** 2).sum(-1))
        PRl3 = 1.0 / (3 * (p ** 2).sum(-1))
        pc = np.clip(p, 1e-300, None)
        H1l = -(pc * np.log(pc)).sum(-1) / np.log(3)
    return dict(FA=FA, PRl3=PRl3, H1l=H1l, ok=ok)


def check_identities(dm, em):
    ok = dm["ok"]; rows = []
    e1 = np.nanmax(np.abs(dm["PRN"][ok] - 1 / (1 + dm["Jquad"][ok])))
    rows.append(dict(identity="PR/N = 1/(1+CV^2), directional", max_abs_err=e1, n_vox=int(ok.sum())))
    ok2 = em["ok"] & np.isfinite(em["FA"])
    e2 = np.nanmax(np.abs(em["FA"][ok2] ** 2 - 1.5 * (1 - em["PRl3"][ok2])))
    rows.append(dict(identity="FA^2 = (3/2)(1 - PR_lambda/3), eigenvalues", max_abs_err=e2, n_vox=int(ok2.sum())))
    per = np.exp(em["H1l"][ok2] * np.log(3))
    rows.append(dict(identity="eigenvalue perplexity in [1,3]", max_abs_err=float(max(1 - per.min(), per.max() - 3, 0)), n_vox=int(ok2.sum())))
    df = pd.DataFrame(rows); df.to_csv(HERE / "reflections_maps_check.csv", index=False)
    print(df.to_string(index=False))
    assert e1 < 1e-9 and e2 < 1e-6, "identity check failed"
    return df


def panel(ax, bg, mp, ok, vmin, vmax, cmap, title):
    ax.imshow(bg.T, cmap="gray", origin="lower", vmin=0, vmax=1)
    im = ax.imshow(np.ma.masked_where(~ok.T, mp.T), cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    ax.set_title(title, fontsize=7); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=6, width=0.5, length=2); cb.outline.set_linewidth(0.4)


def main():
    bval, img, mask, lam = load()
    dm = directional_maps(bval, img, mask); em = eigen_maps(lam, mask)
    check_identities(dm, em)
    z = SLICE_Z; bg = np.nan_to_num(em["FA"][:, :, z]); okd = dm["ok"][:, :, z]; oke = em["ok"][:, :, z] & np.isfinite(em["FA"][:, :, z])
    # drawn at its printed size (5.5 in wide) so font sizes are the printed sizes; include without scaling
    fig, axes = plt.subplots(2, 3, figsize=(5.5, 3.75), layout="constrained")
    specs = [
        (em["FA"][:, :, z], oke, 0.0, 0.9, "viridis", r"FA"),
        (em["PRl3"][:, :, z], oke, 0.45, 1.0, "viridis", r"$\mathrm{PR}_{\lambda}/3$  (order-2 mirror)"),
        (em["H1l"][:, :, z], oke, 0.5, 1.0, "viridis", r"$\tilde H_{1}(\lambda)$  (order 1)"),
        (dm["Jquad"][:, :, z], okd, 0.0, 0.5, "magma", r"$\mathrm{CV}_D^{2}$"),
        (dm["PRN"][:, :, z], okd, 0.65, 1.0, "magma", r"$\mathrm{PR}/N$  (order-2 mirror)"),
        (dm["H1"][:, :, z], okd, 0.95, 1.0, "magma", r"$\tilde H_{1}$  (order 1)"),
    ]
    for ax, (mp, ok, lo, hi, cmap, title) in zip(axes.flat, specs):
        panel(ax, bg, mp, ok, lo, hi, cmap, title)
    axes[0, 0].set_ylabel("tensor eigenvalues ($K=3$)", fontsize=7)
    axes[1, 0].set_ylabel(f"directional diffusivities ($N={dm['N']}$)", fontsize=7)
    FIGURES.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"reflections_maps.{ext}", dpi=300)
    print(f"written -> {FIGURES}/reflections_maps.{{png,pdf}}, analysis/reflections_maps_check.csv")


if __name__ == "__main__":
    main()
