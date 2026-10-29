"""Replication of the abstract's two empirical tissue numbers on a second vendor and protocol.

The identity is exact and needs no replication. Two numbers the ISMRM 2027 abstract quotes are empirical,
because they depend on the tissue's FA range and on noise:
  r(H~_1, H~_2)     voxelwise correlation of the order-1 and order-2 normalized directional entropies
                    of the diffusivity profile (renyi_order_test.py, HCP-Aging 0.998)
  rho(FA, H~_1)     Spearman correlation of FA with the normalized Shannon entropy of the tensor
                    eigenvalues (fa_vs_eigen_entropy.py, HCP-Aging -0.9998 to -0.9999)
This script recomputes both on OpenNeuro ds007345 (Filimonova et al., Front Radiol 2026, CC0), an
olfactory groove meningioma dataset from a Philips Ingenia 3 T: b = 1500 s/mm^2, 64 directions and one
b = 0 volume, 2 mm slices. It uses every preoperative session with the dataset's own topup/eddy-corrected
DWI (derivatives/<sub>/dwi_processing): 35 of the 37 have one (sub-olf25 and sub-olf36 do not), and
sub-olf21's is a truncated 29 kB gzip, which the script skips and reports, so 34 are used. The sub-olfNNfu
sessions are postoperative and are not used. Until 27 October 2026 Part 1 used only sub-olf01 to sub-olf04,
the four of the info-theory paper. They are flagged first_four and kept as a consistency check.

Every rule is the paper's. The profile, D_i = -ln(S_i/S_0)/b with S/S_0 clipped to [0.01, 1] and D to
[0, 5e-3], and the tissue rule (no clipped direction, mean D 0.4 to 1.5e-3 mm^2/s) are copied from
renyi_order_test.dists. The tensor is a DIPY weighted least-squares fit as in b1500_tensor.py, with the
tissue rule of fa_vs_eigen_entropy.py. descriptors() is copied from fa_vs_eigen_entropy.py. The paper's
scripts are not changed. The brain mask is DIPY median_otsu on the b = 0 volume, so the script needs
nothing beyond the public dataset. The mask includes the tumor and edema. If the registered lesion masks
of the OGM pipeline are present (OGM_DERIV/<sub>/reg/{tumor_dil,edema}_dwi.nii.gz), the script repeats
both numbers without them.

The first four patients give r = 0.989 to 0.991 in paper-jensen-gap-info-theory
(analysis/ogm_external_check.py). That script's tissue window runs to 2.0e-3 mm^2/s and keeps voxels with
clipped directions, so it includes CSF partial volume. Both values are correct for their windows. When
the OGM pipeline's BET brain mask is present (OGM_DERIV/<sub>/metrics/<sub>_brainmask.nii.gz), the script
also computes r with that script's rule (BET mask, mean D 0.4 to 2.0e-3, clipped directions kept) as
r_H1_H2_D_infotheory_rule, to show that the window accounts for the difference.

Part 2 (below) repeats the edema-against-NAWM redundancy test of that script's step 2 across every
baseline patient with an edema mask, with the identity paper's rule. It reads no clinical variable.

Paths: OGM_BIDS (default M:/ds007345) and OGM_DERIV (default M:/ds007345-derivatives, needed for the
lesion masks and for part 2). Writes analysis/ogm_replication.csv and, for part 2,
analysis/ogm_replication_edema.csv and ogm_replication_edema_corr.csv. Runs in about 4 minutes.
Usage: python analysis/ogm_replication.py
"""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats

HERE = Path(__file__).resolve().parent
BIDS = Path(os.environ.get("OGM_BIDS", "M:/ds007345"))
DERIV = Path(os.environ.get("OGM_DERIV", "M:/ds007345-derivatives"))
FIRST_FOUR = ["sub-olf01", "sub-olf02", "sub-olf03", "sub-olf04"]   # the info-theory paper's four, kept as a check
SUBJECTS = sorted(d.name for d in (BIDS / "derivatives").iterdir() if d.name.startswith("sub-olf") and not d.name.endswith("fu")
                  and (d / "dwi_processing" / f"{d.name}_dwi_topup_eddy.nii.gz").is_file())
EXCLUDED = []   # unreadable files, printed at the end
LN3 = np.log(3.0)


def renyi_norm(p, alpha):
    # copied from renyi_order_test.py
    n = p.shape[-1]
    if abs(alpha - 1.0) < 1e-9:
        H = -np.sum(p * np.log(np.maximum(p, 1e-20)), axis=-1)
    else:
        H = (1.0 / (1.0 - alpha)) * np.log(np.sum(np.maximum(p, 1e-20) ** alpha, axis=-1))
    return H / np.log(n)


def dists(dwi_s, bvals_s, brain):
    # copied from renyi_order_test.py
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
    return pD, pS, valid


def descriptors(lam):
    # copied from fa_vs_eigen_entropy.py (FA and the normalized order-1 eigenvalue entropy)
    lam = np.clip(lam, 0, None); s = lam.sum(-1, keepdims=True); p = lam / s
    m = lam.mean(-1, keepdims=True); dev = lam - m
    fa = np.sqrt(1.5 * (dev ** 2).sum(-1) / (lam ** 2).sum(-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        h1 = -np.where(p > 0, p * np.log(p), 0.0).sum(-1) / LN3
    return fa, h1


def lesion_mask(sub, shape):
    m, found = np.zeros(shape, bool), False
    for f in ("tumor_dil_dwi.nii.gz", "edema_dwi.nii.gz"):
        p = DERIV / sub / "reg" / f
        if p.is_file():
            m |= np.asarray(nib.load(str(p)).dataobj) > 0.5; found = True
    return m if found else None


rows = []
for sub in SUBJECTS:
    from dipy.core.gradients import gradient_table
    from dipy.reconst.dti import TensorModel
    from dipy.segment.mask import median_otsu
    dp = BIDS / "derivatives" / sub / "dwi_processing"
    try:
        dwi = np.asarray(nib.load(str(dp / f"{sub}_dwi_topup_eddy.nii.gz")).dataobj, dtype=np.float32)
    except (EOFError, OSError) as e:
        print(f"{sub}: SKIPPED, unreadable DWI ({type(e).__name__})", flush=True); EXCLUDED.append(sub); continue
    bval = np.loadtxt(dp / f"{sub}_dwi_topup_eddy.bval").astype(float)
    bvec = np.loadtxt(dp / f"{sub}_dwi_topup_eddy.bvec")
    meta = json.loads((BIDS / sub / "dwi" / f"{sub}_run-01_dwi.json").read_text())
    _, brain = median_otsu(dwi[..., bval < 50].mean(-1), median_radius=4, numpass=4)
    shell = np.abs(bval - 1500) < 150
    assert ((bval < 50) | shell).all(), "expected a single b = 1500 shell"
    row = dict(subject=sub, first_four=sub in FIRST_FOUR, manufacturer=meta.get("Manufacturer"), model=meta.get("ManufacturersModelName"),
               field_T=meta.get("MagneticFieldStrength"), slice_mm=meta.get("SliceThickness"),
               n_dir=int(shell.sum()), n_b0=int((bval < 50).sum()), b=float(np.round(bval[shell].mean(), -2)))

    # tensor, as b1500_tensor.py (DIPY TensorModel default, weighted least squares)
    if bvec.shape[0] != 3:
        bvec = bvec.T
    nrm = np.linalg.norm(bvec, axis=0); bvec = np.where(nrm > 1e-6, bvec / np.where(nrm > 1e-6, nrm, 1.0), 0.0)
    lam = np.nan_to_num(TensorModel(gradient_table(bval, bvecs=bvec.T, atol=0.1)).fit(dwi, mask=brain).evals).astype(np.float64)
    md = np.clip(lam, 0, None).mean(-1)
    lesion = lesion_mask(sub, brain.shape)
    variants = {"all": brain} | ({"no_lesion": brain & ~lesion} if lesion is not None else {})
    for name, msk in variants.items():
        sfx = "" if name == "all" else "_no_lesion"
        # directional profile, as renyi_order_test.py
        pD, pS, valid = dists(dwi, bval, msk)
        for tag, p in (("D", pD), ("S", pS)):
            h1, h2 = renyi_norm(p, 1.0), renyi_norm(p, 2.0); g = np.isfinite(h1) & np.isfinite(h2)
            row[f"r_H1_H2_{tag}{sfx}"] = float(np.corrcoef(h1[g], h2[g])[0, 1])
        row[f"n_profile_voxels{sfx}"] = int(valid.sum())
        # tensor, with the tissue rule of fa_vs_eigen_entropy.py
        ok = msk & np.all(np.isfinite(lam), -1) & (np.clip(lam, 0, None).sum(-1) > 0) & (md > 0.4e-3) & (md < 1.5e-3)
        fa, h1 = descriptors(lam[ok]); keep = fa < 0.999
        row[f"rho_FA_H1{sfx}"] = float(stats.spearmanr(fa[keep], h1[keep])[0])
        row[f"median_FA{sfx}"] = float(np.median(fa[keep])); row[f"n_tensor_voxels{sfx}"] = int(keep.sum())
    bet = DERIV / sub / "metrics" / f"{sub}_brainmask.nii.gz"
    if bet.is_file():
        # the rule of ogm_external_check.py: BET mask, mean D 0.4 to 2.0e-3, clipped directions kept
        b0 = dwi[..., bval < 50].mean(-1); Sp = np.clip(dwi[..., shell] / (b0[..., None] + 1e-10), 1e-2, 1.0)
        D = np.clip(-np.log(Sp) / float(np.mean(bval[shell])), 0.0, 5e-3); del Sp
        v = (np.asarray(nib.load(str(bet)).dataobj) > 0) & (D.mean(-1) > 0.4e-3) & (D.mean(-1) < 2.0e-3)
        p = D[v] / (D[v].sum(-1, keepdims=True) + 1e-20); del D
        row["r_H1_H2_D_infotheory_rule"] = float(np.corrcoef(renyi_norm(p, 1.0), renyi_norm(p, 2.0))[0, 1])
    rows.append(row)
    print(f"{sub}: {row['manufacturer']} {row['model']}, {row['n_dir']} directions, {row['n_b0']} b0; "
          f"r(H1, H2) {row['r_H1_H2_D']:.4f}, rho(FA, H1 eig) {row['rho_FA_H1']:.5f}, median FA {row['median_FA']:.3f}"
          + (f"; without lesions {row['r_H1_H2_D_no_lesion']:.4f}, {row['rho_FA_H1_no_lesion']:.5f}" if "r_H1_H2_D_no_lesion" in row else ""),
          flush=True)
    del dwi, lam

pd.DataFrame(rows).to_csv(HERE / "ogm_replication.csv", index=False)
print(f"written -> analysis/ogm_replication.csv ({len(rows)} patients; skipped as unreadable: {EXCLUDED or 'none'})")

# ---------------------------------------------------------------- 2. redundancy in pathology
# Every baseline patient with a registered peritumoral edema mask: region medians of MD and the
# directional descriptors in the edema and in normal-appearing white matter (NAWM), and the deficit
# (edema minus NAWM). Do the descriptors rank the patients alike? Spearman across patients. No
# clinical variable is read. Needs the OGM pipeline's derived masks in OGM_DERIV/<sub>:
# metrics/<sub>_brainmask.nii.gz (BET, used here because median_otsu crops frontal edema) and
# reg/{edema,tumor_dil}_dwi.nii.gz (the dataset's baseline segmentations registered to the DWI).
# NAWM = brain, tensor FA > 0.3, outside the edema and the dilated tumor, as in
# paper-jensen-gap-info-theory/analysis/ogm_external_check.py, whose step 2 this repeats with the
# identity paper's tissue rule. Edema has a high mean D (median 1.46e-3 mm^2/s in that script), so
# the 1.5e-3 ceiling keeps only its milder part. "csf2" keeps the no-clipped-direction guard with the
# old ceiling of 2.0e-3 as a sensitivity rule.
# Writes analysis/ogm_replication_edema.csv (per patient and rule) and
# analysis/ogm_replication_edema_corr.csv (Spearman correlations of the deficits across patients).
EDEMA_MIN_VOX, FA_WM = 50, 0.3
RULES = {"identity": 1.5e-3, "csf2": 2.0e-3}
DESC = ["MD", "CV2", "PRN", "H1", "H2"]
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel
patients = sorted(d.name for d in DERIV.iterdir() if d.is_dir() and d.name.startswith("sub-") and not d.name.endswith("fu")
                  and (d / "reg" / "edema_dwi.nii.gz").is_file()) if DERIV.is_dir() else []
erows = []
for sub in patients:
    dp = BIDS / "derivatives" / sub / "dwi_processing"
    dimg = nib.load(str(dp / f"{sub}_dwi_topup_eddy.nii.gz"))
    dwi = np.asarray(dimg.dataobj, dtype=np.float32)
    bval = np.loadtxt(dp / f"{sub}_dwi_topup_eddy.bval").astype(float); bvec = np.loadtxt(dp / f"{sub}_dwi_topup_eddy.bvec")

    def load(p):
        # masks are used voxel for voxel, never resampled, so each must be on the DWI grid. Some headers in
        # the dataset's own derivatives (FA, MD) are offset from the DWI header; those maps are not used here.
        m = nib.load(str(p))
        assert m.shape[:3] == dimg.shape[:3] and np.allclose(m.affine, dimg.affine, atol=1e-3), f"{p} is not on the DWI grid"
        return np.asarray(m.dataobj) > 0.5
    brain = load(DERIV / sub / "metrics" / f"{sub}_brainmask.nii.gz"); edema = load(DERIV / sub / "reg" / "edema_dwi.nii.gz")
    tp = DERIV / sub / "reg" / "tumor_dil_dwi.nii.gz"; tumor = load(tp) if tp.is_file() else np.zeros_like(brain)
    if bvec.shape[0] != 3:
        bvec = bvec.T
    nrm = np.linalg.norm(bvec, axis=0); bvec = np.where(nrm > 1e-6, bvec / np.where(nrm > 1e-6, nrm, 1.0), 0.0)
    fa = descriptors(np.nan_to_num(TensorModel(gradient_table(bval, bvecs=bvec.T, atol=0.1)).fit(dwi, mask=brain).evals))[0]
    # the profile, as renyi_order_test.dists, kept for every voxel
    sh = bval >= 50; b0 = dwi[..., bval < 50].mean(-1)
    Sraw = dwi[..., sh] / (b0[..., None] + 1e-10); del dwi
    unclipped = np.all((Sraw > 1e-2) & (Sraw < 1.0), axis=-1)
    D = np.clip(-(1.0 / float(np.mean(bval[sh]))) * np.log(np.clip(Sraw, 1e-2, 1.0)), 0.0, 5e-3); del Sraw
    Dbar = D.mean(-1)
    for rule, ceil in RULES.items():
        tissue = brain & unclipped & (Dbar > 0.4e-3) & (Dbar < ceil)
        ed = edema & tissue; nawm = tissue & (fa > FA_WM) & ~tumor & ~edema
        row = dict(subject=sub, rule=rule, n_edema=int(ed.sum()), n_edema_mask=int((edema & brain).sum()),
                   frac_edema_kept=float(ed.sum() / max((edema & brain).sum(), 1)), n_nawm=int(nawm.sum()))
        if ed.sum() >= EDEMA_MIN_VOX:
            for region, m in (("ed", ed), ("nawm", nawm)):
                Dm = D[m]; mu = Dm.mean(-1); cv2 = Dm.var(-1) / mu ** 2; p = Dm / (Dm.sum(-1, keepdims=True) + 1e-20)
                vals = dict(MD=mu, CV2=cv2, PRN=1 / (1 + cv2), H1=renyi_norm(p, 1.0), H2=renyi_norm(p, 2.0))
                for k_, v_ in vals.items():
                    row[f"{k_}_{region}"] = float(np.median(v_))
            for k_ in DESC:
                row[f"d_{k_}"] = row[f"{k_}_ed"] - row[f"{k_}_nawm"]
        erows.append(row)
    print(f"{sub}: edema kept {erows[-2]['frac_edema_kept']:.2f} (identity rule), {erows[-1]['frac_edema_kept']:.2f} (csf2)", flush=True)
    del D, Dbar
if erows:
    ep = pd.DataFrame(erows); ep.to_csv(HERE / "ogm_replication_edema.csv", index=False)
    crow = []
    for rule in RULES:
        e = ep[(ep.rule == rule) & ep.d_MD.notna()]
        c = e[[f"d_{k_}" for k_ in DESC]].corr(method="spearman"); c.index = c.columns = DESC
        crow += [dict(rule=rule, a=a, b=b, n_patients=len(e), rho=float(c.loc[a, b])) for i, a in enumerate(DESC) for b in DESC[i + 1:]]
        ang = DESC[1:]; within = [abs(c.loc[a, b]) for i, a in enumerate(ang) for b in ang[i + 1:]]
        print(f"rule {rule}: {len(e)} patients with >= {EDEMA_MIN_VOX} edema tissue voxels, median edema kept "
              f"{e.frac_edema_kept.median():.2f}; |rho| among CV2, PR/N, H1, H2 {min(within):.3f} to {max(within):.3f}; "
              f"|rho| with MD " + ", ".join(f"{k_} {abs(c.loc[k_, 'MD']):.2f}" for k_ in ang))
    pd.DataFrame(crow).to_csv(HERE / "ogm_replication_edema_corr.csv", index=False)
    print("written -> analysis/ogm_replication_edema.csv, ogm_replication_edema_corr.csv")
