"""Single-shell b = 1500 tensor eigenvalues for the HCP-Aging sessions used in the paper.

The default processed/tensor_eigenvalues.nii.gz of each session was fitted to both HCP-A
shells (b = 1500 and b = 3000) together, which lowers the diffusivities because diffusion is
non-Gaussian at b = 3000. The paper uses the b = 1500 shell throughout, so every tensor
quantity comes from this function instead: a DIPY weighted least-squares fit (TensorModel
default) to the b = 0 and b = 1500 volumes of processed/dwi_raw.nii.gz, inside the session
brain mask. Results are cached under analysis/_cache (ignored by git, local data only).
"""
import os
from pathlib import Path
import numpy as np
import nibabel as nib

HERE = Path(__file__).resolve().parent
CACHE = HERE / "_cache"
from paths import DTI_OUTPUT as OUT
B, TOL = 1500.0, 150.0


def b1500_eigenvalues(session_id):
    """Return (eigenvalues (X, Y, Z, 3) in mm^2/s, brain mask) for one session."""
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{session_id}_evals_b1500.npz"
    if f.exists():
        z = np.load(f)
        return z["evals"].astype(np.float64), z["mask"]
    from dipy.core.gradients import gradient_table
    from dipy.reconst.dti import TensorModel
    s = OUT / session_id
    bval = np.loadtxt(s / "inputs" / "dwi.bval")
    bvec = np.loadtxt(s / "processed" / "dwi_raw.bvec")
    if bvec.shape[0] != 3:
        bvec = bvec.T
    sel = (bval < 50) | (np.abs(bval - B) < TOL)
    n = np.linalg.norm(bvec, axis=0)
    bvec = np.where(n > 1e-6, bvec / np.where(n > 1e-6, n, 1.0), 0.0)
    gtab = gradient_table(bval[sel], bvecs=bvec[:, sel].T, atol=0.1)
    img = nib.load(str(s / "processed" / "dwi_raw.nii.gz"))
    data = np.asarray(img.dataobj, dtype=np.float32)[..., sel]
    mask = np.asarray(nib.load(str(s / "processed" / "mask.nii.gz")).dataobj) > 0
    fit = TensorModel(gtab).fit(data, mask=mask)
    evals = np.nan_to_num(fit.evals).astype(np.float32)
    np.savez_compressed(f, evals=evals, mask=mask)
    return evals.astype(np.float64), mask
