"""Acquisition facts quoted in Section 5 for the four HCP-Aging participants of the paper.

For each session (first four rows of the b = 1500 manifest) this records the number of b = 1500
directions, the number of b ~ 0 volumes and where they sit in the series (interleaved or not), and
how close the b = 1500 directions are to a uniform sampling of the sphere, measured by the largest
deviation of their second-moment matrix <g g^T> from I/3. Equal-weight averages over directions stand
in for averages over the sphere only when this deviation is small.

Reads DTI_OUTPUT_DIR/<session>/{inputs/dwi.bval, processed/dwi_raw.bvec}. Participants are labelled
S1 to S4 by manifest position. No identifier is written.
Writes analysis/acquisition_counts.csv, read by identity_checks.py.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from paths import MANIFEST, DTI_OUTPUT

HERE = Path(__file__).resolve().parent
B, TOL = 1500.0, 150.0

rows = []
for k, sid in enumerate(pd.read_csv(MANIFEST, sep="\t").head(4).session_id, 1):
    bval = np.loadtxt(DTI_OUTPUT / sid / "inputs" / "dwi.bval")
    bvec = np.loadtxt(DTI_OUTPUT / sid / "processed" / "dwi_raw.bvec")
    bvec = bvec if bvec.shape[0] == 3 else bvec.T
    shell = np.abs(bval - B) < TOL
    zero = np.abs(bval) < TOL
    g = bvec[:, shell].T
    g = g / np.linalg.norm(g, axis=1, keepdims=True)
    m2 = g.T @ g / len(g)
    z = np.flatnonzero(zero)
    rows.append(dict(subject=f"S{k}", n_volumes=int(bval.size), n_b1500=int(shell.sum()), n_b0=int(zero.sum()),
                     first_b0=int(z[0]), last_b0=int(z[-1]), largest_b0_gap=int(np.diff(z).max()),
                     second_moment_max_dev=float(np.abs(m2 - np.eye(3) / 3).max())))
    print(rows[-1])

pd.DataFrame(rows).to_csv(HERE / "acquisition_counts.csv", index=False)
print("written -> analysis/acquisition_counts.csv")
