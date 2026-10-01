"""Order-1 link between the directional profile and the tensor eigenvalues.

Ozarslan and Mareci (ISMRM 2003) approximated the Shannon entropy of the normalized profile,
sigma = -(3/2pi) int D_N ln D_N with D_N = D/trace(D), by the von Neumann entropy of D/trace(D).
With x = D/Dbar on the sphere, ln 3 - sigma = <x ln x>, the order-1 deficit of the profile, and
ln 3 - sigma_vN = ln 3 - H_1(normalized eigenvalues), the order-1 deficit of the eigenvalues
(the quantitative anisotropy). Each direction's diffusivity is a weighted mean of the eigenvalues,
D(u) = sum_i lambda_i (u.e_i)^2, so Jensen's inequality for x ln x bounds the profile deficit by
the eigenvalue deficit. This script maps their ratio over the whole simplex of normalized
eigenvalues (the ratio does not depend on orientation or scale).

Writes analysis/order1_link.csv, read by identity_checks.py.
"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n); theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)


G2 = fibonacci_sphere(60_000) ** 2  # squared direction cosines, the Jensen weights


def xlogx(x):
    return np.where(x > 0, x * np.log(np.where(x > 0, x, 1.0)), 0.0)


def deficits(R):
    """R: (T, 3) normalized eigenvalue triples. Returns order-1 deficits (profile, eigenvalues)."""
    D = G2 @ R.T                        # (M, T) profile of each tensor, mean over sphere = 1/3
    x = D / D.mean(0, keepdims=True)
    prof = xlogx(x).mean(0)
    eig = np.log(3) + xlogx(R).sum(1)
    return prof, eig


def fa(R):
    m = R.mean(1, keepdims=True)
    return np.sqrt(1.5 * ((R - m) ** 2).sum(1) / (R ** 2).sum(1))


def mode(R):
    dev = R - R.mean(1, keepdims=True); n = np.sqrt((dev ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return 3 * np.sqrt(6) * np.prod(dev, 1) / n ** 3


# the simplex of sorted normalized eigenvalues, rho1 >= rho2 >= rho3 >= 0
step = 0.0025
tri = []
for r3 in np.arange(0, 1 / 3 + 1e-12, step):
    for r2 in np.arange(r3, (1 - r3) / 2 + 1e-12, step):
        tri.append((1 - r2 - r3, r2, r3))
R = np.array(tri)
prof = np.concatenate([deficits(R[k:k + 400])[0] for k in range(0, len(R), 400)])
eig = np.log(3) + xlogx(R).sum(1)
F, Mo = fa(R), mode(R)
keep = F > 0.02  # away from the 0/0 at isotropy
ratio = prof[keep] / eig[keep]

named = {"strong fiber (1.7, 0.3, 0.3)": [1.7, 0.3, 0.3], "stick (1, 0, 0)": [1, 0, 0],
         "flat disc (1, 1, 0)": [1, 1, 0], "planar (1.2, 1.2, 0.3)": [1.2, 1.2, 0.3],
         "near isotropic (1.01, 1, 0.99)": [1.01, 1.0, 0.99]}
Rn = np.array([np.array(v, float) / sum(v) for v in named.values()])
pn, en = deficits(Rn)

i_min, i_max = np.argmin(ratio), np.argmax(ratio)
rows = [
    ("simplex triples", float(len(R))),
    ("Jensen violations, profile deficit > eigenvalue deficit", float((prof > eig + 1e-9).sum())),
    ("ratio min (FA > 0.02)", float(ratio.min())),
    ("ratio min at FA", float(F[keep][i_min])),
    ("ratio min at mode", float(Mo[keep][i_min])),
    ("ratio max (FA > 0.02)", float(ratio.max())),
    ("ratio max at FA", float(F[keep][i_max])),
    ("ratio max at mode", float(Mo[keep][i_max])),
    ("ratio median, FA < 0.05", float(np.median(ratio[F[keep] < 0.05]))),
]
for (name, _), p, e in zip(named.items(), pn, en):
    rows += [(f"{name}: profile deficit", float(p)), (f"{name}: eigenvalue deficit", float(e)),
             (f"{name}: ratio", float(p / e))]
df = pd.DataFrame(rows, columns=["quantity", "value"])
df.to_csv(HERE / "order1_link.csv", index=False, float_format="%.6f")
print(df.to_string(index=False))
print("written -> analysis/order1_link.csv")
