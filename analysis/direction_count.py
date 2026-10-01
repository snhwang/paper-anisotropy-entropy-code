"""Directional entropies depend on the number of directions. The coefficient of variation and the
deficits ln N - H_alpha do not.

The strong single fiber of Table 1 (eigenvalues 1.7, 0.3, 0.3 x 1e-3 mm^2/s along x) is sampled on
near-uniform Fibonacci sets of N directions, the construction used for Table 1. CV_D and the deficits
estimate properties of the profile on the sphere and stay fixed as N changes. The normalized entropies
H_alpha / ln N move with N, because ln N does.

Writes analysis/direction_count.csv, read by identity_checks.py.
"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NS = [30, 64, 93, 256]
ALPHAS = [0.5, 1.0, 2.0, 4.0]
TENSOR = np.diag([1.7e-3, 0.3e-3, 0.3e-3])


def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n); theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)


def renyi(p, a):
    if a == 1.0:
        return -(p * np.log(p)).sum()
    return np.log((p ** a).sum()) / (1 - a)


rows = []
for n in NS:
    G = fibonacci_sphere(n)
    D = np.einsum("ij,jk,ik->i", G, TENSOR, G)  # single tensor: the profile is g^T D g
    p = D / D.sum()
    row = {"N": n, "CV_D": float(np.sqrt(D.var() / D.mean() ** 2))}
    for a in ALPHAS:
        h = renyi(p, a)
        row[f"deficit_{a:g}"] = float(np.log(n) - h)
        row[f"Hn_{a:g}"] = float(h / np.log(n))
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(HERE / "direction_count.csv", index=False, float_format="%.6f")
print(df.to_string(index=False))
print("written -> analysis/direction_count.csv")
