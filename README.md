# Diffusion anisotropy and entropy: code

Analysis code for the paper *Diffusion anisotropy and entropy: reflections of each other* (Scott N. Hwang, Jonathan K. Maffie, Sangam G. Kanekar).

The paper shows that diffusion anisotropy indices and entropies are one quantity at matched order. For any positive values, the Rényi entropy of their proportions at order α is a strictly monotone function of the power-mean Jensen gap of the same order. This repository holds the scripts that produce every number, table and figure in the paper. It also holds a script that checks every stated relation and every quoted number.

## Contents

| Script | Paper item | Output | Needs imaging data |
|---|---|---|---|
| `analysis/identity_checks.py` | every relation and quoted number | console report, non-zero exit on failure | no |
| `analysis/synthetic_profiles.py` | Table 1 (synthetic voxels) | `synthetic_profiles.csv` | no |
| `analysis/order1_link.py` | Section 4.2, order-1 profile against eigenvalue deficit | `order1_link.csv` | no |
| `analysis/direction_count.py` | Discussion, dependence on the number of directions | `direction_count.csv` | no |
| `analysis/fa_vs_eigen_entropy.py` | Figure 1 | `fa_vs_eigen_entropy.{png,pdf}`, `fa_vs_eigen_entropy.csv` | yes |
| `analysis/build_reflections_maps.py` | Figure 2 | `reflections_maps.{png,pdf}`, `reflections_maps_check.csv` | yes |
| `analysis/renyi_order_test.py` | Section 5, order-1 against order-2 entropies in tissue | `renyi_order_test.csv` | yes |
| `analysis/b1500_tensor.py` | helper, b = 1500 tensor fit | cached eigenvalues in `analysis/_cache/` | yes |
| `analysis/paths.py` | helper, data and output locations | none | no |

The CSV outputs are committed, so `identity_checks.py` runs without the imaging data.

## Installation

The code was run with Python 3.12 (Linux) and 3.14 (Windows), with the package versions in `requirements.txt`.

```
pip install -r requirements.txt
```

## Running

The checks need only the committed CSVs.

```
python analysis/identity_checks.py
```

The scripts without imaging data rebuild their CSVs in a few seconds each.

```
python analysis/synthetic_profiles.py
python analysis/order1_link.py
python analysis/direction_count.py
```

`synthetic_profiles.py` also draws a diagnostic figure into `figures/` in this repository. It is not a figure of the paper.

## Imaging data

The in-tissue results use the Lifespan Human Connectome Project in Aging (HCP-A) data in the consortium's minimally preprocessed form. The data are distributed through the BALSA repository (https://balsa.wustl.edu) under the AABC Data Use Terms. They are not included here.

The data scripts expect two inputs, located through environment variables (see `analysis/paths.py`).

- `INFO_DATA` is a folder holding `HCP/manifest_n1379_b1500.tsv`, a tab-separated list of processed sessions with a `session_id` column. The paper uses its first four rows. Participant identifiers are withheld under the AABC Data Use Terms, so the manifest is not included.
- `DTI_OUTPUT_DIR` is a folder with one subfolder per session. Each holds `inputs/dwi.bval` and `processed/dwi_raw.nii.gz`, `processed/dwi_raw.bvec` and `processed/mask.nii.gz`.

Tensor eigenvalues come from a DIPY weighted least-squares fit to the b ≈ 0 and b = 1500 s/mm² volumes (`b1500_tensor.py`). The fits are cached in `analysis/_cache/`, which is not tracked.

Figures are written to the `figures/` folder of the manuscript when `PAPER_DIR` points to it, or when the manuscript folder sits beside this repository as `paper-anisotropy-entropy`. Otherwise they go to `figures/` here.
