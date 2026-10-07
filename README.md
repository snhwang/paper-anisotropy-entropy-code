# Diffusion anisotropy and entropy: code

Analysis code for the paper *Diffusion anisotropy and entropy: reflections of each other* (Scott N. Hwang, Jonathan K. Maffie, Sangam G. Kanekar).

The paper shows that diffusion anisotropy indices and entropies are one quantity at matched order. For any positive values, the Rényi entropy of their proportions at order α is a strictly monotone function of the power-mean Jensen gap of the same order. This repository holds the scripts that produce every number, table and figure in the paper. It also holds a script that checks every stated relation and every quoted number.

## Contents

| Script | Paper item | Output | Needs imaging data |
|---|---|---|---|
| `analysis/identity_checks.py` | every relation and quoted number, the signal-domain factor (b D̄)² of Section 4.1, and the Discussion's worked numbers (scale of the order-2 deficit, region averages, effective numbers), which it computes directly | console report, non-zero exit on failure | no |
| `analysis/tensor_concept.py` | Figure 1, three tensors as distributions of their normalized eigenvalues, and the worked example of Section 3.1 | `tensor_concept.{png,pdf}`, `tensor_concept.csv` | no |
| `analysis/synthetic_profiles.py` | Table 1 and Section 4.3 (synthetic voxels), the two-fifths ratio on 93 directions in Section 4.2 | `synthetic_profiles.csv` | no |
| `analysis/order1_link.py` | Section 4.2, order-1 profile deficit against the eigenvalue deficit | `order1_link.csv` | no |
| `analysis/direction_count.py` | Discussion, dependence on the number of directions | `direction_count.csv` | no |
| `analysis/fa_vs_eigen_entropy.py` | Figure 2, and the tissue numbers of Sections 3.2 and 5 (FA against the eigenvalue entropy, ordering by tensor mode, median FA) | `fa_vs_eigen_entropy.{png,pdf}`, `fa_vs_eigen_entropy.csv` | yes |
| `analysis/build_reflections_maps.py` | Figure 3 and its voxelwise identities | `reflections_maps.{png,pdf}`, `reflections_maps_check.csv` | yes |
| `analysis/renyi_order_test.py` | Abstract and Section 5, order-1 against order-2 directional entropies in tissue, and split-half reliability across orders | `renyi_order_test.csv` | yes |
| `analysis/acquisition_counts.py` | Section 5, number of directions and of b ≈ 0 volumes, near-uniform sampling | `acquisition_counts.csv` | yes |
| `analysis/b1500_tensor.py` | helper, b = 1500 tensor fit used by the figure scripts | cached eigenvalues in `analysis/_cache/` | yes |
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
python analysis/tensor_concept.py
python analysis/synthetic_profiles.py
python analysis/order1_link.py
python analysis/direction_count.py
```

`synthetic_profiles.py` also draws a diagnostic figure into `figures/` in this repository. It is not a figure of the paper.

With the imaging data in place (see below), the remaining scripts rebuild the figures and the tissue CSVs. The first run of a figure script fits the tensors and caches them, which takes a few minutes per participant.

```
python analysis/acquisition_counts.py
python analysis/fa_vs_eigen_entropy.py
python analysis/build_reflections_maps.py
python analysis/renyi_order_test.py
python analysis/identity_checks.py
```

## Imaging data

The in-tissue results use the Lifespan Human Connectome Project in Aging (HCP-A) data in the consortium's minimally preprocessed form. The data are distributed through the BALSA repository (https://balsa.wustl.edu) under the AABC Data Use Terms. They are not included here.

The data scripts expect two inputs.

- A manifest, `HCP/manifest_n1379_b1500.tsv`, a tab-separated list of processed sessions with a `session_id` column. The paper uses its first four rows. Participant identifiers are withheld under the AABC Data Use Terms, so the manifest is not included.
- One folder per session. Each holds `inputs/dwi.bval` and `processed/dwi_raw.nii.gz`, `processed/dwi_raw.bvec` and `processed/mask.nii.gz`.

By default the manifest is read from `data/HCP/` and the session folders from `data/dti_output/` in this repository. Git ignores `data/`, so the data are never committed. To keep the data elsewhere, set the environment variables `INFO_DATA` (the folder above `HCP/`) and `DTI_OUTPUT_DIR`, or put them in a `local_paths.json` file in the repository root, which git also ignores.

```
{
  "INFO_DATA": "/path/to/folder/holding/HCP",
  "DTI_OUTPUT_DIR": {"nt": "D:/dti_output", "posix": "/mnt/d/dti_output"}
}
```

A value is either one path or one path per platform, keyed `nt` (Windows) and `posix`. See `analysis/paths.py`.

Tensor eigenvalues come from a DIPY weighted least-squares fit to the b ≈ 0 and b = 1500 s/mm² volumes (`b1500_tensor.py`). The fits are cached in `analysis/_cache/`, which is not tracked.

Figures are written to the `figures/` folder of the manuscript when `PAPER_DIR` points to it, or when the manuscript folder sits beside this repository as `paper-anisotropy-entropy`. Otherwise they go to `figures/` here.

## License

Licensed under the Open Core Ventures Source Available License (OCVSAL) v1.0. See [LICENSE](LICENSE). Reading, running and modifying the code for non-production use, including reproducing the results of the paper, is permitted. Production use requires a commercial agreement. For commercial licensing, contact the Penn State Office of Technology Transfer at ottinfo@psu.edu.
