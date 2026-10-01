"""Where the scripts read data and write figures. Set the environment variables to override.

INFO_DATA       folder holding HCP/manifest_n1379_b1500.tsv, the list of processed sessions
                (default: a folder named diffusion beside this repository)
DTI_OUTPUT_DIR  folder holding one processing folder per session, each with inputs/dwi.bval and
                processed/{dwi_raw.nii.gz, dwi_raw.bvec, mask.nii.gz}
PAPER_DIR       manuscript folder. Figures go to its figures/ folder when it exists, otherwise to
                figures/ in this repository.
The defaults work from Windows and from WSL.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("INFO_DATA", ROOT.parent / "diffusion"))
DTI_OUTPUT = Path(os.environ.get("DTI_OUTPUT_DIR", "D:/dti_output" if os.name == "nt" else "/mnt/d/dti_output"))
MANIFEST = DATA / "HCP" / "manifest_n1379_b1500.tsv"
PAPER = Path(os.environ.get("PAPER_DIR", ROOT.parent / "paper-anisotropy-entropy"))
FIGURES = PAPER / "figures" if PAPER.is_dir() else ROOT / "figures"
