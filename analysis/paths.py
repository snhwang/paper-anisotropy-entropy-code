"""Where the scripts read data and write figures.

Each location is taken from the first of these that sets it.
1. An environment variable.
2. local_paths.json in the repository root (not tracked), for a machine's own locations. A value is
   either a path or an object with one path per platform, keyed "nt" (Windows) and "posix".
3. The default below.

INFO_DATA       folder holding HCP/manifest_n1379_b1500.tsv, the list of processed sessions
                (default: data/ in this repository)
DTI_OUTPUT_DIR  folder holding one processing folder per session, each with inputs/dwi.bval and
                processed/{dwi_raw.nii.gz, dwi_raw.bvec, mask.nii.gz} (default: data/dti_output/)
PAPER_DIR       manuscript folder. Figures go to its figures/ folder when it exists, otherwise to
                figures/ in this repository (default: paper-anisotropy-entropy beside this repository)
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_LOCAL = ROOT / "local_paths.json"
_local = json.loads(_LOCAL.read_text(encoding="utf-8")) if _LOCAL.is_file() else {}


def _get(name, default):
    if name in os.environ:
        return Path(os.environ[name])
    value = _local.get(name)
    if isinstance(value, dict):
        value = value.get(os.name)
    return Path(value) if value else default


DATA = _get("INFO_DATA", ROOT / "data")
DTI_OUTPUT = _get("DTI_OUTPUT_DIR", ROOT / "data" / "dti_output")
MANIFEST = DATA / "HCP" / "manifest_n1379_b1500.tsv"
PAPER = _get("PAPER_DIR", ROOT.parent / "paper-anisotropy-entropy")
FIGURES = PAPER / "figures" if PAPER.is_dir() else ROOT / "figures"
