"""Create the working-folder layout expected by the notebooks.

The notebooks were written against the original project layout
(03_Python_Optimization, 05_Results, 06_Figures, 07_Tables). This script
recreates that layout in a separate folder by copying the repository files,
so the notebooks can be run without editing their paths.

Usage (from the repository root):
    python notebooks/setup_workspace.py workspace
    export MST_PROJECT_ROOT=$PWD/workspace        # Windows: set MST_PROJECT_ROOT=%CD%\\workspace
    jupyter lab notebooks/
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*"):
        if f.is_file():
            shutil.copy2(f, dst / f.name)


def main(target: str = "workspace") -> None:
    root = Path(target).resolve()
    (root / "03_Python_Optimization").mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / "mst" / "mst_core.py", root / "03_Python_Optimization" / "mst_core.py")
    for sub in ("raw", "audits", "sarl"):
        copy_tree(REPO / "data" / sub, root / "05_Results")
    copy_tree(REPO / "tables", root / "07_Tables")
    copy_tree(REPO / "figures", root / "06_Figures")
    copy_tree(REPO / "docs", root / "01_Administrative")
    (root / "04_SARL_Verification").mkdir(exist_ok=True)
    print(f"Workspace created at {root}")
    print(f"Set MST_PROJECT_ROOT={root} before starting Jupyter.")


if __name__ == "__main__":
    main(*sys.argv[1:2])
