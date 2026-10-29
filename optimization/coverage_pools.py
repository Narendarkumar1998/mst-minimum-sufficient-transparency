"""
Load the coverage-utility instance generator from notebook 02 (Steps 124 and 126)
without running the notebook's experiments: only imports, constants, classes and
function definitions of the relevant cells are executed.

The generator is used unchanged so that the coverage pools are identical to those
of the coverage benchmark (coverage_exact_vs_greedy_main_benchmark_v1_pool_normalized.csv).
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "02_benchmark_generation_and_statistical_analysis.ipynb"
CELLS = (1, 2, 50, 82, 84)   # design constants, product generators, coverage generator, pool-normalized utility


def load():
    sys.path.insert(0, str(ROOT / "mst"))
    import mst_core  # noqa: F401
    nb = json.loads(NOTEBOOK.read_text())
    code = [''.join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    import types
    module = types.ModuleType("coverage_notebook")
    sys.modules["coverage_notebook"] = module
    ns = module.__dict__
    ns["mst_core"] = mst_core
    for i in CELLS:
        tree = ast.parse(code[i])
        keep = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
                keep.append(node)
            elif isinstance(node, ast.Assign) and all(
                    isinstance(t, ast.Name) and t.id.isupper() for t in node.targets):
                keep.append(node)
            elif isinstance(node, ast.If):   # guarded legacy-generator assignment in cell 50
                keep.append(node)
        mod = ast.Module(body=keep, type_ignores=[])
        try:
            exec(compile(mod, f"<notebook cell {i}>", "exec"), ns)
        except NameError:
            # guarded blocks that refer to the notebook's interactive state are skipped
            mod = ast.Module(body=[n for n in keep if not isinstance(n, ast.If)], type_ignores=[])
            exec(compile(mod, f"<notebook cell {i}>", "exec"), ns)
    return ns


def coverage_pool(ns, n_items, regime, matched_seed):
    """Return (A, w, p, c) for one coverage instance of the benchmark."""
    inst = ns["generate_matched_coverage_instances"](
        n_items=n_items, n_features=ns["COVERAGE_POOL_NORMALIZED_FEATURES"] if "COVERAGE_POOL_NORMALIZED_FEATURES" in ns
        else ns["COVERAGE_FEATURES"], seed=matched_seed)[regime]
    items = inst["items"]
    A = [list(it.feature_coverage) for it in items]
    w = list(map(float, inst["feature_weights"]))
    p = [it.privacy for it in items]
    c = [it.cognitive_cost for it in items]
    ids = tuple(it.item_id for it in items)
    return ids, A, w, p, c
