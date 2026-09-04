"""Shared fixtures for the Coso test suite."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
WELLBORES_XLSX = REPO_ROOT / "data" / "wellbores.xlsx"


@pytest.fixture
def repo_root_as_script_dir():
    """Make the repo root look like the running script's directory.

    geometry.py's `set_well_network()` (and the straight-well helper next to it)
    build their data path as `f"{sys.path[0]}/data/wellbores.xlsx"`. Under pytest
    `sys.path[0]` is the tests/ directory, so the read fails with a
    FileNotFoundError on `tests/data/wellbores.xlsx`. This is the same "wellbores
    path gotcha" that `test_wells_intersections.py` works around inline; anything
    that builds a model with `use_wells=True` needs it.

    """
    original_sys_path = list(sys.path)
    original_cwd = os.getcwd()
    sys.path.insert(0, str(REPO_ROOT))
    os.chdir(REPO_ROOT)
    try:
        yield REPO_ROOT
    finally:
        os.chdir(original_cwd)
        sys.path[:] = original_sys_path
