"""Opt-in end-to-end check that the monolithic solver actually solves.

Currently xfail: the harness it drives does not converge on this branch, for ANY
solver scheme. Measured on the single-fracture diagnostic geometry, all three
configurations fail identically with "Max retries (10) exhausted" at time step #0,
dt halved ten times from 5.0e9 with no converged Newton solve:

    monolithic + iterative        failed
    monolithic + direct           failed
    sequential + iterative        failed   <- the pre-existing default

The control failing is the point: this says nothing about the monolithic scheme, so
the test cannot currently distinguish a working port from a broken one. It is kept
because it is the only check that would catch a `pp_solvers.thm_factory` group/DoF
mismatch against this model's `RadialReturnTangentialContactMechanicsEquation` and
`DarcysLawAdEverywhere` -- something the wiring assertions in test_solver_setup.py
cannot see, since that failure only surfaces once a real linear system is assembled.

Before trusting a green run here, first make `run_diagnostic` converge with
`scheme="sequential"`; until then an XPASS is more likely to mean the harness was
fixed than that the solver was. XPASS is reported separately by pytest, so the
marker is non-strict.

Excluded from the default run by the `slow` marker (see pyproject.toml); run with
`pytest -m slow`.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import WELLBORES_XLSX

pytest.importorskip("pp_solvers")

import diagnose_example_3  # noqa: E402


@pytest.mark.slow
@pytest.mark.xfail(
    reason=(
        "diagnose_example_3's geometry does not get past time step #0 on this branch "
        "under any scheme, including the sequential default -- see module docstring"
    ),
    strict=False,
)
@pytest.mark.skipif(
    not WELLBORES_XLSX.exists(),
    reason="real well data (data/wellbores.xlsx) not available",
)
def test_monolithic_runs_end_to_end(tmp_path, repo_root_as_script_dir):
    init_model, model = diagnose_example_3.run_diagnostic(
        iterative_linear_solver=True,
        scheme="monolithic",
        run_init=True,
        folder_name=str(tmp_path / "monolithic"),
    )

    # Both phases must have advanced past t=0, i.e. Newton actually converged.
    assert init_model.time_data.time_index_successful > 0
    assert model.time_data.time_index_successful > 0

    for name in ("pressure", "temperature"):
        values = model.equation_system.get_variable_values(
            variables=[name], time_step_index=0
        )
        assert np.all(np.isfinite(values)), f"non-finite {name} in the final state"
