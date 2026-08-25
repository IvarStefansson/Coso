"""Check whether shrinking fault 0001 (fault_extensions.json) removed a genuine
well-fracture intersection.

Well-fracture connections in the mesh are NOT direct (well 1d) <-> (fracture 2d)
interfaces; they go through a shared 0d intersection-point subdomain:
    well (1d, well_num>=0) <-codim1-> point (0d) <-codim2-> fracture (2d, frac_num)
So identifying "well W intersects fracture F" means finding a 0d point subdomain
that appears in both a (well, point) interface and a (fracture, point) interface.

Builds the real geometry (with wells) twice -- once with the current
fault_extensions.json (fault 0001 shrunk top/bottom by 100m), once with an otherwise
identical config that omits the fault 0001 entry -- and compares which
(well_num, frac_num, fault_name) triples end up connected.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import run_example_3 as ex3
from diagnose_well_fracture_geometry import build_model_params
from wells import well_fracture_intersections as _well_fracture_intersections


def well_fracture_intersections(
    fault_extension_config: str,
) -> set[tuple[int, int, str | None]]:
    model_params = build_model_params(fault_extension_config)
    model = ex3.MainModel(model_params)
    model.set_materials()
    model.set_geometry()

    return _well_fracture_intersections(model.mdg, model.fracture_names())


def main() -> None:
    no_0001_config = "/tmp/fault_extensions_no_0001.json"
    with open(REPO_ROOT / "fault_extensions.json") as f:
        full_config = json.load(f)
    full_config.pop("0001", None)
    with open(no_0001_config, "w") as f:
        json.dump(full_config, f)

    print("=== WITH fault 0001 shrink (current fault_extensions.json) ===")
    with_shrink = well_fracture_intersections(str(REPO_ROOT / "fault_extensions.json"))
    for r in sorted(with_shrink):
        print(f"  well_num={r[0]}  frac_num={r[1]}  fault={r[2]}")

    print("\n=== WITHOUT fault 0001 shrink (0001 entry removed) ===")
    without_shrink = well_fracture_intersections(no_0001_config)
    for r in sorted(without_shrink):
        print(f"  well_num={r[0]}  frac_num={r[1]}  fault={r[2]}")

    print()
    print("Identical in both cases:", with_shrink == without_shrink)
    missing = without_shrink - with_shrink
    extra = with_shrink - without_shrink
    if missing:
        print("*** LOST by shrinking (present without shrink, missing with shrink):")
        for r in sorted(missing):
            print(f"    well_num={r[0]}  frac_num={r[1]}  fault={r[2]}")
    if extra:
        print("*** GAINED by shrinking (present with shrink, missing without):")
        for r in sorted(extra):
            print(f"    well_num={r[0]}  frac_num={r[1]}  fault={r[2]}")
    if not missing and not extra:
        print("No well-fracture intersections were added or removed by the shrink.")


if __name__ == "__main__":
    main()
