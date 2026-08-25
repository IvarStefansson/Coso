"""Standalone geometry-creation-and-VTU-export tool for example 3.

Builds the *real* meshed geometry -- domain, faults (as configured by
run_example_3.py's current fault selection/extension config), and wells --
via ``ex3.MainModel.set_materials()`` + ``set_geometry()`` (same recipe as
diagnose_well_fracture_geometry.py, which this reuses), then exports the
resulting mixed-dimensional grid to VTU for visual inspection in ParaView.

This differs from preprocess_fault_planes.py, which only previews the raw,
*unmeshed* fault-plane rectangles (no domain, no wells, no meshing) -- this
script exports the actual meshed subdomains (3D domain, 2D fault planes, 1D
wells and fracture-intersection lines, 0D intersection points), exactly as
they will appear in a real run, without running any physics/solving.

Usage (run as a script, from anywhere -- see the sys.path/cwd bootstrap
below and the "wellbores path gotcha" this guards against):

    python diagnostics/export_geometry_vtu.py [--no-wells] [--extensions PATH]
        [--out-folder DIR] [--file-name NAME]

Output: ``{out_folder}/geometry/{file_name}_*.vtu`` (+ a .pvd index), the
same layout ``GeometryExporting.initialize_data_saving()`` writes for a real
run -- open the .pvd in ParaView.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import porepy as pp

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)  # run_example_3.py / geometry.py resolve data paths via
# sys.path[0] and relative dirs (point_cloud_clusters/, fault_extensions.json),
# both of which assume the repo root as cwd.

import run_example_3 as ex3  # noqa: E402
from diagnose_well_fracture_geometry import build_model_params  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extensions",
        default="fault_extensions.json",
        help="Path to the fault-extensions config to use (default: fault_extensions.json).",
    )
    parser.add_argument(
        "--no-wells",
        action="store_true",
        help="Build the geometry without wells (faults/domain only).",
    )
    parser.add_argument(
        "--out-folder",
        default="diagnostics/geometry_preview_saved_data",
        help="Output data folder (VTU is written under '<out-folder>/geometry/').",
    )
    parser.add_argument(
        "--file-name",
        default="geometry_preview",
        help="Base file name for the exported VTU/PVD files.",
    )
    args = parser.parse_args()

    model_params = build_model_params(args.extensions, use_wells=not args.no_wells)
    model_params["folder_name"] = args.out_folder
    model_params["file_name"] = args.file_name

    print(f"Fault extensions config: {args.extensions}")
    print(f"Wells: {'excluded' if args.no_wells else 'included'}")
    print(f"USE_CONSTRAINTS (from run_example_3.py): {ex3.USE_CONSTRAINTS}")

    model = ex3.MainModel(model_params)
    model.set_materials()
    model.set_geometry()

    fracture_names = model.fracture_names()
    print(f"\nFractures: {fracture_names}")
    for dim in range(model.mdg.dim_max() + 1):
        subdomains = model.mdg.subdomains(dim=dim)
        print(f"  {len(subdomains)} subdomain(s) of dim {dim}")

    # Tag each fault-plane subdomain with its fault_planes.py fault number, so it
    # can be colored/thresholded/labeled by fault in ParaView. sd.frac_num indexes
    # fracture_names() (fault_planes.py's CSV stem, e.g. "0000"); leading zeros are
    # trimmed via int() since ParaView handles a numeric field better than a string
    # one for coloring/thresholding.
    fault_number_data = []
    for sd in model.mdg.subdomains(dim=2):
        fault_name = fracture_names[sd.frac_num]
        try:
            fault_number = int(fault_name)
        except ValueError:
            fault_number = sd.frac_num
        fault_number_data.append((sd, "fault_number", np.full(sd.num_cells, fault_number)))

    exporter = pp.Exporter(
        model.mdg,
        file_name=args.file_name,
        folder_name=str(Path(args.out_folder) / "geometry"),
        export_constants_separately=False,
    )
    exporter.write_vtu(data=fault_number_data, time_dependent=False)

    out_dir = REPO_ROOT / args.out_folder / "geometry"
    print(f"\nSUCCESS: geometry exported to {out_dir}")
    print("Open the .pvd file there in ParaView to inspect.")


if __name__ == "__main__":
    main()
