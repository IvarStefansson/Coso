"""Minimal, Coso-independent reproduction of a well/meshing-constraint IndexError.

Background: a model can mark some of its "fractures" as pure meshing constraints
(pp.create_mdg(..., constraints=[...])) rather than real physical fractures -- e.g.
a throughgoing horizontal plane used only to force mesh conformity at a given depth,
with no fracture-flow physics of its own. Such constraint planes correctly get no
2d subdomain of their own in the resulting mdg.

However, WellNetwork3d.mesh() -> compute_well_fracture_intersections() runs its
well-vs-fracture intersection detection against the *full* fracture list handed to
it, with no way to know that some of those "fractures" are meshing-only constraints
that were excluded from becoming real subdomains. If a well happens to cross such a
constraint plane -- essentially unavoidable for a plane spanning the whole domain
footprint at a fixed depth -- the resulting WellFractureIntersection references a
fracture_index that has no corresponding entry in mdg.subdomains(dim=2), and
_add_well_fracture_interfaces raises IndexError instead of e.g. skipping the
intersection or raising an informative error.

This script builds the smallest case that reproduces it (one constraint plane, one
well crossing it, no other fractures), then shows the workaround (excluding the
constraint from the fracture network passed to well meshing) succeeding on a fresh
mdg. Re-run after any porepy fix: the first block should then also succeed.
"""

import gmsh
import numpy as np
import porepy as pp
from porepy.fracs.well_network import WellNetwork3d

BOX = {"xmin": -5.0, "xmax": 5.0, "ymin": -5.0, "ymax": 5.0, "zmin": -5.0, "zmax": 5.0}
MESHING_ARGS = {
    "cell_size": 5.0,
    "refinement_proximity_multiplier": 1e-6,
    "refinement_size_multiplier": 1.0,
    "background_transition_multiplier": 1.01,
}


def build_domain_and_well():
    domain = pp.Domain(BOX)
    # A single vertical well crossing z=0 -- i.e. crossing the constraint plane below,
    # unavoidable for any well reaching from above to below a throughgoing plane.
    well = pp.Well(np.array([[0.0, 0.0], [0.0, 0.0], [-4.0, 4.0]]))
    return domain, well


def build_constraint_plane(domain: pp.Domain) -> pp.PlaneFracture:
    # A throughgoing horizontal "constraint" plane spanning the domain's full x/y
    # footprint at z=0 -- e.g. a stratigraphic boundary meant only to force mesh
    # conformity, not to act as a flow fracture.
    box = domain.bounding_box
    return pp.PlaneFracture(
        np.array(
            [
                [box["xmin"], box["xmax"], box["xmax"], box["xmin"]],
                [box["ymin"], box["ymin"], box["ymax"], box["ymax"]],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
    )


def build_mdg(fracture_network: pp.FractureNetwork3d, constraints: list[int]):
    return pp.create_mdg(
        "simplex",
        MESHING_ARGS,
        fracture_network=fracture_network,
        constraints=constraints,
    )


def reset_gmsh() -> None:
    try:
        gmsh.clear()
        gmsh.finalize()
    except Exception:
        pass


def demonstrate_bug() -> None:
    print("--- Reproducing the bug: meshing wells against the constraint-including ---")
    print("--- fracture network (this is what geometry.py's create_well_mesh does) ---")
    domain, well = build_domain_and_well()
    constraint_plane = build_constraint_plane(domain)
    fracture_network = pp.create_fracture_network([constraint_plane], domain=domain)
    mdg = build_mdg(fracture_network, constraints=[0])
    print(
        f"2d subdomains in mdg: {len(mdg.subdomains(dim=2))} "
        "(expected 0 -- it's a constraint, not a fracture)"
    )

    well_network = WellNetwork3d([well], domain)
    try:
        # Expected to raise IndexError: list index out of range, from
        # well_network.py's _add_well_fracture_interfaces:
        #     g_high = mdg.subdomains(dim=mdg.dim_max() - 1)[frac_inds[0]]
        well_network.mesh(fracture_network, mdg, {"cell_size": 5.0})
        print("NO ERROR RAISED -- bug is fixed, or this script is out of date.\n")
    except IndexError as exc:
        print(f"FAILED as expected: {exc!r}\n")
    reset_gmsh()


def demonstrate_workaround() -> None:
    print("--- Workaround: meshing wells against a constraint-free fracture network ---")
    print("--- (only real fractures -- none, in this minimal case) ---")
    domain, well = build_domain_and_well()
    constraint_plane = build_constraint_plane(domain)
    fracture_network = pp.create_fracture_network([constraint_plane], domain=domain)
    mdg = build_mdg(fracture_network, constraints=[0])

    well_network = WellNetwork3d([well], domain)
    fracture_network_no_constraints = pp.create_fracture_network([], domain=domain)
    well_network.mesh(fracture_network_no_constraints, mdg, {"cell_size": 5.0})
    print("SUCCESS: no error when constraints are excluded from well-intersection detection.")
    reset_gmsh()


if __name__ == "__main__":
    demonstrate_bug()
    demonstrate_workaround()
