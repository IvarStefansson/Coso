from typing import Any, Callable
import porepy as pp
import numpy as np
from geometry import CosoGeometry, EllipticFractureGeometry
from material_parameters import granodiorite_values

from physical_model import PhysicalModel
from boundary_conditions import CosoBoundaryConditions
from initial_conditions import InitialCondition, CopyInitialCondition
from exporting import CosoExporter, IterationExporting
from porepy.numerics.nonlinear import line_search

import logging
import sys
import copy

logger = logging.getLogger("run_example_1")
logging.basicConfig(level=logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class DarcysLawAd(pp.constitutive_laws.DarcysLawAd):
    def darcy_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Discretization of the Darcy flux.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator for the Darcy flux discretization.

        """
        if all([sd.dim < self.nd for sd in subdomains]):
            return pp.ad.TpfaAd(self.darcy_keyword, subdomains)
        else:
            return super().darcy_flux_discretization(subdomains)

    def fourier_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Discretization of the Darcy flux.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator for the Darcy flux discretization.

        """
        if all([sd.dim < self.nd for sd in subdomains]):
            return pp.ad.TpfaAd(self.fourier_keyword, subdomains)
        else:
            return super().fourier_flux_discretization(subdomains)


def override_methods(
    cls,
    method_name: list[str],
    dofs: list[str, int],
    sort_criterion: Callable[[Any, pp.GridLike], bool] = None,
):
    if sort_criterion is None:
        sort_criterion = lambda g, cls: True

    def new_method(self, domains):
        super_method = getattr(super(cls, self), method_name)

        if len(domains) == 0 or all([isinstance(g, pp.BoundaryGrid) for g in domains]):
            return super_method(domains)

        domains_h = [g for g in domains if sort_criterion(self, g)]
        domains_l = [g for g in domains if not sort_criterion(self, g)]
        proj = pp.ad.SubdomainProjections(domains, dofs[1])
        dof_type = dofs[0]
        # Check if dof_type value is plural (e.g. "faces"). If so, remove the last
        # character to get the singular form.
        if dof_type[-1] == "s":
            dof_type = dof_type[:-1]
        prol_h = getattr(proj, dof_type + "_prolongation")(domains_h)
        prol_l = getattr(proj, dof_type + "_prolongation")(domains_l)
        result = prol_h @ super_method(domains_h) + prol_l @ super_method(domains_l)
        return result

    setattr(cls, method_name, new_method)


def sort_criterion(cls, domain):
    return domain.dim == cls.nd


methods = ["darcy_flux"]
dofs = [("faces", 1)]


class SolutionStrategy:
    """Solution strategy for the Coso geothermal reservoir."""

    # def after_nonlinear_convergence(self):
    #     """Prepare the model for the nonlinear loop."""
    #     if self.time_manager.time == self.time_manager.schedule[1]:
    #         self.assign_wells()
    #     super().after_nonlinear_convergence()

    # def assign_wells(self):
    #     self.params["use_wells"] = True
    #     self.set_well_network()
    #     self.equation_system = pp.ad.EquationSystem(self.mdg)
    #     self.create_variables()
    #     self.set_equations()
    #     self.params["file_name"] = self.params["file_name"].strip("_initialize")
    #     self.initialize_data_saving()
    def assemble_linear_system(self) -> None:
        """Assemble the linearized system and store it in :attr:`linear_system`.

        The linear system is defined by the current state of the model.

        """

        def add_domain(d):
            if isinstance(d, pp.MortarGrid):
                # Skip mortar domains if their primary subdomain is a well
                g = self.mdg.interface_to_subdomain_pair(d)[0]
            else:
                g = d
            if self.is_well(g):
                # Skip well equations
                return False
            elif g.tags.get("parent_well_index", -1) > -1:
                # Skip well equations
                return False
            return True

        if self.time_manager.time < self.time_manager.schedule[1] + 1e-5:
            variables = []
            for var in self.equation_system.variables:
                if "well" in var.name:
                    # Skip well variables
                    continue
                if add_domain(var.domain):
                    variables.append(var)

            equations = {}
            for name, eq in self.equation_system.equations.items():
                # Skip well equations
                if "well" in name:
                    continue
                doms = self.equation_system._equation_image_space_composition[name]
                domains = [d for d in doms.keys() if add_domain(d)]
                equations[name] = domains

            self.assembled_variables = variables
            sub = self.equation_system.sub(
                variables=variables,
                equations=equations,
            )
            self.linear_system = sub.assemble()
        else:
            self.assembled_variables = self.equation_system.variables
            super().assemble_linear_system()


class Model(
    IterationExporting,
    CosoExporter,
    EllipticFractureGeometry,
    InitialCondition,
    CosoBoundaryConditions,
    DarcysLawAd,
    pp.models.solution_strategy.ContactIndicators,
    PhysicalModel,
):
    """Model for the Coso geothermal reservoir."""


class CopyModel(
    CopyInitialCondition,
    Model,
):
    """Model for the Coso geothermal reservoir."""


for method, dof in zip(methods, dofs):
    override_methods(CopyModel, method, dof, sort_criterion)


class ConstraintLineSearchNonlinearSolver(
    line_search.ConstrainUpdateLineSearch,
    line_search.ConstrainVariableLineSearch,
    line_search.ConstraintLineSearch,
    line_search.SplineInterpolationLineSearch,
    line_search.LineSearchNewtonSolver,
):
    pass


if __name__ == "__main__":
    fast = 1 == 11  # Set to 1 for fast run, 0 for full run
    # Define the time parameters
    logger.info("Starting the simulation")
    dt = 5
    injection_start_time = 20
    schedule = np.arange(25) * pp.DAY
    if fast:
        injection_start_time = 10  # 0.2 * pp.YEAR
        schedule = np.arange(2) * 10.0  # pp.HOUR
    schedule += injection_start_time
    # Add the initial time step to the schedule
    schedule = np.insert(schedule, 0, 0)  # Initial time step at 0
    time_manager = pp.TimeManager(
        schedule=schedule,
        dt_init=dt,
        dt_min_max=(1, pp.YEAR),
        iter_max=20,
        iter_optimal_range=(5, 12),
        iter_relax_factors=(0.5, 2.0),
        recomp_factor=0.2,
        recomp_max=5,
    )
    dt_init = 10 * pp.YEAR
    time_manager_init = pp.TimeManager(
        [0, dt_init],
        dt_init=dt_init,
        dt_min_max=(1, 2 * dt_init),
        constant_dt=False,
    )
    cell_size = 9e2
    if fast:
        cell_size = 11e2
    model_params_init = {
        "material_constants": {
            "solid": pp.SolidConstants(**granodiorite_values),
            "fluid": pp.FluidComponent(**pp.fluid_values.water),
        },
        "time_manager": time_manager_init,
        "grid_type": "simplex",
        "meshing_arguments": {
            "cell_size": cell_size,
            "cell_size_fracture": 0.4 * cell_size,
        },
        "file_name": "elliptic_fractures_initialize",
        "data_folder_name": "saved_data",
        "adaptive_indicator_scaling": 1,  # Scale the indicator adaptively to increase robustness
        "use_wells": False,
        "reference_variable_values": pp.ReferenceVariableValues(
            temperature=293.15, pressure=1e5
        ),
    }
    if fast:
        model_params_init["folder_name"] = "fast_runs"
        model_params_init["data_folder_name"] = "saved_data_fast_runs"
    model_params = copy.deepcopy(model_params_init)
    model_params.update(
        {
            "time_manager": time_manager,
            "file_name": "elliptic_fractures",
            "use_wells": True,
        }
    )
    # Create the model
    init_model = Model(model_params_init)
    solver_params = {
        "nl_convergence_tol_res": 1e-1,
        "nl_convergence_tol": 1e-3,  # Seems to be the best we can do with current condition number
        "max_iterations": 30,
        "nonlinear_solver": ConstraintLineSearchNonlinearSolver,
        "local_line_search": 1,
        "variable_line_search": 0,
        "global_line_search": 0,
        "update_line_search": 0,
        "residual_line_search_interval_size": 1e-5,
        "constrained_variables_and_ranges": {
            "pressure": (-1e5, 1e8),
            "temperature": (50, 5000),
        },
        "constrained_variable_updates": {
            "pressure": 1e7,
            "temperature": 1e3,
            "interface_darcy_flux": 1e-2,
            "well_flux": 1e-2,
        },
        "linear_solver": "scipy_sparse",
    }
    pp.run_time_dependent_model(init_model, solver_params)
    init_model.fracture_gap(init_model.mdg.subdomains(dim=2))
    model = CopyModel(model_params)
    model.initialization_model = init_model
    solver_params.update(
        {
            "local_line_search": 1,
            "update_line_search": 0,
            "nl_convergence_tol_res": 5e0,
        }
    )
    pp.run_time_dependent_model(model, solver_params)
    model.plot_well_monitoring()
