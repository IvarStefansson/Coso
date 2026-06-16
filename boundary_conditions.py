from typing import Callable

import numpy as np
import porepy as pp


class InitialCosoBoundaryConditions:
    """Boundary conditions for the Coso geothermal reservoir model.

    We impose a displacement value scaling linearly with x and time on all boundaries
    except the top, where we impose a zero normal traction.
    """

    hydrostatic_pressure: Callable[[np.ndarray], np.ndarray]

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Temperature boundary values.

        Parameters:
            boundary_grid: Boundary grid for which temperature values are to be returned.

        Returns:
            Array of temperature boundary values for each cell in the boundary grid.
        """
        return (
            self.temperature_at_depth(self.depth(boundary_grid.cell_centers))
            * self.initialization_time_dependency_factor()
        )

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Pressure boundary values.

        Parameters:
            boundary_grid: Boundary grid for which pressure values are to be returned.

        Returns:
            Array of pressure boundary values for each cell in the boundary grid.
        """
        return (
            self.hydrostatic_pressure(self.depth(boundary_grid.cell_centers))
            * self.initialization_time_dependency_factor()
        )

    def bc_values_displacement(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Displacement values.

        Parameters:
            boundary_grid: Boundary grid for which boundary values are to be returned.

        Returns:
            Array of boundary values, with one value for each dimension of the
                problem, for each face in the subdomain.

        """

        #
        # if self.time_manager.time > 1e-5:
        #     values = self.displacement_from_coordinates(boundary_grid.cell_centers)
        # else:
        values = np.zeros((self.nd, boundary_grid.num_cells))
        return values.ravel("F") * self.initialization_time_dependency_factor()

    def initialization_time_dependency_factor(self) -> float:
        val = self.time_manager.time / self.time_manager.time_final
        self.ad_time_step_factor.set_value(val)
        return 1  # self.time_manager.time > 0


class CosoBoundaryConditionsDisplacement:
    """Boundary conditions for the Coso geothermal reservoir model.

    We impose a displacement value scaling linearly with x and time on all boundaries
    except the top, where we impose a zero normal traction.
    """

    hydrostatic_pressure: Callable[[np.ndarray], np.ndarray]

    def bc_values_displacement(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Displacement values.

        Parameters:
            boundary_grid: Boundary grid for which boundary values are to be returned.

        Returns:
            Array of boundary values, with one value for each dimension of the
                problem, for each face in the subdomain.

        """

        if boundary_grid.dim < self.nd - 1:
            return super().bc_values_displacement(boundary_grid)
        values = self.boundary_displacement_from_initialization(boundary_grid)
        # Set velocity for all cells. The displacement is scaled with the x-coordinate
        # and time.
        if not self.params["initialization"]:
            coords = boundary_grid.cell_centers
            x_scaling = coords[0] - self.domain.bounding_box["xmin"]
            # y_scaling = coords[1] - self.domain.bounding_box["ymin"]

            offset_time = 0 * pp.YEAR if self.time_manager.time > 0 else 0
            vals_time = np.outer(self.boundary_displacement_velocity, x_scaling) * (
                self.time_manager.time + offset_time
            )
            values += vals_time.ravel("F")
        return values

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Boundary condition type for mechanics.

        Dirichlet boundary conditions are defined on all boundaries except the top.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        dir_sides = self.domain_boundary_sides(sd).all_bf
        bc = pp.BoundaryConditionVectorial(sd, dir_sides, "dir")
        bc.internal_to_dirichlet(sd)
        return bc

    @property
    def boundary_displacement_velocity(self) -> np.ndarray:
        """Displacement velocity on the boundary [m*s^-1*m^-1].

        Returns:
            Array of displacement velocities of shape (nd,).
        """
        # Relative displacement rate according to the following paper: Satellite
        # observations of surface deformation at the Coso Geothermal Field, California
        """Surface deformation time series and rates are identified at the Coso
        Geothermal Field (CGF) and surrounding areas by applying interferometric
        synthetic aperture radar (InSAR) to satellite scenes from Envisat (June 2004 ̶
        October 2010) and Sentinel (November 2014–April 2018). The measurements are done
        in the line of sight (LOS) to each satellite, within an area of size ~450 km2,
        at the locations of hundreds of thousands permanent and distributed scatterers.
        Thirty descending (satellite moves north to south) and 45 ascending (south to
        north) images were used from Envisat, and 63 descending and 65 ascending from
        Sentinel. A decomposition into average vertical and east horizontal components
        is also performed in more than 35,000 100-m pixels where both types of LOS
        measurements are available. The main observations at CGF include: (1) a
        subsidence area of size ~70 km2, with a maximum subsidence of –27.6 mm/year for
        the Envisat period and lower maximum subsidence of –19.1 mm/year for the
        Sentinel period; (2) eastward movements in the western part of the subsidence
        area, with Envisat maximum of +23.9 mm/year and a lower Sentinel maximum of
        +15.9 mm/year; (3) westward displacements in the eastern part of the subsidence
        area, with Envisat maximum of ̶ 14.2 mm/year and Sentinel maximum of –11.9
        mm/year; (4) very good agreement of the InSAR observations with leveling survey
        data; (5) earthquake clusters in the subsidence area and hypocentral
        cross-sections showing clusters at various depths and migration in time; and (6)
        good predictions of the overall geothermal resource, based on poroelastic
        modeling using both leveling and InSAR data. The ultimate goal of the project is
        to provide geothermal operators with tools that can be used in reservoir
        management."""
        # The relative displacement rate is approx. 24 + 14 mm/year.
        # The side length of the Coso Geothermal Field is roughly 10 km.
        # rate = (38 * pp.MILLI * pp.METER / (10 * (pp.KILO * pp.METER)) / pp.YEAR)
        rate = self.params.get("boundary_displacement_velocity")
        # , 38 * pp.MILLI * pp.METER / (10 * (pp.KILO * pp.METER)) / pp.YEAR,)
        return self.units.convert_units(rate, "s^-1") * np.array([1.0, 0.0, 0.0])


class CosoBoundaryConditions:
    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Pressure boundary values.

        Parameters:
            boundary_grid: Boundary grid for which pressure values are to be returned.

        Returns:
            Array of pressure boundary values for each cell in the boundary grid.
        """
        if boundary_grid.dim == 2:
            return self.hydrostatic_pressure(self.depth(boundary_grid.cell_centers))
        # Else, we are at the top of a well. The pressure is set to the well head
        # pressure.
        # Get the parent well of the subdomain.
        sd = boundary_grid.parent
        parent_well = self.parent_well(sd)
        vals = np.zeros(boundary_grid.num_cells)
        domain_sides = self.domain_boundary_sides(boundary_grid)
        if parent_well is None or not self.wells_active():
            # We are not at a well or not in the schedule yet.
            vals[domain_sides.top] = self.reference_variable_values.pressure
        else:
            # Get the well data for the well.
            well_data = parent_well.data
            # Get the well head pressure for the well.
            vals[domain_sides.top] = well_data["well_head_pressure"].values[
                self.well_protocol_index(well_data)
            ]
        return vals

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Temperature boundary values.

        Parameters:
            boundary_grid: Boundary grid for which temperature values are to be returned.

        Returns:
            Array of temperature boundary values for each cell in the boundary grid.
        """
        if boundary_grid.dim == 2:
            return self.temperature_at_depth(self.depth(boundary_grid.cell_centers))
        # Get the parent well of the subdomain.
        sd = boundary_grid.parent
        parent_well = self.parent_well(sd)
        vals = np.zeros(boundary_grid.num_cells)
        ind = self.domain_boundary_sides(boundary_grid).top
        if parent_well is None or not self.wells_active():
            # We are not at a well or not in the schedule yet.
            vals[ind] = self.reference_variable_values.temperature
        else:
            # Get the well data for the well.
            well_data = parent_well.data
            # Get the temperature for the well.
            vals[ind] = well_data["temperature"].values[
                self.well_protocol_index(well_data)
            ]
        return vals

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type for Darcy flux.

        For the 3d domain, Dirichlet conditions are defined on all boundaries. For
        injection wells, Neumann conditions are imposed. Dirichlet conditions are
        set on production wells.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        return self._bc_type_diffusion(sd)

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type for Fourier flux.

        For the 3d domain, Dirichlet conditions are defined on all boundaries. For
        injection wells, Neumann conditions are imposed. Dirichlet conditions are
        set on production wells.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        domain_sides = self.domain_boundary_sides(sd)
        parent_well = self.parent_well(sd)
        if (
            self.is_injection_well(parent_well) or self.is_production_well(parent_well)
        ) and not self.wells_active():
            return pp.BoundaryCondition(sd, domain_sides.top, "neu")

        return pp.BoundaryCondition(sd, domain_sides.all_bf, "dir")

    def _bc_type_diffusion(self, sd: pp.Grid) -> pp.BoundaryCondition:
        domain_sides = self.domain_boundary_sides(sd)
        parent_well = self.parent_well(sd)
        if self.is_injection_well(parent_well):
            # This is an injection well subdomain. It always has Neumann conditions,
            # albeit with different values depending on whether we are in the schedule
            # or not.
            neumann_sides = domain_sides.top
            return pp.BoundaryCondition(sd, neumann_sides, "neu")
        if self.is_production_well(parent_well) and not self.wells_active():
            # This is a production well subdomain, but we are not in the schedule
            # yet. Set (zero) Neumann conditions on the top boundary. If we are in the
            # schedule, we will set Dirichlet conditions on the top boundary.
            neumann_sides = domain_sides.top
            bc = pp.BoundaryCondition(sd, neumann_sides, "neu")
            return bc

        dirichlet_sides = domain_sides.all_bf
        bc = pp.BoundaryCondition(sd, dirichlet_sides, "dir")
        return bc

    def bc_values_darcy_flux(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Darcy flux boundary values.

        Parameters:
            boundary_grid: Boundary grid for which Darcy flux values are to be returned.

        Returns:
            Array of Darcy flux boundary values for each cell in the grid.
        """
        # Only need to set nonzero values for the injection wells.
        sd = boundary_grid.parent
        # Get the parent well of the subdomain.
        parent_well = self.parent_well(sd)
        vals = np.zeros(boundary_grid.num_cells)

        if self.is_injection_well(parent_well):
            domain_sides = self.domain_boundary_sides(boundary_grid)
            neumann_sides = domain_sides.top
            if not np.any(neumann_sides):
                return vals
            # Get the well data for the well.
            well_data = parent_well.data
            # Get the mass rate for the well.

            if not self.wells_active():
                return np.zeros(boundary_grid.num_cells)
            mass_rate = well_data["mass_rate"].values[
                self.well_protocol_index(well_data)
            ]
            # Get the Darcy flux values for the well.
            darcy_flux = mass_rate / self.equation_system.evaluate(
                self.advection_weight_mass_balance([boundary_grid])
            )
            # if self.well_protocol_index() < 2:
            #     # If we are before the first well protocol, we assume a zero flux.
            #     darcy_flux = np.zeros_like(darcy_flux)
            vals[neumann_sides] = -darcy_flux[neumann_sides]
        return vals


class NeumannWellBCsFromSchedule(pp.PorePyModel):
    """Class defining Neumann BCs on well grids during the first time interval."""

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Return Neumann BC for Darcy flux on well grids during first time interval.

        Parameters:
            sd: The subdomain for which to return the BC type.

        Returns:
            The boundary condition type for Darcy flux on the given subdomain.
        """
        if self.is_well_grid(sd) and self.neumann_bcs_active(sd):
            # Before start of injection, impose Neumann BCs on well grids. A zero-flux
            # condition is imposed by default when no BC values are specified. The <=
            # comparison ensures that the BCs are kept as Neumann as long as the time
            # step is within the first time interval [0, t1], where t1 is the start time
            # of injection, consistent with the implicit time stepping employed in
            # PorePy.
            domain_sides = self.domain_boundary_sides(sd)
            inds = domain_sides.top if self.nd == 3 else domain_sides.north
            return pp.BoundaryCondition(sd, inds, "neu")
        else:
            has_domain_sizes = hasattr(self, "domain_sizes")
            has_reservoir_depth = hasattr(self, "reservoir_depth")
            if (
                has_domain_sizes
                and has_reservoir_depth
                and np.isclose(self.domain_sizes()[2], self.reservoir_depth())
            ):
                # Domain boundary coincides with the reservoir bottom, so we set
                # 0 Neumann.
                faces = self.domain_boundary_sides(sd).bottom
                return pp.BoundaryCondition(sd, faces, "neu")
            return super().bc_type_darcy_flux(sd)  # type: ignore[misc]

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Return Neumann BC for Fourier flux on well grids during first time interval.

        Parameters:
            sd: The subdomain for which to return the BC type.

        Returns:
            The boundary condition type for Fourier flux on the given subdomain.
        """
        if self.is_well_grid(sd) and self.neumann_bcs_active(sd):
            # Before start of injection, impose Neumann BCs on well grids. A zero-flux
            # condition is imposed by default when no BC values are specified. The <=
            # comparison ensures that the BCs are kept as Neumann as long as the time
            # step is within the first time interval [0, t1], where t1 is the start time
            # of injection, consistent with the implicit time stepping employed in
            # PorePy.
            domain_sides = self.domain_boundary_sides(sd)
            inds = domain_sides.top if self.nd == 3 else domain_sides.north
            return pp.BoundaryCondition(sd, inds, "neu")
        else:
            return super().bc_type_fourier_flux(sd)  # type: ignore[misc]

    def neumann_bcs_active(self, sd: pp.Grid) -> bool:
        """Check if Neumann BCs on well grids are active.

        Parameters:
            sd: The subdomain for which to check if Neumann BCs are active.

        Returns:
            True if the current time is within the first time interval, False otherwise.
        """
        is_neumann = False
        if self.params.get("neumann_intervals") is not None:
            neumann_intervals = self.params["neumann_intervals"]
            current_time = self.time_manager.time
            if np.isclose(current_time, 0.0):
                # At time zero, we are before the first time interval, so Neumann BCs are active.
                is_neumann = True
                return is_neumann
            for interval in neumann_intervals:
                if interval[0] < current_time <= interval[1]:
                    is_neumann = True
                    break
        return is_neumann

    def bc_values_darcy_flux(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Return boundary values for pressure on all boundaries.

        Parameters:
            bg: The boundary grid for which to return the BC values.

        Returns:
            The boundary values for pressure on the given boundary grid.
        """
        sd = bg.parent
        # Ignore super call for type checking, as it is assumed to be present for this
        # mixin class.
        values = super().bc_values_darcy_flux(bg)  # type: ignore[misc]
        if self.is_well_grid(sd) and "darcy_fluxes" in self.well_protocol_variables():
            well = self.well_network.wells[sd.tags["parent_well_index"]]
            well_tag = well.tags["well_name"]
            values = self.well_protocols(well_tag, "darcy_fluxes")
            # Find indices of the well boundary sides.
            domain_sides = self.domain_boundary_sides(bg)
            # The top of the domain is '.top' in 3d, '.north' in 2d.
            inds = domain_sides.top if self.nd == 3 else domain_sides.north
            # Set pressure values according to the well protocol.
            values[inds] = self.units.convert_units(
                self.get_well_value(
                    values,
                    self.time_manager.schedule,
                    self.time_manager.time,
                ),
                "Pa",
            )
        return values

    def well_protocol_variables(self) -> list[str]:
        """List of variable names for which well protocols are defined."""
        return ["temperatures", "pressures"]


class OnlyInjectionWellNeumannBCsFromSchedule(NeumannWellBCsFromSchedule):
    """Class defining Neumann BCs only on injection well grids during the first time interval."""

    def neumann_bcs_active(self, sd: pp.Grid) -> bool:
        well = self.parent_well(sd)
        if well is None:
            return False
        is_injection = well.tags["well_name"] in self.injection_well_names
        return super().neumann_bcs_active(sd) and is_injection

    def well_protocols(self, well_tag: str, variable: str) -> NDArray[np.float64]:
        """Dictionary mapping well tags to well protocols.

        Returns:
            Dictionary with well protocols, each containing a dictionary with
            time-dependent temperatures and pressures, with each value being an array of
            size equal to the number of scheduled times in the time manager.
        """
        num_times = self.time_manager.schedule.size
        protocols: dict[str, dict[str, NDArray[np.float64]]] = {}
        # Initialize protocol dictionary for the well.
        protocols[well_tag] = {}
        # Set values for temperatures and pressures.
        input_values = self.params.get(f"{well_tag}_{variable}", 0.0)
        if isinstance(input_values, (float, int)):
            # Broadcast single value to all time steps for convenient user
            # definition of well protocols.
            values = np.full(num_times, input_values, dtype=float)

        elif isinstance(input_values, (list, np.ndarray)):
            # Enforce array of float values.
            values = np.array(input_values, dtype=float)
            if values.size != num_times:
                raise ValueError(
                    f"Well protocol for {well_tag} {variable} has size "
                    f"{values.size}, expected {num_times}."
                )
        else:
            raise TypeError(
                f"Well protocol for {well_tag} {variable} has unsupported "
                f"type {type(input_values)}."
            )

        return values


class SmoothWellTransitions(NeumannWellBCsFromSchedule):
    """Smooth Neumann↔Dirichlet transitions at production well boundaries.

    At production-start (Neumann → Dirichlet): the BC type remains Dirichlet
    (as it would be outside the shut-in window), but the prescribed pressure is
    linearly ramped from the last recorded reservoir pressure at the well top
    face toward the scheduled well-head pressure over ``well_transition_duration``
    seconds.

    At shut-in start (Dirichlet → Neumann): the BC type remains Neumann (as
    returned by :class:`NeumannWellBCsFromSchedule`), but the prescribed Darcy
    flux is linearly ramped from the last recorded converged value toward zero
    over ``well_transition_duration`` seconds.

    Set ``"well_transition_duration"`` (float, seconds) in the model params to
    activate smoothing.  If absent or zero no smoothing is applied and the class
    behaves identically to :class:`NeumannWellBCsFromSchedule`.
    """

    def _transition_duration(self) -> float:
        dt = float(self.params.get("well_transition_duration", 0.0))
        neumann_intervals = self.params.get("neumann_intervals", [])
        if dt > 0.0 and neumann_intervals:
            # Each Neumann interval is (shut_in_start, shut_in_start + shut_in_duration
            # - transition_duration).  Recover the original shut_in_duration from the
            # first interval and verify the transition fits within it.
            start, end = neumann_intervals[0]
            shut_in_duration = end - start + dt  # end = start + d_shutin - dt
            if dt > shut_in_duration:
                raise ValueError(
                    f"well_transition_duration ({dt:.3g} s) exceeds the shut-in "
                    f"duration ({shut_in_duration:.3g} s). The two ramp windows "
                    "would overlap."
                )
        return dt

    def _transition_progress(self, start_times: list[float]) -> float | None:
        """Return linear progress ∈ (0, 1] if the current time is inside a
        transition window starting at one of ``start_times``, else ``None``."""
        dt = self._transition_duration()
        if dt <= 0.0:
            return None
        t = self.time_manager.time
        for t0 in start_times:
            if t0 < t <= t0 + dt:
                return (t - t0) / dt
        return None

    def _shutin_start_times(self) -> list[float]:
        """Start time of each shut-in interval (Dirichlet → Neumann transition)."""
        return [s for s, _ in self.params.get("neumann_intervals", [])]

    def _production_start_times(self) -> list[float]:
        """Start of each production ramp window (Neumann → Dirichlet transition).

        These are the *end* times of the Neumann intervals stored in params, which
        are ``well_transition_duration`` seconds *before* the true shut-in end (cycle
        boundary).  The ramp window therefore spans
        ``(neumann_end, neumann_end + well_transition_duration]``, i.e. the closing
        ``well_transition_duration`` seconds of each shut-in period.

        If an initial shut-in interval ``(0, t_end)`` is included in
        ``neumann_intervals``, its end time ``t_end`` appears first so that the
        opening ramp at simulation start is handled by the same mechanism as
        mid-simulation production ramps.
        """
        return [e for _, e in self.params.get("neumann_intervals", [])]

    def _top_face_index(self, sd: pp.Grid) -> int | None:
        """Index of the top boundary face on a well grid, or ``None``."""
        domain_sides = self.domain_boundary_sides(sd)
        top_faces = domain_sides.top if self.nd == 3 else domain_sides.north
        idxs = np.where(top_faces)[0]
        return int(idxs[0]) if len(idxs) > 0 else None

    def after_nonlinear_convergence(self) -> None:
        """Cache Darcy flux and pressure at well top faces.

        Called after each converged time step.  The cached values are used as
        the starting point for the ramps at the next transition boundary.
        Covers both production and injection wells.
        """
        super().after_nonlinear_convergence()
        if self._transition_duration() <= 0.0:
            return
        if not hasattr(self, "_cached_well_top_flux"):
            self._cached_well_top_flux: dict[str, float] = {}
            self._cached_well_top_pressure: dict[str, float] = {}
        for sd in self.mdg.subdomains(dim=1):
            if not self.is_well_grid(sd):
                continue
            parent_well = self.parent_well(sd)
            if not (
                self.is_production_well(parent_well)
                or self.is_injection_well(parent_well)
            ):
                continue
            face_idx = self._top_face_index(sd)
            if face_idx is None:
                continue
            well_name = parent_well.tags["well_name"]
            # Only update each cache while the corresponding ramp is NOT active.
            # Updating mid-ramp would re-anchor the starting value and distort all
            # subsequent steps in that window (the formula would use a partially-ramped
            # value as its "from" endpoint rather than the true pre-ramp value).
            if self._transition_progress(self._shutin_start_times()) is None:
                # Not inside a shut-in-start ramp: safe to refresh flux cache.
                flux_vals = self.darcy_flux([sd]).value(self.equation_system)
                # When setting BC values, the sign of the flux is with respect to the
                # outward normal of the boundary. However, the flux values returned by
                # the equation system are with respect to the normal of the face. Flip
                # sign if actual normal is opposite to outward normal.
                out_normal = (
                    np.array([0.0, 0.0, 1.0]) if self.nd == 3 else np.array([0.0, 1.0])
                )
                outwards_normal_sign = sd.face_normals[:, face_idx] @ out_normal
                self._cached_well_top_flux[well_name] = float(
                    flux_vals[face_idx] * outwards_normal_sign
                )
            if self._transition_progress(self._production_start_times()) is None:
                # Not inside a production-start ramp: safe to refresh pressure cache.
                p_vals = self.pressure_trace([sd]).value(self.equation_system)
                self._cached_well_top_pressure[well_name] = float(p_vals[face_idx])

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Ramp well-head pressure from the last known reservoir pressure toward the
        scheduled well-head pressure over the transition window at production-start.

        Applies to both production and injection wells. Outside all ramp windows
        the values are identical to the parent implementation.
        """
        vals = super().bc_values_pressure(bg)  # type: ignore[misc]
        sd = bg.parent
        parent_well = self.parent_well(sd)
        if not (
            self.is_well_grid(sd)
            and (
                self.is_production_well(parent_well)
                or self.is_injection_well(parent_well)
            )
        ):
            return vals
        alpha = self._transition_progress(self._production_start_times())
        if alpha is None:
            return vals
        domain_sides = self.domain_boundary_sides(bg)
        top_inds = domain_sides.top if self.nd == 3 else domain_sides.north
        if not np.any(top_inds):
            return vals
        top_idx = int(np.where(top_inds)[0][0])
        well_name = parent_well.tags["well_name"]
        # Use cached reservoir pressure; fall back to hydrostatic if unavailable
        # (e.g., at the very first production-start after a shut-in with no prior
        # converged production step).
        p_reservoir = getattr(self, "_cached_well_top_pressure", {}).get(
            well_name,
            float(self.hydrostatic_pressure(self.depth(bg.cell_centers))[top_idx]),
        )
        p_target = float(vals[top_idx])
        vals[top_inds] = (1.0 - alpha) * p_reservoir + alpha * p_target
        return vals

    def bc_values_darcy_flux(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Ramp Darcy flux to zero at both wells during the shut-in transition.

        Applies to both production and injection wells. Outside the transition
        window the values are identical to the parent implementation.
        """
        vals = super().bc_values_darcy_flux(bg)  # type: ignore[misc]
        sd = bg.parent
        parent_well = self.parent_well(sd)
        if not (
            self.is_well_grid(sd)
            and (
                self.is_production_well(parent_well)
                or self.is_injection_well(parent_well)
            )
        ):
            return vals
        alpha = self._transition_progress(self._shutin_start_times())
        if alpha is None:
            return vals
        domain_sides = self.domain_boundary_sides(bg)
        top_inds = domain_sides.top if self.nd == 3 else domain_sides.north
        if not np.any(top_inds):
            return vals
        well_name = parent_well.tags["well_name"]
        q_last = getattr(self, "_cached_well_top_flux", {}).get(well_name, 0.0)
        # alpha: 0 at transition start → 1 at end; ramp q_last → 0.
        vals[top_inds] = (1.0 - alpha) * q_last
        return vals
