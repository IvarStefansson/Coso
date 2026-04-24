from typing import Any, Callable

import porepy as pp


class DarcysLawAdEverywhere(
    pp.constitutive_laws.DarcysLawAd,
    pp.constitutive_laws.FouriersLawAd,
):
    pass


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


methods = ["darcy_flux", "fourier_flux"]
dofs = [("faces", 1), ("faces", 1)]


class DarcysLawAdInLowerDimensions(DarcysLawAd, pp.constitutive_laws.AdTpfaFlux):
    def subdomain_lists_and_prolongations(
        self, subdomains: list[pp.Grid] | list[pp.BoundaryGrid]
    ) -> tuple[list[pp.Grid], list[pp.Grid], pp.ad.Operator, pp.ad.Operator]:
        """Split subdomains into lower-dimensional and nD subdomains, and create prolongation operators.

        Parameters:
            subdomains: List of subdomains.
        Returns:
            Tuple containing:
                - List of lower-dimensional subdomains.
                - List of nD subdomains.
                - Prolongation operator for lower-dimensional subdomains.
                - Prolongation operator for nD subdomains.
        """

        subdomain_projections = pp.ad.SubdomainProjections(subdomains, 1)
        sd_lower = [sd for sd in subdomains if sd.dim < self.nd]
        sd_nd = [sd for sd in subdomains if sd.dim == self.nd]
        prolongation_lower = subdomain_projections.face_prolongation(sd_lower)
        prolongation_nd = subdomain_projections.face_prolongation(sd_nd)

        return sd_lower, sd_nd, prolongation_lower, prolongation_nd

    def darcy_flux(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Discretization of the Darcy flux.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator for the Darcy flux discretization.

        """
        if len(subdomains) > 0:
            if isinstance(subdomains[0], pp.BoundaryGrid):
                if not all([isinstance(sd, pp.BoundaryGrid) for sd in subdomains]):
                    raise ValueError(
                        "All subdomains must be BoundaryGrids if one is a BoundaryGrid."
                    )
                return super().darcy_flux(subdomains)
        sd_lower, sd_nd, prolongation_lower, prolongation_nd = (
            self.subdomain_lists_and_prolongations(subdomains)
        )
        darcy_flux_lower = pp.constitutive_laws.DarcysLawAd.darcy_flux(self, sd_lower)
        darcy_flux_nd = super().darcy_flux(sd_nd)
        return prolongation_lower @ darcy_flux_lower + prolongation_nd @ darcy_flux_nd

    def pressure_trace(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Discretization of the Darcy flux.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator for the Darcy flux discretization.

        """
        if len(subdomains) > 0:
            if isinstance(subdomains[0], pp.BoundaryGrid):
                if not all([isinstance(sd, pp.BoundaryGrid) for sd in subdomains]):
                    raise ValueError(
                        "All subdomains must be BoundaryGrids if one is a BoundaryGrid."
                    )
                return super().pressure_trace(subdomains)
        sd_lower, sd_nd, prolongation_lower, prolongation_nd = (
            self.subdomain_lists_and_prolongations(subdomains)
        )
        pressure_trace_lower = pp.constitutive_laws.DarcysLawAd.pressure_trace(
            self, sd_lower
        )
        pressure_trace_nd = super().pressure_trace(sd_nd)
        return (
            prolongation_lower @ pressure_trace_lower
            + prolongation_nd @ pressure_trace_nd
        )

    def fourier_flux(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Discretization of the Fourier flux.

        Parameters:
            domains: List of subdomains.

        Returns:
            Operator for the Fourier flux discretization.

        """
        if len(domains) > 0:
            if isinstance(domains[0], pp.BoundaryGrid):
                if not all([isinstance(sd, pp.BoundaryGrid) for sd in domains]):
                    raise ValueError(
                        "All subdomains must be BoundaryGrids if one is a BoundaryGrid."
                    )
                return super().fourier_flux(domains)
        sd_lower, sd_nd, prolongation_lower, prolongation_nd = (
            self.subdomain_lists_and_prolongations(domains)
        )
        fourier_flux_lower = pp.constitutive_laws.FouriersLawAd.fourier_flux(
            self, sd_lower
        )
        fourier_flux_nd = super().fourier_flux(sd_nd)
        return (
            prolongation_lower @ fourier_flux_lower + prolongation_nd @ fourier_flux_nd
        )

    def temperature_trace(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Discretization of the temperature trace.

        Parameters:
            domains: List of subdomains.

        Returns:
            Operator for the temperature trace discretization.

        """
        if len(domains) > 0:
            if isinstance(domains[0], pp.BoundaryGrid):
                if not all([isinstance(sd, pp.BoundaryGrid) for sd in domains]):
                    raise ValueError(
                        "All subdomains must be BoundaryGrids if one is a BoundaryGrid."
                    )
                return super().temperature_trace(domains)
        sd_lower, sd_nd, prolongation_lower, prolongation_nd = (
            self.subdomain_lists_and_prolongations(domains)
        )
        temperature_trace_lower = pp.constitutive_laws.FouriersLawAd.temperature_trace(
            self, sd_lower
        )
        temperature_trace_nd = super().temperature_trace(sd_nd)
        return (
            prolongation_lower @ temperature_trace_lower
            + prolongation_nd @ temperature_trace_nd
        )
