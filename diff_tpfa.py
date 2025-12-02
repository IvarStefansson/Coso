import porepy as pp
from typing import Callable, Any


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
