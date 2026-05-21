from __future__ import annotations

from typing import Optional, Union

import numpy as np
import porepy as pp
from numpy.typing import ArrayLike


class CosoTimeManager(pp.TimeManager):
    """Extension of :class:`pp.TimeManager` with per-interval control parameters.

    Each of ``dt_min_max``, ``iter_max``, ``iter_optimal_range``,
    ``iter_relax_factors``, ``recomp_factor``, and ``recomp_max`` may be supplied
    either as a single value/pair (as in the base class) **or** as a sequence of
    length ``schedule.size - 1``, providing one value per schedule interval.

    When a list is provided, the value for the interval that contains the current
    simulation time is selected automatically before each adaptation/correction step.
    """

    # ------------------------------------------------------------------
    # Helpers: distinguish "single value/pair" from "per-interval list"
    # ------------------------------------------------------------------

    @staticmethod
    def _is_pair_list(val) -> bool:
        """Return True if *val* is a list/tuple of pairs, not a single pair.

        Heuristic: if the first element is itself a non-numeric sequence, *val* is
        treated as a collection of pairs.
        """
        if val is None:
            return False
        try:
            items = list(val)
        except TypeError:
            return False
        if not items:
            return False
        try:
            float(items[0])  # first element is a plain number -> single pair
            return False
        except (TypeError, ValueError):
            return True  # first element is a sequence -> list of pairs

    @staticmethod
    def _is_scalar_list(val) -> bool:
        """Return True if *val* is a list/tuple of scalars, not a single scalar."""
        if val is None:
            return False
        try:
            float(val)
            return False  # it IS a single number
        except (TypeError, ValueError):
            return True  # it's a sequence of numbers

    # ------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------

    def __init__(
        self,
        schedule: ArrayLike,
        dt_init: Union[int, float],
        constant_dt: bool = False,
        dt_min_max: Optional[Union[tuple, list[tuple]]] = None,
        iter_max: Union[int, list] = 15,
        iter_optimal_range=(4, 7),
        iter_relax_factors=(0.7, 1.3),
        recomp_factor: Union[float, list] = 0.5,
        recomp_max: Union[int, list] = 10,
        print_info: bool = False,
        rtol: float = 1e-10,
        atol: float = 1e-16,
    ) -> None:
        n = np.asarray(schedule).size - 1  # number of intervals

        def _check_len(name: str, seq) -> None:
            if len(seq) != n:
                raise ValueError(
                    f"'{name}' has length {len(seq)} but schedule has {n} interval(s)"
                    f" (schedule.size - 1 = {n})."
                )

        # --- dt_min_max ---
        if self._is_pair_list(dt_min_max):
            vals = list(dt_min_max)
            _check_len("dt_min_max", vals)
            self._dt_min_max_intervals: Optional[list] = [tuple(v) for v in vals]
            dt_min_max_for_super = tuple(vals[0])
        else:
            self._dt_min_max_intervals = None
            dt_min_max_for_super = dt_min_max

        # --- iter_max ---
        if self._is_scalar_list(iter_max):
            vals_im = list(iter_max)
            _check_len("iter_max", vals_im)
            self._iter_max_intervals: Optional[list] = vals_im
            iter_max_for_super = int(vals_im[0])
        else:
            self._iter_max_intervals = None
            iter_max_for_super = iter_max

        # --- iter_optimal_range ---
        if self._is_pair_list(iter_optimal_range):
            vals_ior = list(iter_optimal_range)
            _check_len("iter_optimal_range", vals_ior)
            self._iter_optimal_range_intervals: Optional[list] = [
                tuple(v) for v in vals_ior
            ]
            iter_optimal_range_for_super = tuple(vals_ior[0])
        else:
            self._iter_optimal_range_intervals = None
            iter_optimal_range_for_super = iter_optimal_range

        # --- iter_relax_factors ---
        if self._is_pair_list(iter_relax_factors):
            vals_irf = list(iter_relax_factors)
            _check_len("iter_relax_factors", vals_irf)
            self._iter_relax_factors_intervals: Optional[list] = [
                tuple(v) for v in vals_irf
            ]
            iter_relax_factors_for_super = tuple(vals_irf[0])
        else:
            self._iter_relax_factors_intervals = None
            iter_relax_factors_for_super = iter_relax_factors

        # --- recomp_factor ---
        if self._is_scalar_list(recomp_factor):
            vals_rf = list(recomp_factor)
            _check_len("recomp_factor", vals_rf)
            self._recomp_factor_intervals: Optional[list] = vals_rf
            recomp_factor_for_super = float(vals_rf[0])
        else:
            self._recomp_factor_intervals = None
            recomp_factor_for_super = recomp_factor

        # --- recomp_max ---
        if self._is_scalar_list(recomp_max):
            vals_rm = list(recomp_max)
            _check_len("recomp_max", vals_rm)
            self._recomp_max_intervals: Optional[list] = vals_rm
            recomp_max_for_super = int(vals_rm[0])
        else:
            self._recomp_max_intervals = None
            recomp_max_for_super = recomp_max

        super().__init__(
            schedule=schedule,
            dt_init=dt_init,
            constant_dt=constant_dt,
            dt_min_max=dt_min_max_for_super,
            iter_max=iter_max_for_super,
            iter_optimal_range=iter_optimal_range_for_super,
            iter_relax_factors=iter_relax_factors_for_super,
            recomp_factor=recomp_factor_for_super,
            recomp_max=recomp_max_for_super,
            print_info=print_info,
            rtol=rtol,
            atol=atol,
        )

    # ------------------------------------------------------------------
    # Current-interval index
    # ------------------------------------------------------------------

    def _get_interval_idx(self, is_recomputing: bool = False) -> int:
        """Return the 0-based index of the current schedule interval.

        After :meth:`_correction_based_on_schedule` re-syncs ``_scheduled_idx``
        to ``searchsorted(schedule, time, side='right')``, the current interval
        is ``_scheduled_idx - 1``.

        When called from :meth:`_adaptation_based_on_recomputation` (before the
        internal ``_scheduled_idx`` decrement for a schedule-boundary hit),
        *is_recomputing* should be ``True`` so the index is offset by one.
        """
        n = len(self.schedule) - 1
        if is_recomputing and self._is_about_to_hit_schedule:
            idx = self._scheduled_idx - 2
        else:
            idx = self._scheduled_idx - 1
        return max(0, min(idx, n - 1))

    # ------------------------------------------------------------------
    # Overrides: inject per-interval values via temporary attribute swap
    # ------------------------------------------------------------------

    def _correction_based_on_dt_min(self) -> None:
        if self._dt_min_max_intervals is not None:
            old = self.dt_min_max
            self.dt_min_max = self._dt_min_max_intervals[self._get_interval_idx()]
            super()._correction_based_on_dt_min()
            self.dt_min_max = old
        else:
            super()._correction_based_on_dt_min()

    def _correction_based_on_dt_max(self) -> None:
        if self._dt_min_max_intervals is not None:
            old = self.dt_min_max
            self.dt_min_max = self._dt_min_max_intervals[self._get_interval_idx()]
            super()._correction_based_on_dt_max()
            self.dt_min_max = old
        else:
            super()._correction_based_on_dt_max()

    def _adaptation_based_on_iterations(self, iterations) -> None:
        idx = self._get_interval_idx()
        old_im = self.iter_max
        old_ior = self.iter_optimal_range
        old_irf = self.iter_relax_factors
        if self._iter_max_intervals is not None:
            self.iter_max = self._iter_max_intervals[idx]
        if self._iter_optimal_range_intervals is not None:
            self.iter_optimal_range = self._iter_optimal_range_intervals[idx]
        if self._iter_relax_factors_intervals is not None:
            self.iter_relax_factors = self._iter_relax_factors_intervals[idx]
        super()._adaptation_based_on_iterations(iterations)
        self.iter_max = old_im
        self.iter_optimal_range = old_ior
        self.iter_relax_factors = old_irf

    def _adaptation_based_on_recomputation(self) -> None:
        idx = self._get_interval_idx(is_recomputing=True)
        old_dm = self.dt_min_max
        old_rf = self.recomp_factor
        old_rm = self.recomp_max
        if self._dt_min_max_intervals is not None:
            self.dt_min_max = self._dt_min_max_intervals[idx]
        if self._recomp_factor_intervals is not None:
            self.recomp_factor = self._recomp_factor_intervals[idx]
        if self._recomp_max_intervals is not None:
            self.recomp_max = self._recomp_max_intervals[idx]
        super()._adaptation_based_on_recomputation()
        self.dt_min_max = old_dm
        self.recomp_factor = old_rf
        self.recomp_max = old_rm

    # ------------------------------------------------------------------
    # Schedule correction (robust _scheduled_idx re-sync)
    # ------------------------------------------------------------------

    def _correction_based_on_schedule(self) -> None:
        """Correct time step if time + dt would reach or pass the next scheduled time.

        The next schedule point is found via ``np.searchsorted`` rather than relying
        on the internal ``_scheduled_idx`` counter, so this is robust against the
        counter going stale (e.g. when dt_init lands exactly on a schedule point, or
        after a restart).  ``_scheduled_idx`` is re-synced here so that the recompute
        path's decrement in ``_adaptation_based_on_recomputation`` remains correct.
        """
        # Find the first schedule point strictly greater than the current time.
        idx = int(np.searchsorted(self.schedule, self.time, side="right"))
        idx = min(idx, len(self.schedule) - 1)
        self._scheduled_idx = idx  # re-sync so recompute decrements land correctly
        super()._correction_based_on_schedule()
