import re
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    import meshio
except ImportError as exc:
    raise ImportError("meshio is required for load_from_pvd") from exc
import numpy as np


def split_by_subdomain(
    dim_data: dict[str, np.ndarray],
) -> dict[int, dict[str, np.ndarray]]:
    """Split a single dimension-entry from :func:`load_from_pvd` by subdomain id.

    PorePy concatenates all subdomains of the same dimension into one VTU file,
    so a single entry in the ``subdomains`` dict returned by :func:`load_from_pvd`
    may contain cells from multiple subdomains interleaved.  This function uses the
    ``"subdomain_id"`` cell-data array — exported by PorePy as constant data and
    available as a normal quantity when passed to :func:`load_from_pvd` — to recover
    the per-subdomain cell partition.

    Parameters:
        dim_data: A single value from the ``subdomains`` (or ``interfaces``) dict
            returned by :func:`load_from_pvd`, i.e. a dict with ``"times"`` and one
            array per quantity, all of shape ``(n_timesteps, n_cells[, nd])``.  Must
            include the key ``"subdomain_id"`` (or ``"interface_id"`` for interface
            data); pass it as one of the ``quantities`` to :func:`load_from_pvd`.

    Raises:
        KeyError: If neither ``"subdomain_id"`` nor ``"interface_id"`` is present.

    Returns:
        Dict keyed by integer subdomain/interface id. Each value has the same
        structure as *dim_data* (``"times"`` plus one array per quantity), but
        restricted to the cells belonging to that subdomain.

    Example:
        >>> sds, _ = load_from_pvd(
        ...     "my_output_directory/",
        ...     ["displacement_jump", "subdomain_id"],
        ...     subdomain_dimensions=[2],
        ... )
        >>> per_sd = split_by_subdomain(sds[2])
        >>> jump_sd0 = per_sd[<id>]["displacement_jump"]  # shape (T, C_sd0, 3)
    """
    if "subdomain_id" in dim_data:
        id_key = "subdomain_id"
    elif "interface_id" in dim_data:
        id_key = "interface_id"
    else:
        raise KeyError(
            "dim_data must contain 'subdomain_id' or 'interface_id'. "
            "Pass it as one of the quantities to load_from_pvd."
        )

    # subdomain_id is constant in time; read from the first time step.
    ids: np.ndarray = dim_data[id_key][0]  # shape (n_cells,)
    unique_ids = np.unique(ids)

    result: dict[int, dict[str, np.ndarray]] = {}
    for uid in unique_ids:
        mask = ids == uid
        entry: dict[str, np.ndarray] = {"times": dim_data["times"]}
        for key, arr in dim_data.items():
            if key == "times":
                continue
            # arr has shape (T, n_cells) or (T, n_cells, nd)
            entry[key] = arr[:, mask]
        result[int(uid)] = entry

    return result


def load_from_pvd(
    run_dir: str | Path,
    quantities: list[str],
    pvd_name: str = "data",
    subdomain_dimensions: list[int] | None = None,
    interface_dimensions: list[int] | None = (),
) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, dict[str, np.ndarray]]]:
    """Load cell-data arrays from VTU files referenced by a PorePy PVD file.

    PorePy writes one VTU file per grid dimension per time step, following the
    naming conventions:

    - Subdomains: ``<pvd_name>_<dim>_<step>.vtu``
    - Interfaces (mortar grids): ``<pvd_name>_mortar_<dim>_<step>.vtu``

    Both dicts in the return value are therefore keyed by grid **dimension**
    (e.g. ``2`` for 2-D fractures, ``1`` for 1-D mortar grids).

    Parameters:
        run_dir: Directory containing the PVD file and the associated VTU
            files. Accepts both ``str`` and :class:`pathlib.Path`.
        quantities: Names of cell-data arrays to extract (e.g.
            ``["displacement_jump", "pressure"]``).
        pvd_name: Stem of the PVD index file (without the ``.pvd`` extension).
            Defaults to ``"data"``, matching PorePy's default output name.
        subdomain_dimensions: Dimensions of subdomain files to read. ``None``
            (default) reads all dimensions; an empty list skips all subdomains.
        interface_dimensions: Dimensions of interface (mortar) files to read.
            ``None`` reads all dimensions; an empty list (default) skips all
            interfaces, preserving the behaviour of earlier versions of this
            function.

    Raises:
        ImportError: If ``meshio`` is not installed.
        FileNotFoundError: If ``<pvd_name>.pvd`` is not found in ``run_dir``.

    Returns:
        A 2-tuple ``(subdomains, interfaces)`` where each element is a dict
        keyed by grid dimension. Each value is itself a dict with:

        - ``"times"``: :class:`numpy.ndarray` of shape ``(n_timesteps,)`` —
          simulation times in seconds.
        - One key per requested quantity that was found in the files:
          :class:`numpy.ndarray` of shape ``(n_timesteps, n_cells)`` for scalar
          fields or ``(n_timesteps, n_cells, nd)`` for vector fields.

        Subdomains or interfaces that contain none of the requested quantities
        are silently omitted from their respective dict.

    Example:
        >>> sds, ifs = load_from_pvd(
        ...     "my_output_directory/",
        ...     ["displacement_jump", "dilation_damage"],
        ...     subdomain_dimensions=[2],
        ...     interface_dimensions=[1],
        ... )
        >>> jump = sds[2]["displacement_jump"]  # 2-D fractures, shape (T, C, 3)
        >>> times = sds[2]["times"]             # shape (T,)
        >>> mortar_times = ifs[1]["times"]      # 1-D interface times
    """

    run_dir = Path(run_dir)
    pvd_path = run_dir / f"{pvd_name}.pvd"
    if not pvd_path.exists():
        raise FileNotFoundError(f"No {pvd_name}.pvd found in {run_dir}")

    sd_dims = None if subdomain_dimensions is None else set(subdomain_dimensions)
    if_dims = None if interface_dimensions is None else set(interface_dimensions)

    sd_pat = re.compile(rf"{re.escape(pvd_name)}_(\d+)_\d+\.vtu")
    if_pat = re.compile(rf"{re.escape(pvd_name)}_mortar_(\d+)_\d+\.vtu")

    # dim -> {"times": [...], qty: [...], ...}
    sd_raw: dict[int, dict[str, list]] = {}
    if_raw: dict[int, dict[str, list]] = {}

    tree = ET.parse(pvd_path)
    for ds in tree.getroot().find("Collection").findall("DataSet"):
        fname = ds.get("file", "")
        t = float(ds.get("timestep", 0))

        if m := sd_pat.fullmatch(fname):
            raw, dim_filter = sd_raw, sd_dims
        elif m := if_pat.fullmatch(fname):
            raw, dim_filter = if_raw, if_dims
        else:
            continue

        dim = int(m.group(1))
        if dim_filter is not None and dim not in dim_filter:
            continue

        vtu_path = run_dir / fname
        if not vtu_path.exists():
            continue

        mesh = meshio.read(vtu_path)
        found = {q: mesh.cell_data[q][0] for q in quantities if q in mesh.cell_data}
        if not found:
            continue  # This dimension has none of the requested quantities; skip it.

        if dim not in raw:
            raw[dim] = {"times": []}
        raw[dim]["times"].append(t)
        for q, arr in found.items():
            raw[dim].setdefault(q, []).append(arr)

    def _finalise(raw: dict[int, dict[str, list]]) -> dict[int, dict[str, np.ndarray]]:
        result: dict[int, dict[str, np.ndarray]] = {}
        for dim, entry in raw.items():
            times = np.array(entry["times"])

            # PorePy writes a VTU entry for every solver attempt, including
            # non-converged ones, before writing the eventually converged step
            # at the same (or a later) timestamp.  When the solver backtracks,
            # the next entry has a *smaller* timestamp than the previous one.
            # We keep only the *last* entry at each unique timestamp (= the
            # converged solution) and discard t=0 (initial condition set by
            # model initialisation, not a converged time step).
            n = len(times)
            keep = np.ones(n, dtype=bool)
            for i in range(n - 1):
                if times[i] >= times[i + 1]:
                    # This entry will be superseded — drop it.
                    keep[i] = False
            keep[times == 0.0] = False

            idx = np.where(keep)[0]
            out: dict[str, np.ndarray] = {"times": times[idx]}
            for q in quantities:
                if q in entry:
                    stacked = np.stack(entry[q], axis=0)  # (T, n_cells[, nd])
                    out[q] = stacked[idx]
            result[dim] = out
        return result

    return _finalise(sd_raw), _finalise(if_raw)
