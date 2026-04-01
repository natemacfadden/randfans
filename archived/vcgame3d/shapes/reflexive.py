"""
Generate integer vectors from the lattice points of a 4D reflexive polytope.

Data source: Kreuzer-Skarke database, fetched via CYTools.

Each polytope's lattice points are returned as 4-vectors (rows), with the
origin excluded.
"""
from __future__ import annotations

from cytools import fetch_polytopes


class ReflexiveFetchError(OSError):
    """Raised when the polytope database cannot be reached."""


def reflexive_vectors(h11: int, polytope_id: int = 0) -> list[list[int]]:
    """Return non-origin lattice points of a 4D reflexive polytope.

    Fetches data from the Kreuzer-Skarke database via CYTools.

    Parameters
    ----------
    h11 : int
        The Hodge number h^{1,1} of the associated Calabi-Yau hypersurface.
    polytope_id : int, optional
        Index into the list of polytopes with the given h11. Defaults to 0.

    Returns
    -------
    list[list[int]]
        Non-origin integer lattice points as 4-vectors (row format).

    Raises
    ------
    ValueError
        If polytope_id is out of range for the given h11.
    ReflexiveFetchError
        If the database cannot be reached.
    """
    try:
        polys = fetch_polytopes(h11=h11, limit=polytope_id + 1, dim=4)
    except Exception as exc:
        raise ReflexiveFetchError(
            f"Could not fetch polytopes (h11={h11}): {exc}"
        ) from exc

    if polytope_id >= len(polys):
        raise ValueError(
            f"polytope_id {polytope_id} out of range: only {len(polys)} "
            f"polytopes found for h11={h11}"
        )

    # points_not_interior_to_facets: vertices + lower-dim face points, origin-first
    # [1:] trims the origin
    pts = polys[polytope_id].points_not_interior_to_facets()[1:]
    return [[int(x) for x in pt] for pt in pts]
