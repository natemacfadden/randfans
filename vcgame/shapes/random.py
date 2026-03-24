"""
Generate integer vectors from the surface of the convex hull of a set
of random lattice points whose convex hull strictly contains the origin.

If a sampled set does not strictly contain the origin (i.e. the positive
span is not all of R³), the set is discarded and resampled.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError

_LATTICE_TOL = 1e-6


def _surface_lattice_points(
    pts: list[list[int]],
    hull: ConvexHull,
) -> list[list[int]]:
    """Return every non-origin integer point on the surface of ``hull``.

    For each facet plane, enumerate candidate integer points and keep those
    that satisfy all hull inequalities.

    Parameters
    ----------
    pts : list[list[int]]
        Integer points used to build the hull.
    hull : scipy.spatial.ConvexHull
        Convex hull of ``pts``.

    Returns
    -------
    list[list[int]]
        Non-origin integer 3-vectors lying on the hull surface.
    """
    arr = np.array(pts, dtype=float)
    lo  = np.floor(arr.min(axis=0)).astype(int)
    hi  = np.ceil (arr.max(axis=0)).astype(int)

    result: set[tuple[int, ...]] = set()
    eqs = hull.equations  # (nfacets, 4): n·x + d ≤ 0 for interior/surface

    for row in eqs:
        n, d = row[:3], float(row[3])
        # Pivot on the axis with the largest |coefficient| for accuracy.
        pivot = int(np.argmax(np.abs(n)))
        o0, o1 = [i for i in range(3) if i != pivot]

        if abs(n[pivot]) < 1e-9:
            continue

        for v0 in range(lo[o0], hi[o0] + 1):
            for v1 in range(lo[o1], hi[o1] + 1):
                pv  = (-d - n[o0] * v0 - n[o1] * v1) / n[pivot]
                pvi = round(pv)
                if abs(pv - pvi) > _LATTICE_TOL:
                    continue
                if not (lo[pivot] <= pvi <= hi[pivot]):
                    continue
                p        = [0, 0, 0]
                p[o0]    = v0
                p[o1]    = v1
                p[pivot] = pvi
                p_arr    = np.array(p, dtype=float)
                # Keep only if strictly inside or on all half-spaces.
                if np.all(eqs @ np.append(p_arr, 1.0) <= _LATTICE_TOL):
                    pt = tuple(p)
                    if pt != (0, 0, 0):
                        result.add(pt)

    return [list(p) for p in sorted(result)]


def random_vectors(
    seed:      int = 1102,
    n_vectors: int = 12,
    max_coord: int = 3,
) -> list[list[int]]:
    """Sample random integer vectors whose convex hull strictly contains origin.

    Samples ``n_vectors`` distinct non-zero integer vectors from
    [−max_coord, max_coord]³, checks that the origin lies in the strict
    interior of their convex hull (equivalently, the positive span is R³),
    and retries if not. Returns all non-origin lattice points on the hull
    surface.

    Parameters
    ----------
    seed : int, optional
        RNG seed. Defaults to 1102.
    n_vectors : int, optional
        Number of seed vectors. Must be >= 4. Defaults to 12.
    max_coord : int, optional
        Coordinate range (inclusive). Must be >= 1.

    Returns
    -------
    list[list[int]]
        A list of integer 3-vectors.

    Raises
    ------
    ValueError
        If ``n_vectors`` < 4 or ``max_coord`` < 1.
    """
    if n_vectors < 4:
        raise ValueError(f"n_vectors must be >= 4, got {n_vectors}")
    if max_coord < 1:
        raise ValueError(f"max_coord must be >= 1, got {max_coord}")

    rng = np.random.default_rng(seed)

    _MAX_ATTEMPTS = 10_000
    for _attempt in range(_MAX_ATTEMPTS):
        seen: set[tuple[int, ...]] = set()
        while len(seen) < n_vectors:
            v  = rng.integers(-max_coord, max_coord + 1, size=3)
            vt = tuple(v.tolist())
            if vt == (0, 0, 0) or vt in seen:
                continue
            seen.add(vt)

        pts = [list(p) for p in seen]

        try:
            hull = ConvexHull(pts)
        except QhullError:
            continue

        # Origin is strictly interior iff all half-space offsets are negative.
        if np.all(hull.equations[:, 3] < 0):
            return _surface_lattice_points(pts, hull)

    raise ValueError(
        f"random_vectors: origin not strictly interior after {_MAX_ATTEMPTS} attempts "
        f"(seed={seed}, n_vectors={n_vectors}, max_coord={max_coord})"
    )
