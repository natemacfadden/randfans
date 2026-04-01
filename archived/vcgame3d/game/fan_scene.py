"""
Build a renderable scene from a regfans Fan.

The pipeline:
    Fan (integer ray vectors + cone structure)
    → normalize rays to S³
    → subdivide each edge via slerp
    → project each point S³ → R³
    → return (pts, edges, styles) for the renderer

Public API
----------
fan_to_scene(fan, project, n_subdivisions=4)
    Main entry point.  Returns (pts, edges, styles) compatible with
    vcgame3d.renderer.renderer.draw().

_crosspolytope_fan()
    Convenience: returns a Fan over the 4D cross-polytope for testing
    without CYTools.
"""
from __future__ import annotations

from itertools import combinations
from typing import Callable

import numpy as np

from ..renderer.projection import normalize, edge_points


def fan_to_scene(
    fan,
    project: Callable[[np.ndarray], np.ndarray],
    n_subdivisions: int = 4,
) -> tuple[list[np.ndarray], list[tuple[int, int]], list[str]]:
    """Convert a regfans Fan to a renderer-compatible scene.

    Parameters
    ----------
    fan : regfans.Fan
        The fan to render.  Rays are taken from fan.vectors() in the
        order given by fan.labels.
    project : callable
        A function S³ → R³.  Build one with
        ``vcgame3d.renderer.projection.stereographic_proj()``, or supply
        any callable with the same signature.
    n_subdivisions : int, optional
        Number of slerp steps per fan edge.  1 gives straight projected
        lines; ≥4 gives visibly curved great-circle arcs.  Default 4.

    Returns
    -------
    pts : list of np.ndarray, shape (3,)
        Projected 3D positions.
    edges : list of (int, int)
        Index pairs into pts.
    styles : list of str
        One style tag per edge (all "simplex").
    """
    # ── rays ─────────────────────────────────────────────────────────
    labels   = fan.labels                          # tuple of ints
    raw_vecs = np.array(fan.vectors(), dtype=float)  # shape (n, 4)

    sphere_vecs = np.array([normalize(v) for v in raw_vecs])  # shape (n, 4)
    label_to_sphere = {l: sphere_vecs[i] for i, l in enumerate(labels)}

    # ── unique edges from cone structure ─────────────────────────────
    edge_set: set[tuple[int, int]] = set()
    for cone in fan.cones():
        for a, b in combinations(cone, 2):
            edge_set.add((min(a, b), max(a, b)))

    # ── project and subdivide ─────────────────────────────────────────
    pts:               list[np.ndarray]        = []
    edges:             list[tuple[int, int]]   = []
    styles:            list[str]               = []
    arc_pts:           dict[tuple, list[int]]  = {}   # (la,lb) → ordered pts indices
    edge_label_per_edge: list[tuple[int, int]] = []   # one per entry in edges

    for (la, lb) in edge_set:
        pa = label_to_sphere[la]
        pb = label_to_sphere[lb]

        arc = [np.array(project(p)) for p in edge_points(pa, pb, n_subdivisions)]

        base = len(pts)
        pts.extend(arc)
        arc_pts[(la, lb)] = list(range(base, base + n_subdivisions + 1))
        for i in range(n_subdivisions):
            edges.append((base + i, base + i + 1))
            styles.append("simplex")
            edge_label_per_edge.append((la, lb))

    return pts, edges, styles, arc_pts, edge_label_per_edge


# ── pole helpers ─────────────────────────────────────────────────────

def auto_pole(fan) -> np.ndarray:
    """Compute the projection pole as the antipode of the mean ray direction.

    Placing the stereographic pole opposite the centre of mass of the rays
    keeps the projected image roughly centred in R³ and avoids the
    distortion spike that occurs when a ray lands near the default (0,0,0,1)
    pole.

    Parameters
    ----------
    fan : regfans Fan
        Fan whose rays will be projected.

    Returns
    -------
    np.ndarray, shape (4,)
        Unit vector on S³ to use as the projection pole.
    """
    raw_vecs = np.array(fan.vectors(), dtype=float)
    sphere_vecs = np.array([normalize(v) for v in raw_vecs])
    mean = sphere_vecs.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm < 1e-10:
        return np.array([0.0, 0.0, 0.0, 1.0])   # degenerate: fall back to default
    return -(mean / norm)   # antipode


# ── per-vertex helpers ────────────────────────────────────────────────

def fan_vertices(fan, project: Callable) -> dict:
    """Return the projected R³ position for each ray of the fan.

    Parameters
    ----------
    fan : regfans Fan
    project : callable
        The same S³ → R³ callable used to build the scene.

    Returns
    -------
    dict mapping int label → np.ndarray shape (3,)
    """
    labels     = fan.labels
    raw_vecs   = np.array(fan.vectors(), dtype=float)
    sphere_vecs = [normalize(v) for v in raw_vecs]
    return {l: np.array(project(sphere_vecs[i])) for i, l in enumerate(labels)}


def make_cone_finder(fan):
    """Return a callable that maps an S³ point → labels of the containing cone.

    The returned function takes a unit 4-vector on S³ and returns the list of
    ray labels whose positive span contains that point, or None if no maximal
    cone of the fan contains it (e.g. the point is near the projection pole).

    Parameters
    ----------
    fan : regfans Fan

    Returns
    -------
    find_cone : callable (np.ndarray shape (4,)) → list[int] | None
    """
    labels       = fan.labels
    raw_vecs     = np.array(fan.vectors(), dtype=float)
    sphere_vecs  = [normalize(v) for v in raw_vecs]
    label_to_sv  = {l: sphere_vecs[i] for i, l in enumerate(labels)}

    # Pre-compute matrices for each maximal (4-ray) cone
    max_cones = [list(c) for c in fan.cones() if len(c) == 4]
    cone_mats = [
        (cone_labels, np.column_stack([label_to_sv[l] for l in cone_labels]))
        for cone_labels in max_cones
    ]

    def find_cone(p4d: np.ndarray):
        """Return ray labels of the cone containing p4d, or None."""
        p4d = np.asarray(p4d, dtype=float)
        for cone_labels, M in cone_mats:
            try:
                coeffs = np.linalg.solve(M, p4d)
            except np.linalg.LinAlgError:
                continue
            if np.all(coeffs >= -1e-8):
                return cone_labels
        return None

    return find_cone


# ── built-in fans ─────────────────────────────────────────────────────

class _Simplex4dFan:
    """The standard 4-simplex fan in R⁴ — no CYTools or regfans required.

    Rays (integral, sum to zero):
        r1 = ( 1,  1,  1,  1)
        r2 = (-1,  0,  0,  0)
        r3 = ( 0, -1,  0,  0)
        r4 = ( 0,  0, -1,  0)
        r5 = ( 0,  0,  0, -1)

    Any 4 of the 5 rays form a basis for R⁴, so all C(5,4) = 5 four-element
    subsets are maximal cones and the fan is complete.

    preferred_pole is chosen so that |dot(pole, rᵢ_hat)| ≤ 0.5 for all i,
    avoiding the degenerate case where auto_pole would land on r1_hat.
    """
    labels = (1, 2, 3, 4, 5)
    preferred_pole = np.array([1., 1., -1., -1.]) / 2.0

    def vectors(self):
        return [
            ( 1,  1,  1,  1),
            (-1,  0,  0,  0),
            ( 0, -1,  0,  0),
            ( 0,  0, -1,  0),
            ( 0,  0,  0, -1),
        ]

    def cones(self):
        return [tuple(c) for c in combinations(self.labels, 4)]


def _crosspolytope_fan():
    """Return a Fan over the 4D cross-polytope for offline testing.

    The cross-polytope has 8 vertices: ±e₁, ±e₂, ±e₃, ±e₄.
    No CYTools required.
    """
    from regfans import VectorConfiguration

    vectors = [
        [ 1,  0,  0,  0], [-1,  0,  0,  0],
        [ 0,  1,  0,  0], [ 0, -1,  0,  0],
        [ 0,  0,  1,  0], [ 0,  0, -1,  0],
        [ 0,  0,  0,  1], [ 0,  0,  0, -1],
    ]
    return VectorConfiguration(vectors).triangulate()
