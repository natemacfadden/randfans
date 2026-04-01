"""
4D → 3D projection pipeline.

Coordinate flow:
    integer lattice vectors (Z⁴)
    → normalize to S³          via normalize()
    → project S³ → R³          via a projection callable

Available projections
---------------------
hyperspherical_proj()        -- drop r from 4D hyperspherical coordinates (default)
stereographic_proj(pole)     -- classical stereographic from a pole point
"""
from __future__ import annotations

import numpy as np


# ── low-level geometry ────────────────────────────────────────────────

def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a non-zero vector to the unit 3-sphere S³.

    Parameters
    ----------
    v : array-like, shape (4,)

    Returns
    -------
    np.ndarray, shape (4,)
        Unit vector on S³.
    """
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-14:
        raise ValueError(f"Cannot normalize the zero vector: {v}")
    return v / n


def slerp(p0: np.ndarray, p1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two points on S³.

    Parameters
    ----------
    p0, p1 : np.ndarray, shape (4,)
        Unit vectors on S³.
    t : float
        Interpolation parameter. t=0 → p0, t=1 → p1.

    Returns
    -------
    np.ndarray, shape (4,)
        Unit vector on S³ at parameter t.

    Notes
    -----
    Falls back to normalised lerp when p0 ≈ ±p1 (collinear case).
    """
    dot = float(np.clip(np.dot(p0, p1), -1.0, 1.0))
    if abs(dot) > 1.0 - 1e-10:
        # Nearly (anti)parallel — normalised linear fallback
        return normalize((1.0 - t) * p0 + t * p1)
    theta = np.arccos(dot)
    s = np.sin(theta)
    return (np.sin((1.0 - t) * theta) * p0 + np.sin(t * theta) * p1) / s


def edge_points(p0: np.ndarray, p1: np.ndarray, n: int) -> list[np.ndarray]:
    """Sample n+1 points along the great-circle arc from p0 to p1 on S³.

    Parameters
    ----------
    p0, p1 : np.ndarray, shape (4,)
        Unit vectors on S³ (endpoints).
    n : int
        Number of segments (n+1 points, including both endpoints).

    Returns
    -------
    list of np.ndarray, shape (4,), length n+1
    """
    return [slerp(p0, p1, i / n) for i in range(n + 1)]


# ── projection factories ──────────────────────────────────────────────

def _make_stereo_basis(pole: np.ndarray) -> np.ndarray:
    """Compute the orthonormal basis for the equatorial hyperplane orthogonal to pole."""
    basis: list[np.ndarray] = []
    for e in np.eye(4):
        v = e - np.dot(e, pole) * pole
        for b in basis:
            v -= np.dot(v, b) * b
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            basis.append(v / norm)
        if len(basis) == 3:
            break
    return np.array(basis)   # shape (3, 4)

def stereographic_proj(
    pole: np.ndarray | None = None,
) -> "Callable[[np.ndarray], np.ndarray]":
    """Return a stereographic projection S³ → R³ from the given pole.

    Projects from `pole` onto the equatorial hyperplane {x : x·pole = 0},
    returning 3D coordinates in an orthonormal basis for that hyperplane.

    Parameters
    ----------
    pole : array-like, shape (4,), optional
        A non-zero vector indicating the projection pole on S³.
        Defaults to (0, 0, 0, 1).

    Returns
    -------
    project : callable
        project(p) maps a unit 4-vector on S³ to a 3-vector in R³.
        Points near the pole map far from the R³ origin.

    Notes
    -----
    To change the effective pole at runtime, call this factory again and
    pass the new callable to fan_to_scene.
    """
    if pole is None:
        _pole = np.array([0.0, 0.0, 0.0, 1.0])
    else:
        _pole = normalize(np.asarray(pole, dtype=float))

    _basis = _make_stereo_basis(_pole)

    def project(p: np.ndarray) -> np.ndarray:
        """Map a unit 4-vector on S³ to R³ via stereographic projection."""
        p = np.asarray(p, dtype=float)
        dot = float(np.dot(p, _pole))
        denom = 1.0 - dot
        if abs(denom) < 1e-10:
            denom = np.sign(denom) * 1e-10 or 1e-10
        q = (p - dot * _pole) / denom   # 4D vector in the equatorial plane
        return _basis @ q               # 3D coordinates in the basis

    return project


def inverse_stereographic_proj(
    pole: np.ndarray | None = None,
) -> "Callable[[np.ndarray], np.ndarray]":
    """Return the inverse of stereographic_proj: R³ → S³.

    Maps a 3-vector back to the unit 4-vector on S³ it came from.
    Use the same pole as the corresponding stereographic_proj call.
    """
    if pole is None:
        _pole = np.array([0.0, 0.0, 0.0, 1.0])
    else:
        _pole = normalize(np.asarray(pole, dtype=float))

    _basis = _make_stereo_basis(_pole)   # shape (3, 4)

    def inverse(x: np.ndarray) -> np.ndarray:
        """Map a 3-vector in R³ back to a unit 4-vector on S³."""
        x    = np.asarray(x, dtype=float)
        v4d  = _basis.T @ x          # 4D vector in the equatorial hyperplane
        r2   = float(np.dot(x, x))
        d    = (r2 - 1.0) / (r2 + 1.0)
        p4d  = (1.0 - d) * v4d + d * _pole
        norm = np.linalg.norm(p4d)
        if norm < 1e-14:
            return _pole.copy()
        return p4d / norm

    return inverse


def hyperspherical_proj() -> "Callable[[np.ndarray], np.ndarray]":
    """Project S³ → R³ via hyperspherical coordinates (drop r = 1).

    For (x₁, x₂, x₃, x₄) ∈ S³ returns (χ, ψ, φ) where:

        χ = arctan2(√(x₂²+x₃²+x₄²), x₁)  ∈ [0, π]
        ψ = arctan2(√(x₃²+x₄²),      x₂)  ∈ [0, π]
        φ = arctan2(x₄,               x₃)  ∈ (−π, π]

    Unlike stereographic projection there is no pole at infinity — the
    map is defined (up to a measure-zero singularity set) everywhere on S³.
    Singularities (zero denominators) return 0 for the undefined angle.
    """
    def project(p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        x1, x2, x3, x4 = p
        r1  = np.sqrt(x2*x2 + x3*x3 + x4*x4)
        chi = np.arctan2(r1, x1)
        r2  = np.sqrt(x3*x3 + x4*x4)
        psi = np.arctan2(r2, x2) if r1 > 1e-10 else 0.0
        phi = np.arctan2(x4, x3) if r2 > 1e-10 else 0.0
        return np.array([chi, psi, phi])

    return project


def inverse_hyperspherical_proj() -> "Callable[[np.ndarray], np.ndarray]":
    """Exact inverse of hyperspherical_proj: R³ → S³.

    Given (χ, ψ, φ) returns the unit 4-vector:

        (cos χ,  sin χ cos ψ,  sin χ sin ψ cos φ,  sin χ sin ψ sin φ)
    """
    def inverse(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        chi, psi, phi = x
        s1 = np.sin(chi)
        s2 = np.sin(psi)
        p4d = np.array([
            np.cos(chi),
            s1 * np.cos(psi),
            s1 * s2 * np.cos(phi),
            s1 * s2 * np.sin(phi),
        ])
        norm = np.linalg.norm(p4d)
        return p4d / norm if norm > 1e-14 else p4d

    return inverse
