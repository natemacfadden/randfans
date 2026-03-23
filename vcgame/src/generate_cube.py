"""
Generate a vector configuration and triangulation from the boundary lattice
points of an n x n x n integer cube.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from regfans import VectorConfiguration

if TYPE_CHECKING:
    from regfans import Fan


def cube_vectors(n: int) -> list[list[int]]:
    """
    **Description:**
    Return the boundary lattice points of an n x n x n integer cube, centered
    at the origin. Boundary points are those with at least one coordinate equal
    to ±(n-1)/2. The origin (interior center) is excluded.

    **Arguments:**
    - `n`: Grid size. Must be odd and >= 3.

    **Returns:**
    A list of integer 3-vectors.
    """
    if n % 2 == 0 or n < 3:
        raise ValueError(f"n must be odd and >= 3, got {n}")
    half = (n - 1) // 2
    return [
        [x - half, y - half, z - half]
        for x in range(n)
        for y in range(n)
        for z in range(n)
        if x == 0 or x == n-1 or y == 0 or y == n-1 or z == 0 or z == n-1
    ]


def cube_vc(n: int) -> VectorConfiguration:
    """
    **Description:**
    Return the VectorConfiguration of the n x n x n cube boundary
    lattice points.

    **Arguments:**
    - `n`: Grid size. Must be odd and >= 3.

    **Returns:**
    A VectorConfiguration.
    """
    return VectorConfiguration(cube_vectors(n))


def cube_fan(n: int) -> Fan:
    """
    **Description:**
    Return a triangulation of the n x n x n cube vector configuration.

    **Arguments:**
    - `n`: Grid size. Must be odd and >= 3.

    **Returns:**
    A Fan.
    """
    return VectorConfiguration(cube_vectors(n)).triangulate()
