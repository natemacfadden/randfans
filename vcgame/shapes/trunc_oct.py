# =============================================================================
#    Copyright (C) 2026  Nate MacFadden
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
# =============================================================================

"""
Generate integer vectors from the vertices of the truncated octahedron
(permutohedron).

The 24 vertices are all distinct points of the form (0, ±1, ±2)
and its permutations. The shape tiles R³ by translation, has 36
edges and 14 faces (8 regular hexagons + 6 squares), and gives a
richer fan structure than the cube.
"""

from __future__ import annotations

from itertools import permutations


def trunc_oct_vectors() -> list[list[int]]:
    """Return the 24 vertices of the truncated octahedron.

    Returns all distinct permutations of (0, ±1, ±2).

    Returns
    -------
    list[list[int]]
        A list of 24 integer 3-vectors.
    """
    # Generate all permutations of (0,1,2) with all sign combinations.
    # Applying signs to the zero component produces duplicates (e.g.
    # (-1)*0 == 0), so the set deduplicates down to exactly 24 points.
    pts: set[tuple[int, ...]] = set()
    for a, b, c in permutations([0, 1, 2]):
        for sb in (-1, 1):
            for sc in (-1, 1):
                pts.add((a, sb * b, sc * c))
                pts.add((-a, sb * b, sc * c))
    pts.discard((0, 0, 0))
    return [list(p) for p in sorted(pts)]
