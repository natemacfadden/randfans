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
Generate integer vectors from the lattice points of a 3D reflexive polytope.

Data source: http://coates.ma.ic.ac.uk/3DReflexivePolytopes/
There are 4319 polytopes, indexed 0–4318.

Each polytope page contains a "Integer points" table cell with a
3 × K matrix (3 coordinate rows, K lattice-point columns).  The
last column is always the origin, which is excluded.  All other
columns are used as vectors.
"""

from __future__ import annotations

import re
from urllib.error import URLError
from urllib.request import urlopen

_BASE_URL   = "http://coates.ma.ic.ac.uk/3DReflexivePolytopes/{}.html"
N_POLYTOPES = 4319   # polytope ids 0 … 4318


class ReflexiveFetchError(OSError):
    """Raised when the polytope database cannot be reached."""


def reflexive_vectors(polytope_id: int = 0) -> list[list[int]]:
    """Return non-origin lattice points of a 3D reflexive polytope.

    Fetches data from the Coates–Corti–Galkin–Golyshev–Kasprzyk database.

    Parameters
    ----------
    polytope_id : int, optional
        Polytope index in [0, 4318]. Defaults to 0.

    Returns
    -------
    list[list[int]]
        Non-origin integer lattice points.

    Raises
    ------
    ValueError
        If ``polytope_id`` is out of range or the page cannot be parsed.
    ReflexiveFetchError
        If the database cannot be reached (network error).
    """
    if not 0 <= polytope_id < N_POLYTOPES:
        raise ValueError(
            f"polytope_id must be in [0, {N_POLYTOPES - 1}], got {polytope_id}"
        )

    url = _BASE_URL.format(polytope_id)
    try:
        with urlopen(url, timeout=15) as resp:
            html = resp.read().decode("utf-8")
    except URLError as exc:
        raise ReflexiveFetchError(
            f"Could not reach reflexive polytope database "
            f"(polytope_id={polytope_id}): {exc}"
        ) from exc

    match = re.search(
        r"Integer\s+points.*?<td[^>]*>(.*?)</td>",
        html,
        re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        raise ValueError(
            f"Could not find 'Integer points' data for polytope {polytope_id}"
        )

    cell = match.group(1)
    rows_html = re.split(r"<br\s*/?>", cell, flags=re.IGNORECASE)

    rows: list[list[int]] = []
    for rh in rows_html:
        text = re.sub(r"<[^>]+>", "", rh)
        text = text.replace("[", "").replace("]", "").strip()
        if not text:
            continue
        nums = [int(x) for x in text.split()]
        if nums:
            rows.append(nums)

    if len(rows) != 3:
        raise ValueError(
            f"Expected 3 coordinate rows, got {len(rows)} "
            f"for polytope {polytope_id}"
        )
    if len(set(len(r) for r in rows)) != 1:
        raise ValueError(
            f"Rows have unequal length for polytope {polytope_id}"
        )

    # Transpose: rows are coordinates (x/y/z), columns are points.
    k = len(rows[0])
    vectors = []
    for j in range(k):
        pt = [rows[0][j], rows[1][j], rows[2][j]]
        if any(x != 0 for x in pt):   # exclude origin
            vectors.append(pt)

    return vectors
