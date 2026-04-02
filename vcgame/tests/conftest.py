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

"""Shared fixtures for vcgame tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from regfans import VectorConfiguration
from shapes import vectors_to_fan
from shapes.cube import cube_vectors

if TYPE_CHECKING:
    from regfans import Fan


@pytest.fixture(scope="session")
def fan3() -> Fan:
    return vectors_to_fan(cube_vectors(3))


@pytest.fixture(scope="session")
def vc3() -> VectorConfiguration:
    return VectorConfiguration(cube_vectors(3))


@pytest.fixture(scope="session")
def adjacent_cones(
    fan3: Fan,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    cones = fan3.cones()
    for i, c1 in enumerate(cones):
        for j, c2 in enumerate(cones):
            if i != j and len(set(c1) & set(c2)) == 2:
                return c1, c2
    pytest.skip("no adjacent cones found in fan")
