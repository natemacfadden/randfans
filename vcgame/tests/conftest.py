"""Shared fixtures for vcgame tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.generate_cube import cube_fan, cube_vc

if TYPE_CHECKING:
    from regfans import Fan, VectorConfiguration


@pytest.fixture(scope="session")
def fan3() -> Fan:
    return cube_fan(3)


@pytest.fixture(scope="session")
def vc3() -> VectorConfiguration:
    return cube_vc(3)


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
