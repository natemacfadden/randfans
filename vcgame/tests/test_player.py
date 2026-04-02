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

"""Tests for game/player.py"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from game.player import Player

if TYPE_CHECKING:
    from regfans import Fan, VectorConfiguration


def _unit(v: np.ndarray, tol: float = 1e-10) -> bool:
    return bool(abs(np.linalg.norm(v) - 1.0) < tol)


def _tangent(p: np.ndarray, h: np.ndarray, tol: float = 1e-10) -> bool:
    return bool(abs(np.dot(p, h)) < tol)


# ---------------------------------------------------------------
# Player.__init__
# ---------------------------------------------------------------

def test_init_position_normalized() -> None:
    p = Player([2.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    assert _unit(p.direction)


def test_init_heading_normalized() -> None:
    p = Player([1.0, 0.0, 0.0], [0.0, 3.0, 0.0])
    assert _unit(p.heading)


def test_init_heading_tangent() -> None:
    p = Player([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    assert _tangent(p.direction, p.heading)


def test_init_heading_gram_schmidt() -> None:
    # heading with component along position is projected out
    p = Player([1.0, 0.0, 0.0], [1.0, 1.0, 0.0])
    assert _tangent(p.direction, p.heading)
    assert _unit(p.heading)


def test_init_oblique_position() -> None:
    p = Player([1.0, 1.0, 1.0], [0.0, 0.0, 1.0])
    assert _unit(p.direction)
    assert _unit(p.heading)
    assert _tangent(p.direction, p.heading)


def test_init_zero_position_raises() -> None:
    with pytest.raises(ValueError):
        Player([0.0, 0.0, 0.0], [0.0, 1.0, 0.0])


def test_init_zero_heading_raises() -> None:
    with pytest.raises(ValueError):
        Player([1.0, 0.0, 0.0], [0.0, 0.0, 0.0])


def test_init_parallel_heading_raises() -> None:
    with pytest.raises(ValueError):
        Player([1.0, 0.0, 0.0], [2.0, 0.0, 0.0])


def test_init_wrong_position_shape_raises() -> None:
    with pytest.raises(ValueError):
        Player([1.0, 0.0], [0.0, 1.0, 0.0])


def test_init_wrong_heading_shape_raises() -> None:
    with pytest.raises(ValueError):
        Player([1.0, 0.0, 0.0], [0.0, 1.0])


def test_position_property_is_copy() -> None:
    p = Player([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    pos = p.position
    pos[0] = 99.0
    assert p.position[0] != 99.0


def test_heading_property_is_copy() -> None:
    p = Player([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    h = p.heading
    h[0] = 99.0
    assert p.heading[0] != 99.0


# ---------------------------------------------------------------
# Player.turn
# ---------------------------------------------------------------

def test_turn_heading_stays_unit() -> None:
    p = Player([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    p.turn(0.7)
    assert _unit(p.heading)


def test_turn_heading_stays_tangent() -> None:
    p = Player([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    p.turn(0.7)
    assert _tangent(p.direction, p.heading)


def test_turn_zero_is_identity() -> None:
    p = Player([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    h0 = p.heading.copy()
    p.turn(0.0)
    np.testing.assert_allclose(p.heading, h0, atol=1e-12)


def test_turn_pi_reverses_heading() -> None:
    p = Player([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    h0 = p.heading.copy()
    p.turn(np.pi)
    np.testing.assert_allclose(p.heading, -h0, atol=1e-10)


def test_turn_pi_half_is_perpendicular() -> None:
    p = Player([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    h0 = p.heading.copy()
    p.turn(np.pi / 2)
    assert abs(np.dot(p.heading, h0)) < 1e-10


def test_turn_does_not_move_position() -> None:
    p = Player([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    pos0 = p.position.copy()
    p.turn(1.2)
    np.testing.assert_allclose(p.position, pos0, atol=1e-12)


def test_turn_two_quarter_turns_equals_half_turn() -> None:
    p1 = Player([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    p2 = Player([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    p1.turn(np.pi / 2)
    p1.turn(np.pi / 2)
    p2.turn(np.pi)
    np.testing.assert_allclose(p1.heading, p2.heading, atol=1e-10)


def test_turn_negative_angle() -> None:
    p = Player([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    p.turn(np.pi / 4)
    p.turn(-np.pi / 4)
    h_expected = np.array([0.0, 1.0, 0.0])
    np.testing.assert_allclose(p.heading, h_expected, atol=1e-10)


# ---------------------------------------------------------------
# Player.move
# ---------------------------------------------------------------

def test_move_position_stays_unit() -> None:
    p = Player([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    p.move(0.2)
    assert _unit(p.direction)


def test_move_heading_stays_unit() -> None:
    p = Player([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    p.move(0.2)
    assert _unit(p.heading)


def test_move_heading_stays_tangent() -> None:
    p = Player([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    p.move(0.2)
    assert _tangent(p.direction, p.heading)


def test_move_zero_is_identity() -> None:
    p = Player([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    pos0, h0 = p.position.copy(), p.heading.copy()
    p.move(0.0)
    np.testing.assert_allclose(p.position, pos0, atol=1e-12)
    np.testing.assert_allclose(p.heading,  h0,   atol=1e-12)


def test_move_negative_reverses() -> None:
    p = Player([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    pos0 = p.position.copy()
    p.move(0.4)
    p.move(-0.4)
    np.testing.assert_allclose(p.position, pos0, atol=1e-10)


def test_move_no_fan_returns_none() -> None:
    p = Player([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    assert p.move(0.1) is None


def test_move_no_crossing_returns_none(fan3: Fan) -> None:
    p = Player([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    assert p.move(1e-6, fan3) is None


def test_move_invariants_after_many_steps() -> None:
    p = Player([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    for _ in range(30):
        p.move(0.1)
    assert _unit(p.direction)
    assert _unit(p.heading)
    assert _tangent(p.direction, p.heading)


def test_move_with_fan_preserves_invariants(fan3: Fan) -> None:
    p = Player([1.0, 0.2, 0.1], [0.0, 1.0, 0.0])
    for _ in range(30):
        p.move(0.05, fan3)
    assert _unit(p.direction)
    assert _unit(p.heading)
    assert _tangent(p.direction, p.heading)


def test_move_crossing_returns_sorted_pair(fan3: Fan) -> None:
    p = Player([1.0, 0.2, 0.1], [0.0, 1.0, 0.0])
    crossed = None
    for _ in range(200):
        result = p.move(0.05, fan3)
        if result is not None:
            crossed = result
            break
    if crossed is None:
        pytest.skip("no crossing encountered in traversal")
    assert isinstance(crossed, tuple)
    assert len(crossed) == 2
    assert crossed[0] < crossed[1]


# ---------------------------------------------------------------
# Player.current_cone
# ---------------------------------------------------------------

def test_current_cone_is_valid(fan3: Fan) -> None:
    p = Player([1.0, 0.2, 0.1], [0.0, 1.0, 0.0])
    assert p.current_cone(fan3) in fan3.cones()


def test_current_cone_contains_position(fan3: Fan) -> None:
    p = Player([1.0, 0.2, 0.1], [0.0, 1.0, 0.0])
    cone = p.current_cone(fan3)
    vecs = fan3.vectors(which=cone)
    alpha, _, _, _ = np.linalg.lstsq(
        vecs.T, p.direction, rcond=None,
    )
    assert np.all(alpha > -1e-10)


def test_current_cone_consistent_after_move(fan3: Fan) -> None:
    p = Player([1.0, 0.2, 0.1], [0.0, 1.0, 0.0])
    for _ in range(15):
        p.move(0.05)
        assert p.current_cone(fan3) in fan3.cones()


def test_current_cone_consistent_after_turn(fan3: Fan) -> None:
    p = Player([1.0, 0.2, 0.1], [0.0, 1.0, 0.0])
    for angle in (0.3, 0.7, 1.5, 2.8):
        p.turn(angle)
        assert p.current_cone(fan3) in fan3.cones()


@pytest.mark.parametrize("pos, head", [
    ([1.0,  0.2, 0.1], [0.0, 1.0, 0.0]),
    ([0.2,  1.0, 0.1], [1.0, 0.0, 0.0]),
    ([0.1,  0.2, 1.0], [1.0, 0.0, 0.0]),
    ([-1.0, 0.2, 0.1], [0.0, 1.0, 0.0]),
    ([0.2, -1.0, 0.1], [1.0, 0.0, 0.0]),
    ([0.1,  0.2,-1.0], [1.0, 0.0, 0.0]),
])
def test_current_cone_various_positions(fan3, pos, head) -> None:
    p = Player(pos, head)
    assert p.current_cone(fan3) in fan3.cones()


# ---------------------------------------------------------------
# Player.pointed_facet
# ---------------------------------------------------------------

def test_pointed_facet_is_none_or_sorted_pair(fan3: Fan) -> None:
    p = Player([1.0, 0.2, 0.1], [0.0, 1.0, 0.0])
    facet = p.pointed_facet(fan3)
    assert facet is None or (
        isinstance(facet, tuple)
        and len(facet) == 2
        and facet[0] < facet[1]
    )


def test_pointed_facet_in_current_cone(fan3: Fan) -> None:
    p = Player([1.0, 0.2, 0.1], [0.0, 1.0, 0.0])
    facet = p.pointed_facet(fan3)
    if facet is not None:
        cone = p.current_cone(fan3)
        assert set(facet).issubset(set(cone))


def test_pointed_facet_consistent_after_turn(fan3: Fan) -> None:
    p = Player([1.0, 0.2, 0.1], [0.0, 1.0, 0.0])
    for angle in (0.3, 0.9, 1.8, 2.5):
        p.turn(angle)
        facet = p.pointed_facet(fan3)
        assert facet is None or (
            isinstance(facet, tuple) and len(facet) == 2
        )


# ---------------------------------------------------------------
# Player.find_circuit_for_crossing
# ---------------------------------------------------------------

def test_find_circuit_same_cone_is_none(fan3: Fan) -> None:
    p    = Player([1.0, 0.2, 0.1], [0.0, 1.0, 0.0])
    cone = p.current_cone(fan3)
    assert p.find_circuit_for_crossing(cone, cone, fan3) is None


def test_find_circuit_adjacent_cones(
    fan3: Fan,
    adjacent_cones: tuple[tuple[int, ...], tuple[int, ...]],
) -> None:
    c1, c2 = adjacent_cones
    p    = Player([1.0, 0.2, 0.1], [0.0, 1.0, 0.0])
    circ = p.find_circuit_for_crossing(c1, c2, fan3)
    if circ is not None:
        assert set(circ.Z) == set(c1) | set(c2)


def test_find_circuit_support_union(
    fan3: Fan,
    adjacent_cones: tuple[tuple[int, ...], tuple[int, ...]],
) -> None:
    c1, c2 = adjacent_cones
    p    = Player([1.0, 0.2, 0.1], [0.0, 1.0, 0.0])
    circ = p.find_circuit_for_crossing(c1, c2, fan3)
    if circ is not None:
        assert len(set(circ.Z)) == 4


# ---------------------------------------------------------------
# Player.crossed_circuit
# ---------------------------------------------------------------

def test_crossed_circuit_returns_circuit_or_none(
    fan3: Fan,
    vc3: VectorConfiguration,
    adjacent_cones: tuple[tuple[int, ...], tuple[int, ...]],
) -> None:
    c1, c2 = adjacent_cones
    p    = Player([1.0, 0.2, 0.1], [0.0, 1.0, 0.0])
    circ = p.crossed_circuit(c1, c2, vc3)
    assert circ is None or hasattr(circ, "Z")


def test_crossed_circuit_support(
    fan3: Fan,
    vc3: VectorConfiguration,
    adjacent_cones: tuple[tuple[int, ...], tuple[int, ...]],
) -> None:
    c1, c2 = adjacent_cones
    p    = Player([1.0, 0.2, 0.1], [0.0, 1.0, 0.0])
    circ = p.crossed_circuit(c1, c2, vc3)
    if circ is not None:
        assert set(circ.Z) == set(c1) | set(c2)


# ---------------------------------------------------------------
# Player.__repr__
# ---------------------------------------------------------------

def test_repr_is_string() -> None:
    p = Player([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    assert isinstance(repr(p), str)


def test_repr_contains_player() -> None:
    p = Player([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    assert "Player" in repr(p)


def test_repr_contains_coordinates() -> None:
    p = Player([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    s = repr(p)
    assert "1.0" in s or "1.0000" in s
