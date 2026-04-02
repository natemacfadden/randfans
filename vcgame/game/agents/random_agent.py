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
Random agent: navigates the fan along a Lévy walk on S².

Each segment is a circular arc whose length and curvature are
drawn independently. Arc lengths follow a Pareto distribution
(power-law tail); curvatures follow a log-normal distribution
centred at κ ≈ 1, corresponding to a turn radius of ≈ 57°.
With log-std 0.7, 68 % of arcs have radii in [28°, 115°],
giving smooth medium-scale curves as the dominant behaviour
while still occasionally producing tight spirals or near-
straight sweeps.

The fractal character comes from the Pareto arc-length
distribution: as the exponent α → 1, arcs become arbitrarily
long and the path approaches a Lévy flight. For 1 < α < 2 the
walk is superdiffusive (mean displacement grows faster than
√t); for α > 2 it recovers ordinary diffusion at large scales.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..player import Player
    from regfans import Fan


_L_MIN        = 0.2   # minimum arc length (radians)
_LOG_K_MU     = 0.0   # log-curvature mean  → κ ≈ 1.0  (≈ 57° radius)
_LOG_K_SIG    = 0.7   # log-curvature std   → 68 % of radii in [28°, 115°]
_MAX_TURN_STEP = 0.08  # max heading rotation (rad) per advance call


class RandomAgent:
    """Navigates a player along Lévy-walk arcs on S².

    Arc lengths are Pareto-distributed; curvatures are log-normal, centred
    at a medium turn radius (≈ 57°) so the path is dominated by smooth arcs
    with occasional tight or sweeping excursions.

    Parameters
    ----------
    player : Player
        The ``Player`` to control.
    alpha : float, optional
        Pareto exponent for arc lengths. Must be > 1. Values near 1 give
        long, wandering arcs; larger values give shorter, more uniform
        arcs.
    step : float, optional
        Arc length advanced per call to ``advance`` (radians). Should
        match the game's movement step.

    Attributes
    ----------
    player : Player
        The ``Player`` being controlled (read-only).
    alpha : float
        Pareto exponent for arc-length distribution.
    step : float
        Arc length advanced per call to ``advance``.

    Raises
    ------
    ValueError
        If ``alpha <= 1``.
    """

    def __init__(
        self,
        player: Player,
        alpha: float = 1.5,
        step:  float = 0.04,
    ) -> None:
        if alpha <= 1.0:
            raise ValueError(f"alpha must be > 1, got {alpha}")
        self._player        = player
        self._alpha         = alpha
        self._step          = step
        self._arc_remaining = 0.0
        self._kappa         = 0.0
        self._new_arc()

    @property
    def player(self) -> Player:
        """The player being controlled."""
        return self._player

    @property
    def alpha(self) -> float:
        """Pareto exponent for arc-length distribution."""
        return self._alpha

    @property
    def step(self) -> float:
        """Arc length advanced per call to ``advance``."""
        return self._step

    @step.setter
    def step(self, value: float) -> None:
        self._step = float(value)

    def _new_arc(self) -> None:
        """Sample a new arc segment.

        Arc length is Pareto-distributed; curvature magnitude is log-normal
        with random sign.
        """
        self._arc_remaining = _L_MIN * (
            np.random.pareto(self._alpha - 1) + 1.0
        )
        log_kappa   = np.random.normal(_LOG_K_MU, _LOG_K_SIG)
        sign        = float(np.random.choice([-1.0, 1.0]))
        self._kappa = sign * np.exp(log_kappa)
        # Cap arc to avoid long spirals (≤ ~216° of heading rotation).
        self._arc_remaining = min(
            self._arc_remaining, 1.2 * np.pi / abs(self._kappa)
        )

    def advance(self, fan: Fan | None = None) -> None:
        """Advance the player by one step along the current arc.

        When the arc is exhausted a new one is sampled.

        Parameters
        ----------
        fan : regfans.Fan or None, optional
            Passed to ``Player.move`` for cone-crossing detection; does
            not trigger flips.
        """
        if self._arc_remaining <= 0.0:
            self._new_arc()
        # Limit arc per step by curvature so tight spirals take more steps.
        s = min(self._step, self._arc_remaining,
                _MAX_TURN_STEP / abs(self._kappa))
        self._player.turn(self._kappa * s)
        self._player.move(s, fan)
        self._arc_remaining -= s

    def __repr__(self) -> str:
        return (
            f"RandomAgent(alpha={self._alpha}, step={self._step}, "
            f"arc_remaining={self._arc_remaining:.3f}, "
            f"kappa={self._kappa:.3f})"
        )
