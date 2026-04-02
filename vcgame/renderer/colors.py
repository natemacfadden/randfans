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

"""Curses colour initialisation for the ASCII renderer."""

from __future__ import annotations

import curses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .renderer import Renderer


_RADIUS_PAIR_START = 6
_EDGE_PAIR_BASE    = 40   # pairs 40-43: front-flip, front-noflip, other-flip, other-noflip
_IREG_BG_PAIR      = 50   # pair for irregular-fan background tint
_FILL_PAIR         = 51   # dim fill for visible surface patches

# (r, g, b) in 0–1000 range for curses
_VIRIDIS_KEYS: list[tuple[int, int, int]] = [
    (267,   4, 329),
    (231, 322, 545),
    (129, 569, 549),
    (173, 694, 478),
    (369, 788, 384),
    (675, 863, 204),
    (867, 902, 114),
    (992, 906, 145),
]


def _viridis_rgb(t: float) -> tuple[int, int, int]:
    """Linearly interpolate between the ``_VIRIDIS_KEYS`` control points.

    Parameters
    ----------
    t : float
        Colormap position in [0, 1], where 0 maps to the first key
        (dark purple) and 1 maps to the last key (yellow). Clamped if
        out of range.

    Returns
    -------
    tuple[int, int, int]
        ``(r, g, b)`` in the 0–1000 range used by curses ``init_color``.
    """
    t  = max(0.0, min(1.0, t))
    s  = t * (len(_VIRIDIS_KEYS) - 1)
    lo = int(s)
    hi = min(lo + 1, len(_VIRIDIS_KEYS) - 1)
    f  = s - lo
    r0, g0, b0 = _VIRIDIS_KEYS[lo]
    r1, g1, b1 = _VIRIDIS_KEYS[hi]
    return int(r0 + f*(r1-r0)), int(g0 + f*(g1-g0)), int(b0 + f*(b1-b0))


def _init_colors(renderer: "Renderer") -> None:
    """Initialise all curses color pairs used by the renderer.

    Sets up Viridis gradient pairs for radius coloring, edge-flip indicator
    pairs (green/red), an irregular-fan background pair, and a dim fill pair.
    Falls back to basic terminal colors if the terminal does not support
    color redefinition.

    Parameters
    ----------
    renderer : Renderer
        The renderer instance; ``renderer._n_radius`` is set to the number
        of Viridis gradient steps successfully initialised.
    """
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN,    -1)
    curses.init_pair(2, curses.COLOR_YELLOW,  -1)
    curses.init_pair(3, curses.COLOR_GREEN,   -1)
    curses.init_pair(4, curses.COLOR_WHITE,   -1)
    curses.init_pair(5, curses.COLOR_WHITE,   -1)
    if curses.can_change_color() and curses.COLORS >= 32:
        n_avail = curses.COLOR_PAIRS - _RADIUS_PAIR_START
        n = max(2, min(32, curses.COLORS - 16, n_avail))
        for i in range(n):
            r, g, b = _viridis_rgb(i / (n - 1))
            curses.init_color(16 + i, r, g, b)
            curses.init_pair(_RADIUS_PAIR_START + i, 16 + i, -1)
        renderer._n_radius = n
        # Edge-flip indicator colours (dark green / dark red variants).
        _cb = 16 + n   # first free custom colour slot
        if curses.COLORS > _cb + 3 and curses.COLOR_PAIRS > _EDGE_PAIR_BASE + 3:
            curses.init_color(_cb + 0,   0, 800,   0)  # bright green  (front, flip)
            curses.init_color(_cb + 1, 800,   0,   0)  # bright red    (front, no-flip)
            curses.init_color(_cb + 2,   0, 500,   0)  # medium green  (other, flip)
            curses.init_color(_cb + 3, 500,   0,   0)  # medium red    (other, no-flip)
            for j in range(4):
                curses.init_pair(_EDGE_PAIR_BASE + j, _cb + j, -1)
        else:
            curses.init_pair(_EDGE_PAIR_BASE + 0, curses.COLOR_GREEN, -1)
            curses.init_pair(_EDGE_PAIR_BASE + 1, curses.COLOR_RED,   -1)
            curses.init_pair(_EDGE_PAIR_BASE + 2, curses.COLOR_GREEN, -1)
            curses.init_pair(_EDGE_PAIR_BASE + 3, curses.COLOR_RED,   -1)
        # Irregular-fan background: dark red bg, default fg.
        if curses.COLORS > _cb + 4 and curses.COLOR_PAIRS > _IREG_BG_PAIR:
            curses.init_color(_cb + 4, 280, 0, 0)
            curses.init_pair(_IREG_BG_PAIR, -1, _cb + 4)
        else:
            curses.init_pair(_IREG_BG_PAIR, -1, curses.COLOR_RED)
        # Dim fill for visible surface patches (foreground colour so block
        # chars are visible without needing a background colour).
        if curses.COLORS > _cb + 5 and curses.COLOR_PAIRS > _FILL_PAIR:
            curses.init_color(_cb + 5, 250, 280, 480)  # dim blue-gray
            curses.init_pair(_FILL_PAIR, _cb + 5, -1)
        else:
            curses.init_pair(_FILL_PAIR, curses.COLOR_BLUE, -1)
    else:
        for i, fg in enumerate([curses.COLOR_BLUE, curses.COLOR_CYAN,
                                 curses.COLOR_GREEN, curses.COLOR_YELLOW,
                                 curses.COLOR_RED]):
            curses.init_pair(_RADIUS_PAIR_START + i, fg, -1)
        renderer._n_radius = 5
        curses.init_pair(_EDGE_PAIR_BASE + 0, curses.COLOR_GREEN, -1)
        curses.init_pair(_EDGE_PAIR_BASE + 1, curses.COLOR_RED,   -1)
        curses.init_pair(_EDGE_PAIR_BASE + 2, curses.COLOR_GREEN, -1)
        curses.init_pair(_EDGE_PAIR_BASE + 3, curses.COLOR_RED,   -1)
        curses.init_pair(_IREG_BG_PAIR, -1, curses.COLOR_RED)
        curses.init_pair(_FILL_PAIR, curses.COLOR_BLUE, -1)
