"""
Curses perspective renderer for 3D flight.
"""
from __future__ import annotations

import curses
import numpy as np
from .player import Player3D

_HUD_ROWS  = 3
_FOV       = 1.2    # focal length; larger = narrower FOV
_NEAR_CLIP = 0.5

# curses color pair indices
_CP_DEFAULT = 1
_CP_CUBE    = 2
_CP_AXIS    = 3
_CP_GRID    = 4
_CP_HUD     = 5


def init_colors() -> None:
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(_CP_DEFAULT, curses.COLOR_WHITE,  -1)
    curses.init_pair(_CP_CUBE,    curses.COLOR_CYAN,   -1)
    curses.init_pair(_CP_AXIS,    curses.COLOR_YELLOW, -1)
    curses.init_pair(_CP_GRID,    curses.COLOR_WHITE,  -1)
    curses.init_pair(_CP_HUD,     curses.COLOR_GREEN,  -1)


_STYLE_ATTR: dict[str, tuple[str, int]] = {
    # style -> (draw char, color pair)
    "cube": ("*", _CP_CUBE),
    "axis": ("+", _CP_AXIS),
    "grid": (".", _CP_GRID),
}


def _project(p_world: np.ndarray, player: Player3D):
    """Project world point to (sx, sy) normalised screen coords.
    Returns None if behind near clip plane."""
    rel = p_world - player._pos
    x   = float(np.dot(rel, player.right))
    y   = float(np.dot(rel, player._up))
    z   = float(np.dot(rel, player._fwd))
    if z < _NEAR_CLIP:
        return None
    return x / z * _FOV, -y / z * _FOV   # negate y: rows increase downward


def _addstr(scr, r: int, c: int, text: str, attr: int = 0) -> None:
    try:
        scr.addstr(r, c, text, attr)
    except curses.error:
        pass


def _draw_line(scr, r0, c0, r1, c1, ch: str, attr: int, max_r: int, max_c: int) -> None:
    """Bresenham line between two screen points."""
    dr = abs(r1 - r0); dc = abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    err = dr - dc
    r, c = r0, c0
    for _ in range(dr + dc + 1):
        if 0 <= r < max_r and 0 <= c < max_c - 1:
            _addstr(scr, r, c, ch, attr)
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc; r += sr
        if e2 <  dr:
            err += dr; c += sc


def draw(
    scr,
    player:  Player3D,
    pts:     list,
    edges:   list,
    styles:  list,
) -> None:
    rows, cols = scr.getmaxyx()
    draw_rows  = rows - _HUD_ROWS
    cx = cols // 2
    cy = draw_rows // 2
    scale = min(draw_rows, cols // 2) * 0.45

    scr.erase()

    # Pre-compute camera-space coordinates for all points
    right = player._right
    up    = player._up
    fwd   = player._fwd
    pos   = player._pos
    cam = []
    for p in pts:
        rel = p - pos
        cam.append((float(np.dot(rel, right)),
                    float(np.dot(rel, up)),
                    float(np.dot(rel, fwd))))

    # Draw edges with near-plane clipping
    for (i, j), style in zip(edges, styles):
        x0, y0, z0 = cam[i]
        x1, y1, z1 = cam[j]
        if z0 < _NEAR_CLIP and z1 < _NEAR_CLIP:
            continue
        if z0 < _NEAR_CLIP:
            t = (_NEAR_CLIP - z0) / (z1 - z0)
            x0, y0, z0 = x0 + t*(x1-x0), y0 + t*(y1-y0), _NEAR_CLIP
        elif z1 < _NEAR_CLIP:
            t = (_NEAR_CLIP - z1) / (z0 - z1)
            x1, y1, z1 = x1 + t*(x0-x1), y1 + t*(y0-y1), _NEAR_CLIP
        c0 = cx + int(round(x0/z0 * _FOV * scale * 2))
        r0 = cy + int(round(-y0/z0 * _FOV * scale))
        c1 = cx + int(round(x1/z1 * _FOV * scale * 2))
        r1 = cy + int(round(-y1/z1 * _FOV * scale))
        ch, cp = _STYLE_ATTR.get(style, (".", _CP_DEFAULT))
        _draw_line(scr, r0, c0, r1, c1, ch, curses.color_pair(cp), draw_rows, cols)

    # crosshair (┼ at true centre cx)
    attr = curses.color_pair(_CP_DEFAULT) | curses.A_BOLD
    _addstr(scr, cy,     cx - 1, "─", attr)
    _addstr(scr, cy,     cx,     "┼", attr)
    _addstr(scr, cy,     cx + 1, "─", attr)
    _addstr(scr, cy - 1, cx,     "│", attr)
    _addstr(scr, cy + 1, cx,     "│", attr)

    # HUD
    _draw_hud(scr, rows, cols, player)


def _draw_hud(scr, rows: int, cols: int, player: Player3D) -> None:
    attr_hud = curses.color_pair(_CP_HUD)
    r0       = rows - _HUD_ROWS
    blank    = " " * (cols - 1)
    for hr in range(_HUD_ROWS):
        _addstr(scr, r0 + hr, 0, blank, attr_hud)

    pos = player.position
    fwd = player.forward
    up  = player.up

    line0 = (
        f"pos ({pos[0]:+6.2f}, {pos[1]:+6.2f}, {pos[2]:+6.2f})  "
        f"fwd ({fwd[0]:+5.2f}, {fwd[1]:+5.2f}, {fwd[2]:+5.2f})  "
        f"up ({up[0]:+5.2f}, {up[1]:+5.2f}, {up[2]:+5.2f})  "
        f"spd {player.speed:.2f}"
    )
    line1 = "[↑↓] pitch   [←→] yaw   [z/c] roll   [w/s] thrust   [a/d] strafe   [r/f] lift"
    line2 = "[+/-] speed  [space] brake  [q] quit"

    _addstr(scr, r0,     0, line0[:cols - 1], attr_hud)
    _addstr(scr, r0 + 1, 0, line1[:cols - 1], attr_hud)
    _addstr(scr, r0 + 2, 0, line2[:cols - 1], attr_hud)
