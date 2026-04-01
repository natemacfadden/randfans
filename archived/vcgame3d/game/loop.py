"""
Main game loop for vcgame3d.
"""
from __future__ import annotations

import curses
import os
import time

import numpy as np

from .player              import Player3D, Player4D
from ..renderer.renderer  import draw, init_colors
from .scene               import build_scene
from .fan_scene           import fan_to_scene, auto_pole, fan_vertices, make_cone_finder
from ..renderer.projection import (stereographic_proj, inverse_stereographic_proj,
                                    hyperspherical_proj, inverse_hyperspherical_proj,
                                    normalize)
from ..renderer.renderer  import _HUD_ROWS, _FOV, _NEAR_CLIP, _CP_LABEL

_TURN_RATE  = 0.04   # radians per frame
_FPS        = 60
_KEY_TTL    = 0.12


def run(fan=None, n_subdivisions: int = 4, proj: str = "spherical") -> None:
    """Start the game loop.

    Parameters
    ----------
    fan : Fan-like object, optional
        Fan to render.  None → reference cube/grid scene.
    n_subdivisions : int
        Arc subdivision steps per fan edge.
    proj : str
        Projection from S³ → R³.  "spherical" (default) uses hyperspherical
        coordinates (no distortion, no poles); "stereo" uses stereographic.
    """
    os.environ.setdefault("ESCDELAY", "25")
    curses.wrapper(_main, fan, n_subdivisions, proj)


def _main(stdscr, fan=None, n_subdivisions: int = 4, proj: str = "spherical") -> None:
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(0)
    init_colors()

    if fan is not None:
        if proj == "stereo":
            _pp     = getattr(fan, 'preferred_pole', None)
            pole    = _pp if _pp is not None else auto_pole(fan)
            project     = stereographic_proj(pole=pole)
            inv_project = inverse_stereographic_proj(pole=pole)
        else:   # "spherical" (default)
            project     = hyperspherical_proj()
            inv_project = inverse_hyperspherical_proj()

        pts, edges, styles, arc_pts, edge_labels = fan_to_scene(fan, project, n_subdivisions)
        vertex_pts  = fan_vertices(fan, project)
        cone_finder = make_cone_finder(fan)

        # Sphere-normalized ray vectors for barycentric coordinate computation
        _fan_label_to_sv = {
            l: normalize(np.array(v, dtype=float))
            for l, v in zip(fan.labels, fan.vectors())
        }

        # Build initial 4D frame for Player4D
        verts_arr  = np.array(list(vertex_pts.values()))
        centroid   = verts_arr.mean(axis=0)
        max_dist   = float(np.max(np.linalg.norm(verts_arr - centroid, axis=1)))
        start_pos3d = centroid + np.array([0.0, 0.0, -max_dist])

        start_pos4d = inv_project(start_pos3d)

        # fwd4d: tangent direction toward the fan centroid on S³
        centroid4d = inv_project(centroid)
        tangent = centroid4d - np.dot(centroid4d, start_pos4d) * start_pos4d
        tn = np.linalg.norm(tangent)
        if tn < 1e-10:
            # Degenerate (already at centroid): pick any tangent
            for e in np.eye(4):
                tangent = e - np.dot(e, start_pos4d) * start_pos4d
                if np.linalg.norm(tangent) > 1e-10:
                    break
        fwd4d = tangent / np.linalg.norm(tangent)

        # right4d, up4d: Gram-Schmidt from standard basis
        axes4d = []
        for e in np.eye(4):
            v = e.copy()
            v -= np.dot(v, start_pos4d) * start_pos4d
            v -= np.dot(v, fwd4d) * fwd4d
            for prev in axes4d:
                v -= np.dot(v, prev) * prev
            nn = np.linalg.norm(v)
            if nn > 1e-10:
                axes4d.append(v / nn)
            if len(axes4d) == 2:
                break
        right4d, up4d = axes4d[0], axes4d[1]

        player = Player4D(
            pos4d=start_pos4d, fwd4d=fwd4d, right4d=right4d, up4d=up4d,
            project=project, speed=0.05,
        )
    else:
        pts, edges, styles = build_scene()
        arc_pts     = {}
        edge_labels = None
        vertex_pts  = {}
        cone_finder = None
        inv_project = None
        player      = Player3D(position=(0.0, 0.0, -5.0))
    hud_on = True

    # ── pynput for smooth key hold ────────────────────────────────────
    _active: set = set()
    _kb_listener = None
    _use_pynput  = False
    _last_seen: dict = {}
    _saved_stderr = None

    try:
        from pynput import keyboard as _kb
        _KEY_MAP = {
            _kb.Key.up:    curses.KEY_UP,
            _kb.Key.down:  curses.KEY_DOWN,
            _kb.Key.left:  curses.KEY_LEFT,
            _kb.Key.right: curses.KEY_RIGHT,
            _kb.Key.space: ord(" "),
        }
        def _on_press(k):
            mk = _KEY_MAP.get(k)
            if mk is not None:
                _active.add(mk)
            elif hasattr(k, "char") and k.char:
                _active.add(k.char)
        def _on_release(k):
            mk = _KEY_MAP.get(k)
            if mk is not None:
                _active.discard(mk)
            elif hasattr(k, "char") and k.char:
                _active.discard(k.char)
        # Redirect stderr before starting listener to suppress macOS
        # accessibility-permission warning that would corrupt the curses display
        _devnull_fd   = os.open(os.devnull, os.O_WRONLY)
        _saved_stderr = os.dup(2)
        os.dup2(_devnull_fd, 2)
        os.close(_devnull_fd)
        _kb_listener = _kb.Listener(on_press=_on_press, on_release=_on_release)
        _kb_listener.start()
        _use_pynput = True
    except Exception:
        if _saved_stderr is not None:
            os.dup2(_saved_stderr, 2)
            os.close(_saved_stderr)
            _saved_stderr = None

    try:
        while True:
            # ── drain curses key buffer ───────────────────────────────
            quit_game = False
            while True:
                key = stdscr.getch()
                if key == -1:
                    break
                if key == 27:  # Escape
                    quit_game = True
                if key == ord("h"):
                    hud_on = not hud_on
                if not _use_pynput:
                    _last_seen[key] = time.monotonic()
            if quit_game:
                break

            # ── build active key set ──────────────────────────────────
            if _use_pynput:
                active = set(_active)
            else:
                _now   = time.monotonic()
                # Normalize printable ASCII codes to strings so flight controls
                # (which check for "w", "s", etc.) work in both pynput and TTL mode
                active = {
                    chr(k) if 32 <= k < 127 else k
                    for k, t in _last_seen.items() if _now - t <= _KEY_TTL
                }
                _last_seen = {k: t for k, t in _last_seen.items() if _now - t <= _KEY_TTL}

            # ── flight controls ───────────────────────────────────────
            if curses.KEY_UP    in active: player.pitch( _TURN_RATE)
            if curses.KEY_DOWN  in active: player.pitch(-_TURN_RATE)
            if curses.KEY_LEFT  in active: player.yaw(   _TURN_RATE)
            if curses.KEY_RIGHT in active: player.yaw(  -_TURN_RATE)
            if "q" in active: player.roll(-_TURN_RATE)
            if "e" in active: player.roll( _TURN_RATE)
            if "w" in active: player.thrust( 1.0)
            if "s" in active: player.thrust(-1.0)
            if "a" in active: player.strafe(-1.0)
            if "d" in active: player.strafe( 1.0)
            if "r" in active: player.lift( 1.0)
            if "f" in active: player.lift(-1.0)
            if ord(" ") in active or " " in active:
                # brake: no translation this frame
                pass
            if "=" in active or "+" in active:
                player.speed = min(player.SPEED_MAX, player.speed + player.SPEED_STEP)
            if "-" in active or "_" in active:
                player.speed = max(player.SPEED_MIN, player.speed - player.SPEED_STEP)

            # ── render ────────────────────────────────────────────────
            # Compute current cone and derive highlight data
            cone = None
            highlighted_pairs = None
            face_arc_world_pts = None
            if cone_finder is not None:
                p4d  = player._pos4d   # direct S³ position — no round-trip error
                cone = cone_finder(p4d)
                if cone is not None:
                    from itertools import combinations as _comb

                    highlighted_pairs = frozenset(
                        (min(a, b), max(a, b)) for a, b in _comb(cone, 2)
                    )

                    # Barycentric coordinates of player in the cone
                    _M = np.column_stack([_fan_label_to_sv[l] for l in cone])
                    try:
                        _coeffs = np.linalg.solve(_M, p4d)
                        _coeffs = np.maximum(_coeffs, 0.0)
                        _coeffs /= max(_coeffs.sum(), 1e-10)
                    except np.linalg.LinAlgError:
                        _coeffs = np.ones(4) * 0.25

                    face_arc_world_pts = []
                    for face_idx, face_labels in enumerate(_comb(cone, 3)):
                        # face_idx i is opposite to cone[3-i] in itertools.combinations order
                        missing_coeff = float(_coeffs[3 - face_idx])
                        proximity = 1.0 - missing_coeff   # 0=far, 1=at the face
                        face_arcs = []
                        for a, b in _comb(face_labels, 2):
                            key = (min(a, b), max(a, b))
                            if key in arc_pts:
                                face_arcs.append([pts[idx] for idx in arc_pts[key]])
                        if len(face_arcs) == 3:
                            face_arc_world_pts.append((face_idx, proximity, face_arcs))

            draw(stdscr, player, pts, edges, styles, hud=hud_on,
                 edge_labels=edge_labels,
                 highlighted_pairs=highlighted_pairs,
                 face_arc_pts=face_arc_world_pts,
                 show_only_simplex=True)

            # ── vertex labels ─────────────────────────────────────────
            if vertex_pts:
                rows, cols = stdscr.getmaxyx()
                draw_rows  = rows - _HUD_ROWS
                cx = cols // 2
                cy = draw_rows // 2
                scale = min(draw_rows, cols // 2) * 0.45
                lbl_attr = curses.color_pair(_CP_LABEL) | curses.A_BOLD
                for lbl, vp in vertex_pts.items():
                    rel = vp - player._pos
                    vx  = float(np.dot(rel, player._right))
                    vy  = float(np.dot(rel, player._up))
                    vz  = float(np.dot(rel, player._fwd))
                    if vz < _NEAR_CLIP:
                        continue
                    sc = cx + int(round(vx / vz * _FOV * scale * 2))
                    sr = cy + int(round(-vy / vz * _FOV * scale))
                    if 0 <= sr < draw_rows and 0 <= sc + 1 < cols - 1:
                        try:
                            stdscr.addstr(sr, sc + 1, str(lbl), lbl_attr)
                        except curses.error:
                            pass

            # ── cone / HUD row 4 ──────────────────────────────────────
            if cone_finder is not None:
                rows, cols = stdscr.getmaxyx()
                if cone is not None:
                    cone_str = f"simplex: [{', '.join(str(l) for l in cone)}]"
                else:
                    cone_str = "simplex: (outside fan)"
                r_hud = rows - 1
                try:
                    stdscr.addstr(r_hud, 0, cone_str[:cols - 1])
                except curses.error:
                    pass

            stdscr.refresh()
            time.sleep(1.0 / _FPS)

    finally:
        if _kb_listener is not None:
            _kb_listener.stop()
        if _saved_stderr is not None:
            os.dup2(_saved_stderr, 2)
            os.close(_saved_stderr)
