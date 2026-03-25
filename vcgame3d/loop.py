"""
Main game loop for vcgame3d.
"""
from __future__ import annotations

import curses
import os
import time

from .player   import Player3D
from .renderer import draw, init_colors
from .scene    import build_scene

_TURN_RATE  = 0.04   # radians per frame
_SPEED_STEP = 0.05
_SPEED_MIN  = 0.01
_SPEED_MAX  = 2.0
_FPS        = 60
_KEY_TTL    = 0.12


def run() -> None:
    curses.wrapper(_main)


def _main(stdscr) -> None:
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(0)
    init_colors()

    player = Player3D(position=(0.0, 0.0, -5.0))
    pts, edges, styles = build_scene()

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
                if key == ord("q"):
                    quit_game = True
                if not _use_pynput:
                    _last_seen[key] = time.monotonic()
            if quit_game:
                break

            # ── build active key set ──────────────────────────────────
            if _use_pynput:
                active = set(_active)
            else:
                _now   = time.monotonic()
                active = {k for k, t in _last_seen.items() if _now - t <= _KEY_TTL}
                _last_seen = {k: t for k, t in _last_seen.items() if _now - t <= _KEY_TTL}

            # ── flight controls ───────────────────────────────────────
            if curses.KEY_UP    in active: player.pitch( _TURN_RATE)
            if curses.KEY_DOWN  in active: player.pitch(-_TURN_RATE)
            if curses.KEY_LEFT  in active: player.yaw(   _TURN_RATE)
            if curses.KEY_RIGHT in active: player.yaw(  -_TURN_RATE)
            if "z" in active: player.roll(-_TURN_RATE)
            if "c" in active: player.roll( _TURN_RATE)
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
                player.speed = min(_SPEED_MAX, player.speed + _SPEED_STEP)
            if "-" in active or "_" in active:
                player.speed = max(_SPEED_MIN, player.speed - _SPEED_STEP)

            # ── render ────────────────────────────────────────────────
            draw(stdscr, player, pts, edges, styles)
            stdscr.refresh()
            time.sleep(1.0 / _FPS)

    finally:
        if _kb_listener is not None:
            _kb_listener.stop()
        if _saved_stderr is not None:
            os.dup2(_saved_stderr, 2)
            os.close(_saved_stderr)
