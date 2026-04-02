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

"""vcgame — entry point."""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np

from game import run_display_demo


def _parse_sph_arg(s: str) -> np.ndarray:
    """Parse 'az,el' (degrees) into a unit Cartesian vector."""
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"expected 'az,el' in degrees, got {s!r}"
        )
    az = math.radians(float(parts[0]))
    el = math.radians(float(parts[1]))
    return np.array([
        math.cos(el) * math.cos(az),
        math.cos(el) * math.sin(az),
        math.sin(el),
    ], dtype=float)


def _fix_negative_args() -> None:
    """Join --pos/--heading with their value using '=' so argparse doesn't
    mistake a leading '-' in a negative angle for a flag.
    """
    _vec_flags = {"--pos", "--heading"}
    argv = sys.argv
    fixed = [argv[0]]
    i = 1
    while i < len(argv):
        if argv[i] in _vec_flags and i + 1 < len(argv):
            fixed.append(f"{argv[i]}={argv[i + 1]}")
            i += 2
        else:
            fixed.append(argv[i])
            i += 1
    sys.argv = fixed


def _parse_args() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    p = argparse.ArgumentParser(
        prog="vcgame",
        description="Navigate a simplicial fan on S².",
    )
    p.add_argument(
        "--shape",
        choices=["cube", "trunc_oct", "random", "reflexive"],
        default="cube",
        help="Vector configuration shape (default: cube).",
    )
    p.add_argument(
        "-n",
        type=int,
        default=None,
        metavar="N",
        help="Grid size for 'cube' (nxnxn) or vector count for 'random'.",
    )
    p.add_argument(
        "--maxcoord",
        type=int,
        default=3,
        metavar="C",
        help="Coordinate range for 'random' (default: 3).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=1102,
        metavar="N",
        help="RNG seed for --shape random (default: 1102).",
    )
    p.add_argument(
        "--polytope",
        type=int,
        default=0,
        metavar="ID",
        help="Reflexive polytope id 0–4318 for --shape reflexive (default: 0).",
    )
    p.add_argument(
        "--pos",
        type=str,
        default=None,
        metavar="az,el",
        help="Initial player position as azimuth,elevation in degrees.",
    )
    p.add_argument(
        "--heading",
        type=str,
        default=None,
        metavar="az,el",
        help="Initial player heading as azimuth,elevation in degrees.",
    )
    p.add_argument(
        "--color",
        type=int,
        choices=[0, 1, 2],
        default=0,
        metavar="N",
        help="Initial color mode: 0=wireframe, 1=radius, 2=sun (default: 0).",
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="Render a single frame then exit (useful for benchmarking).",
    )
    return p, p.parse_args()


def main() -> None:
    _fix_negative_args()
    p, args = _parse_args()

    if args.shape in ("trunc_oct", "reflexive"):
        if args.n is not None:
            p.error(f"-n is not valid for --shape {args.shape}")
        if args.maxcoord != 3:
            p.error(f"--maxcoord is not valid for --shape {args.shape}")

    from regfans import VectorConfiguration
    from shapes import get_vectors

    vectors = get_vectors(
        args.shape,
        seed=args.seed,
        polytope_id=args.polytope,
        n=args.n,
        n_vectors=args.n,
        max_coord=args.maxcoord,
    )
    vc = VectorConfiguration(vectors)
    fan = vc.triangulate()

    # Reconstruct the effective CLI for debug dumps.
    cli_cmd = " ".join(sys.argv)

    initial_pos     = _parse_sph_arg(args.pos)     if args.pos     else None
    initial_heading = _parse_sph_arg(args.heading) if args.heading else None

    run_display_demo(
        fan, vc,
        agent=None,
        initial_pos=initial_pos,
        initial_heading=initial_heading,
        initial_color=args.color,
        vectors=vectors,
        cli_cmd=cli_cmd,
        max_frames=1 if args.once else None,
    )


if __name__ == "__main__":
    main()
