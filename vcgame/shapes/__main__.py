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

"""CLI for shape/vector generation.

Usage
-----
    python -m shapes cube -n 5
    python -m shapes random --seed 42
    python -m shapes random -n 20 --maxcoord 4
    python -m shapes reflexive --polytope_id 7
    python -m shapes trunc_oct
"""
from __future__ import annotations

import argparse
import json

from . import _SHAPES, get_vectors


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m shapes",
        description="Print integer vectors for a named shape as JSON.",
    )
    p.add_argument(
        "shape",
        choices=_SHAPES,
        help="Shape to generate.",
    )
    p.add_argument(
        "-n",
        type=int,
        default=None,
        metavar="N",
        help="Grid size for 'cube' (odd, >= 3) or seed vector count for 'random'. Required for 'cube'.",
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
        help="RNG seed. Only used for 'random'. Default: 1102.",
    )
    p.add_argument(
        "--polytope_id",
        type=int,
        default=0,
        metavar="ID",
        help="Reflexive polytope index 0–4318. Only used for 'reflexive'.",
    )
    return p


def main() -> None:
    p = _build_parser()
    args = p.parse_args()
    if args.shape == "cube" and args.n is None:
        p.error("-n is required for 'cube'")
    if args.shape in ("trunc_oct", "reflexive"):
        if args.n is not None:
            p.error(f"-n is not valid for '{args.shape}'")
        if args.maxcoord != 3:
            p.error(f"--maxcoord is not valid for '{args.shape}'")

    vectors = get_vectors(
        args.shape,
        n=args.n,
        n_vectors=args.n,
        max_coord=args.maxcoord,
        seed=args.seed,
        polytope_id=args.polytope_id,
    )
    print(json.dumps(vectors))


if __name__ == "__main__":
    main()
