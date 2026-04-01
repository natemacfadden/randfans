"""CLI for shape/vector generation.

Usage
-----
    python -m vcgame3d.shapes reflexive --h11 3
    python -m vcgame3d.shapes reflexive --h11 3 --polytope_id 1
"""
from __future__ import annotations

import argparse
import json

from . import _SHAPES, get_vectors


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m vcgame3d.shapes",
        description="Print integer vectors for a named shape as JSON.",
    )
    p.add_argument("shape", choices=_SHAPES, help="Shape to generate.")
    p.add_argument(
        "--h11",
        type=int,
        default=1,
        metavar="H11",
        help="Hodge number h^{1,1} for 'reflexive'. Default: 1.",
    )
    p.add_argument(
        "--polytope_id",
        type=int,
        default=0,
        metavar="ID",
        help="Index into polytopes with the given h11. Default: 0.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    vectors = get_vectors(args.shape, h11=args.h11, polytope_id=args.polytope_id)
    print(json.dumps(vectors))


if __name__ == "__main__":
    main()
