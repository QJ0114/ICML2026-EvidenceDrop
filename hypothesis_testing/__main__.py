"""Recompute calibration and held-out metrics on pre-scored JSON files."""

import json
from argparse import ArgumentParser
from pathlib import Path

from hypothesis_testing.testing import calibrate, evaluate


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.1)
    args = parser.parse_args()
    if args.calibration.resolve() == args.test.resolve():
        parser.error("Supply separate calibration and test files.")
    threshold = calibrate(json.loads(args.calibration.read_text()), args.alpha, args.delta)
    metrics = evaluate(json.loads(args.test.read_text()), threshold["threshold"])
    print(json.dumps({**threshold, **metrics}, indent=2))


if __name__ == "__main__":
    main()
