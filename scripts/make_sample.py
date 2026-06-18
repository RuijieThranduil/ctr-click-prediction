"""Create the reproducible Avazu sample used by the CTR project.

The default settings match the reported project sample:
    python scripts/make_sample.py train/train.csv data/filtered_train.csv

This performs a seeded random sample of the full Kaggle train.csv while always
preserving the header row. It streams the input file, so it can handle the full
40M-row Avazu CSV without loading it into memory.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0 or parsed > 1:
        raise argparse.ArgumentTypeError("sample fraction must be in (0, 1]")
    return parsed


def make_sample(input_csv: Path, output_csv: Path, frac: float, seed: int) -> tuple[int, int]:
    rng = random.Random(seed)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows_seen = 0
    rows_written = 0
    with input_csv.open("r", newline="", encoding="utf-8") as src, output_csv.open(
        "w", newline="", encoding="utf-8"
    ) as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst)

        header = next(reader)
        writer.writerow(header)

        for row in reader:
            rows_seen += 1
            if rng.random() < frac:
                writer.writerow(row)
                rows_written += 1

    return rows_seen, rows_written


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a seeded Avazu CSV sample.")
    parser.add_argument("input_csv", type=Path, help="Path to the full Kaggle train.csv")
    parser.add_argument(
        "output_csv",
        type=Path,
        nargs="?",
        default=Path("data/filtered_train.csv"),
        help="Sample output path (default: data/filtered_train.csv)",
    )
    parser.add_argument("--frac", type=positive_float, default=0.01, help="Sample fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rows_seen, rows_written = make_sample(args.input_csv, args.output_csv, args.frac, args.seed)
    pct = 100 * rows_written / rows_seen if rows_seen else 0.0
    print(
        f"Wrote {rows_written:,} of {rows_seen:,} rows ({pct:.2f}%) "
        f"to {args.output_csv}"
    )


if __name__ == "__main__":
    main()
