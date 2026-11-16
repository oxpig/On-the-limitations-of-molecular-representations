#!/usr/bin/env python3

import argparse
from collections import Counter
from pathlib import Path
import numpy as np


def read_hist_npz(path: Path) -> Counter:
    """ Read a fingerprint count histogram and return it """
    data        = np.load(path)
    run_lengths = data["run_lengths"]
    counts      = data["counts"]

    hist = Counter()
    for r, c in zip(run_lengths, counts):
        hist[int(r)] += int(c)

    return hist


def write_hist_tsv(hist: Counter, path: Path) -> None:
    """ Write out the global histogram """
    with open(path, "w") as f:
        f.write("run_length\tn_fingerprints\n")
        for run_length in sorted(hist):
            f.write(f"{run_length}\t{hist[run_length]}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hist-dir", required=True)
    parser.add_argument("--output-tsv", required=True)
    parser.add_argument("--output-npz", required=True)
    args = parser.parse_args()

    hist_dir = Path(args.hist_dir)
    paths    = sorted(hist_dir.glob("bucket_*_run_lengths.npz"))

    if not paths:
        raise RuntimeError(f"No per-bucket histogram files found in {hist_dir}")

    global_hist = Counter()

    # Loop through histograms and combine them
    for path in paths:
        bucket_hist = read_hist_npz(path)
        global_hist.update(bucket_hist)

    output_tsv = Path(args.output_tsv)
    output_npz = Path(args.output_npz)

    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    # Write the combined run histogram out
    write_hist_tsv(global_hist, output_tsv)

    run_lengths = np.array(sorted(global_hist), dtype=np.int64)
    counts      = np.array([global_hist[int(k)] for k in run_lengths], dtype=np.int64)

    np.savez_compressed(output_npz, run_lengths=run_lengths, counts=counts)

    # Summary statistics to print
    n_rows             = int(sum(k * v for k, v in global_hist.items()))
    n_distinct         = int(sum(global_hist.values()))
    n_duplicate_groups = int(sum(v for k, v in global_hist.items() if k > 1))
    n_rows_in_duplicate_groups = int(
        sum(k * v for k, v in global_hist.items() if k > 1)
    )
    max_run_length = int(max(global_hist) if global_hist else 0)

    print(f"Combined {len(paths)} bucket histograms")
    print(f"Total rows: {n_rows:,}")
    print(f"Distinct fingerprints: {n_distinct:,}")
    print(f"Duplicate fingerprint groups: {n_duplicate_groups:,}")
    print(f"Rows in duplicate groups: {n_rows_in_duplicate_groups:,}")
    print(f"Max run length: {max_run_length:,}")
    print(f"Wrote {output_tsv}")
    print(f"Wrote {output_npz}")


if __name__ == "__main__":
    main()
