#!/usr/bin/env python3

import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np


def now() -> str:
    """
    Get current time for logging.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    """
    Log a message with the current time.
    """
    print(f"[{now()}] {msg}", flush=True)


def format_rate(rows: int, seconds: float) -> str:
    """
    Used to format rates for log timings.
    """
    if seconds <= 0:
        return "unknown"
    return f"{rows / seconds:,.0f} rows/s"


def update_hist_from_run_lengths(hist: Counter, run_lengths: np.ndarray) -> None:
    """
    Add many run lengths to histogram efficiently.
    """
    if run_lengths.size == 0:
        return

    values, counts = np.unique(run_lengths, return_counts=True)

    for v, c in zip(values, counts):
        hist[int(v)] += int(c)


def compute_run_length_histogram(sorted_path: Path, row_bytes: int, chunk_rows: int, log_interval_seconds: int,) -> Counter:
    """
    Compute histogram of duplicate run lengths from a sorted packed-fingerprint .npy.

    Input array shape should be (n_rows, row_bytes), dtype uint8.
    Rows must already be sorted lexicographically.
    """
    arr = np.load(sorted_path, mmap_mode="r")

    if arr.ndim != 2:
        raise ValueError(f"{sorted_path} has shape {arr.shape}; expected 2D")

    if arr.dtype != np.uint8:
        raise ValueError(f"{sorted_path} has dtype {arr.dtype}; expected uint8")

    n_rows, observed_row_bytes = arr.shape

    if observed_row_bytes != row_bytes:
        raise ValueError(
            f"{sorted_path} has row width {observed_row_bytes}; expected {row_bytes}"
        )

    void_dtype = np.dtype((np.void, row_bytes))
    hist = Counter()

    carry_key = None
    carry_len = 0

    t0            = time.time()
    last_log_time = t0
    processed     = 0

    for start in range(0, n_rows, chunk_rows):
        stop = min(start + chunk_rows, n_rows)

        chunk = np.ascontiguousarray(arr[start:stop])
        keys = chunk.view(void_dtype).reshape(-1)

        if keys.size == 0:
            continue

        changes = np.flatnonzero(keys[1:] != keys[:-1]) + 1

        starts = np.r_[0, changes]
        stops  = np.r_[changes, keys.size]
        run_lengths = stops - starts

        first_key = bytes(keys[starts[0]])
        last_key  = bytes(keys[starts[-1]])

        if carry_key is not None:
            if first_key == carry_key:
                run_lengths[0] += carry_len
            else:
                hist[carry_len] += 1

        # Hold back the last run because it may continue into the next chunk.
        complete_run_lengths = run_lengths[:-1]
        update_hist_from_run_lengths(hist, complete_run_lengths)

        carry_key = last_key
        carry_len = int(run_lengths[-1])

        processed += stop - start

        current_time = time.time()
        if current_time - last_log_time >= log_interval_seconds or processed == n_rows:
            elapsed = current_time - t0
            log(
                f"Histogram progress: rows {processed:,}/{n_rows:,}, "
                f"elapsed {elapsed / 60:.1f} min, "
                f"rate {format_rate(processed, elapsed)}"
            )
            last_log_time = current_time

    if carry_key is not None:
        hist[carry_len] += 1

    return hist


def write_histogram_tsv(hist: Counter, path: Path) -> None:
    """ Write output histogram """
    with open(path, "w") as f:
        f.write("run_length\tn_fingerprints\n")
        for run_length in sorted(hist):
            f.write(f"{run_length}\t{hist[run_length]}\n")


def write_histogram_npz(hist: Counter, path: Path) -> None:
    """ Write output histogram """
    run_lengths = np.array(sorted(hist), dtype=np.int64)
    counts      = np.array([hist[int(k)] for k in run_lengths], dtype=np.int64)
    np.savez_compressed(path, run_lengths=run_lengths, counts=counts)


def process_bucket(args: argparse.Namespace) -> None:
    """ 
        Full pipeline. Sorts fingerprint buckets and 
        produces a per-bucket degeneracy histogram.
        These will later be combined.
    """
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    hist_dir   = Path(args.hist_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    hist_dir.mkdir(parents=True, exist_ok=True)

    bucket_id = args.bucket_id

    input_path  = input_dir / f"bucket_{bucket_id:05d}.npy"
    output_path = output_dir / f"bucket_{bucket_id:05d}.npy"
    tmp_output_path = output_dir / f"bucket_{bucket_id:05d}.npy.tmp"

    hist_tsv_path = hist_dir / f"bucket_{bucket_id:05d}_run_lengths.tsv"
    hist_npz_path = hist_dir / f"bucket_{bucket_id:05d}_run_lengths.npz"
    summary_json_path = hist_dir / f"bucket_{bucket_id:05d}_summary.json"

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input bucket file: {input_path}")

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Use --overwrite to replace it."
        )

    if tmp_output_path.exists():
        tmp_output_path.unlink()

    log(f"Bucket {bucket_id}: loading {input_path}")

    arr = np.load(input_path, mmap_mode="r")

    if arr.ndim != 2:
        raise ValueError(f"{input_path} has shape {arr.shape}; expected 2D")

    if arr.dtype != np.uint8:
        raise ValueError(f"{input_path} has dtype {arr.dtype}; expected uint8")

    n_rows, row_bytes = arr.shape

    if row_bytes != args.row_bytes:
        raise ValueError(
            f"{input_path} has row width {row_bytes}; expected {args.row_bytes}"
        )

    log(
        f"Bucket {bucket_id}: input shape=({n_rows:,}, {row_bytes}), "
        f"size={n_rows * row_bytes / 1e9:.3f} GB"
    )

    t0 = time.time()

    if n_rows == 0:
        out = np.lib.format.open_memmap(
            tmp_output_path,
            mode="w+",
            dtype=np.uint8,
            shape=(0, row_bytes),
        )
        out.flush()
        del out

        hist = Counter()
    else:
        void_dtype = np.dtype((np.void, row_bytes))

        # View each row as one fixed-width byte string.
        # Sorting these gives lexicographic row order.
        keys = arr.view(void_dtype).reshape(n_rows)

        log(f"Bucket {bucket_id}: starting argsort")
        sort_t0 = time.time()

        order = np.argsort(keys, kind=args.sort_kind)

        sort_elapsed = time.time() - sort_t0
        log(
            f"Bucket {bucket_id}: argsort finished in "
            f"{sort_elapsed / 60:.1f} min"
        )

        out = np.lib.format.open_memmap(
            tmp_output_path,
            mode="w+",
            dtype=np.uint8,
            shape=(n_rows, row_bytes),
        )

        copy_t0       = time.time()
        last_log_time = copy_t0
        copied        = 0

        log(f"Bucket {bucket_id}: writing sorted .npy to temporary file")

        for start in range(0, n_rows, args.copy_rows):
            stop = min(start + args.copy_rows, n_rows)
            out[start:stop] = arr[order[start:stop]]
            copied += stop - start

            current_time = time.time()
            if (
                current_time - last_log_time >= args.log_interval_seconds
                or copied == n_rows
            ):
                elapsed = current_time - copy_t0
                log(
                    f"Bucket {bucket_id}: sorted-write progress "
                    f"{copied:,}/{n_rows:,} rows, "
                    f"elapsed {elapsed / 60:.1f} min, "
                    f"rate {format_rate(copied, elapsed)}"
                )
                last_log_time = current_time

        out.flush()
        del out
        del order

        os.replace(tmp_output_path, output_path)

        log(f"Bucket {bucket_id}: sorted output finalized at {output_path}")

        log(f"Bucket {bucket_id}: computing run-length histogram")
        hist = compute_run_length_histogram(
            sorted_path=output_path,
            row_bytes=row_bytes,
            chunk_rows=args.hist_chunk_rows,
            log_interval_seconds=args.log_interval_seconds,
        )

    write_histogram_tsv(hist, hist_tsv_path)
    write_histogram_npz(hist, hist_npz_path)

    n_distinct         = int(sum(hist.values()))
    n_total            = int(sum(k * v for k, v in hist.items()))
    n_duplicate_groups = int(sum(v for k, v in hist.items() if k > 1))
    n_duplicate_rows   = int(sum(k * v for k, v in hist.items() if k > 1))
    max_run_length     = int(max(hist) if hist else 0)

    summary = {
        "bucket_id": bucket_id,
        "input_path": str(input_path),
        "sorted_output_path": str(output_path),
        "histogram_tsv_path": str(hist_tsv_path),
        "histogram_npz_path": str(hist_npz_path),
        "n_rows": n_total,
        "n_distinct_fingerprints": n_distinct,
        "n_duplicate_fingerprint_groups": n_duplicate_groups,
        "n_rows_in_duplicate_groups": n_duplicate_rows,
        "max_run_length": max_run_length,
        "row_bytes": row_bytes,
    }

    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    total_elapsed = time.time() - t0

    log(
        f"Bucket {bucket_id}: done. "
        f"rows={n_total:,}, distinct={n_distinct:,}, "
        f"duplicate groups={n_duplicate_groups:,}, "
        f"rows in duplicate groups={n_duplicate_rows:,}, "
        f"max run length={max_run_length:,}, "
        f"time={total_elapsed / 60:.1f} min"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hist-dir", required=True)
    parser.add_argument("--bucket-id", type=int, required=True)

    parser.add_argument("--row-bytes", type=int, default=128)
    parser.add_argument("--copy-rows", type=int, default=500_000)
    parser.add_argument("--hist-chunk-rows", type=int, default=1_000_000)

    parser.add_argument(
        "--sort-kind",
        default="stable",
        choices=["quicksort", "mergesort", "heapsort", "stable"],
    )

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-interval-seconds", type=int, default=60)

    args = parser.parse_args()

    try:
        process_bucket(args)
    except Exception as e:
        log(f"ERROR: {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
