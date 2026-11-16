#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


UINT64_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


def now() -> str:
    """
    Get current time for tracking runtime.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    """
    Write message to log with the current time
    """
    print(f"[{now()}] {msg}", flush=True)


def splitmix64(x: np.ndarray) -> np.ndarray:
    """
    Fast deterministic 64-bit mixer.
    """
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x = x ^ (x >> np.uint64(31))
    return x & UINT64_MASK


def hash_packed_rows_u64(chunk: np.ndarray, seed: int = 0) -> np.ndarray:
    """
    Hash packed uint8 rows.

    Input shape: (n_rows, n_bytes)
    Output shape: (n_rows,), dtype uint64
    """
    if chunk.dtype != np.uint8:
        raise ValueError(f"Expected uint8 input, got {chunk.dtype}")

    if chunk.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {chunk.shape}")

    n_rows, n_bytes = chunk.shape

    if n_bytes % 8 != 0:
        raise ValueError("Row width must be divisible by 8 bytes for this hasher")

    # Make sure the fingerprints are stored in c-order
    c = np.ascontiguousarray(chunk)
    # Reshape each fingerprint to be 16 uint64 integers (in 1024 bit case)
    # E.g. shape would be (rows, 16)
    words = c.view("<u8").reshape(n_rows, n_bytes // 8)
    # Create an empty array for the hashes.
    h = np.full(n_rows, np.uint64(seed), dtype=np.uint64)
    # Process column wise
    for j in range(words.shape[1]):
        # Extract first 64-bit int of the 16 entry arrays
        w  = words[:, j]
        h ^= splitmix64(w + np.uint64(j + 1) * np.uint64(0x9E3779B97F4A7C15))
        h  = splitmix64(h)

    return h


def list_npy_files(input_dir: Path) -> list[Path]:
    """ Return a sorted list of all of the .npy fingerprint files """
    return sorted(p for p in input_dir.iterdir() if p.suffix == ".npy")


def assigned_files(files: list[Path], task_id: int, n_tasks: int) -> list[Path]:
    """
    Contiguous partitioning of files across SLURM array tasks.
    """
    n     = len(files)
    start = task_id * n // n_tasks
    end   = (task_id + 1) * n // n_tasks
    return files[start:end]


def format_rate(rows: int, seconds: float) -> str:
    if seconds <= 0:
        return "unknown"
    return f"{rows / seconds:,.0f} rows/s"


def map_stage(args: argparse.Namespace) -> None:
    """ 
    First stage of the pipeline. Map fingerprints 
    into different buckets based on a deterministic 
    hash function (splitmix)
    """
    input_dir = Path(args.input_dir)
    tmp_dir   = Path(args.tmp_dir)

    files    = list_npy_files(input_dir)
    my_files = assigned_files(files, args.task_id, args.n_tasks)

    task_dir = tmp_dir / f"task_{args.task_id:05d}"

    if task_dir.exists():
        if args.overwrite_task_tmp:
            shutil.rmtree(task_dir)
        else:
            raise FileExistsError(
                f"{task_dir} already exists. Use --overwrite-task-tmp to replace it."
            )

    task_dir.mkdir(parents=True, exist_ok=False)

    counts = np.zeros(args.n_buckets, dtype=np.int64)
    total_rows_seen  = 0
    total_bytes_read = 0
    t0 = time.time()
    last_log_time    = t0
    last_log_rows    = 0

    log(
        f"MAP task {args.task_id}/{args.n_tasks - 1} starting. "
        f"Assigned {len(my_files)} files out of {len(files)} total files."
    )

    if len(my_files) == 0:
        np.save(task_dir / "counts.npy", counts)
        log(f"MAP task {args.task_id} has no files. Done.")
        return

    for file_idx, path in enumerate(my_files, 1):
        file_t0 = time.time()
        arr = np.load(path, mmap_mode="r")

        if arr.ndim != 2:
            raise ValueError(f"{path} has shape {arr.shape}; expected 2D")

        if arr.dtype != np.uint8:
            raise ValueError(f"{path} has dtype {arr.dtype}; expected uint8")

        n_rows, row_bytes = arr.shape

        if row_bytes != args.row_bytes:
            raise ValueError(
                f"{path} has row width {row_bytes}; expected {args.row_bytes}"
            )

        log(
            f"MAP task {args.task_id}: file {file_idx}/{len(my_files)} "
            f"{path.name}, shape=({n_rows:,}, {row_bytes}), "
            f"size={n_rows * row_bytes / 1e9:.3f} GB"
        )

        file_rows_written = 0

        for start in range(0, n_rows, args.chunk_rows):
            stop = min(start + args.chunk_rows, n_rows)
            chunk = np.asarray(arr[start:stop])

            h = hash_packed_rows_u64(chunk, seed=args.seed)
            buckets = (h % np.uint64(args.n_buckets)).astype(np.int64)

            counts += np.bincount(buckets, minlength=args.n_buckets)

            # Sort by bucket so each bucket is written as one contiguous block.
            order = np.argsort(buckets, kind="stable")
            sorted_buckets = buckets[order]
            sorted_rows = chunk[order]

            boundaries = np.flatnonzero(sorted_buckets[1:] != sorted_buckets[:-1]) + 1
            starts     = np.r_[0, boundaries]
            stops      = np.r_[boundaries, len(sorted_buckets)]

            for s, e in zip(starts, stops):
                b = int(sorted_buckets[s])
                out_bin = task_dir / f"bucket_{b:05d}.bin"
                with open(out_bin, "ab") as f:
                    sorted_rows[s:e].tofile(f)

            rows_this_chunk    = stop - start
            total_rows_seen   += rows_this_chunk
            file_rows_written += rows_this_chunk
            total_bytes_read  += rows_this_chunk * row_bytes

            current_time = time.time()
            if (
                current_time - last_log_time >= args.log_interval_seconds
                or stop == n_rows
            ):
                elapsed = current_time - t0
                interval_rows = total_rows_seen - last_log_rows
                interval_elapsed = current_time - last_log_time

                log(
                    f"MAP task {args.task_id}: progress "
                    f"file {file_idx}/{len(my_files)}, "
                    f"rows in current file {stop:,}/{n_rows:,}, "
                    f"total rows {total_rows_seen:,}, "
                    f"elapsed {elapsed / 60:.1f} min, "
                    f"overall rate {format_rate(total_rows_seen, elapsed)}, "
                    f"recent rate {format_rate(interval_rows, interval_elapsed)}"
                )

                last_log_time = current_time
                last_log_rows = total_rows_seen

        file_elapsed = time.time() - file_t0
        log(
            f"MAP task {args.task_id}: finished {path.name}. "
            f"Rows={file_rows_written:,}, "
            f"time={file_elapsed / 60:.1f} min, "
            f"rate={format_rate(file_rows_written, file_elapsed)}"
        )

    np.save(task_dir / "counts.npy", counts)

    manifest = {
        "task_id": args.task_id,
        "n_tasks": args.n_tasks,
        "n_buckets": args.n_buckets,
        "row_bytes": args.row_bytes,
        "chunk_rows": args.chunk_rows,
        "seed": args.seed,
        "n_files": len(my_files),
        "total_rows_seen": int(total_rows_seen),
        "total_bytes_read": int(total_bytes_read),
        "files": [str(p) for p in my_files],
    }

    with open(task_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - t0
    log(
        f"MAP task {args.task_id} done. "
        f"Files={len(my_files)}, rows={total_rows_seen:,}, "
        f"data={total_bytes_read / 1e12:.3f} TB, "
        f"time={elapsed / 60:.1f} min, "
        f"rate={format_rate(total_rows_seen, elapsed)}"
    )


def reduce_stage(args: argparse.Namespace) -> None:
    """
    Second stage of the pipeline. Combines bucket 
    files across workers.
    """
    tmp_dir = Path(args.tmp_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bucket = args.bucket_id

    t0 = time.time()

    log(f"REDUCE bucket {bucket} starting.")

    task_dirs = sorted(tmp_dir.glob("task_*"))
    if not task_dirs:
        raise RuntimeError(f"No task directories found under {tmp_dir}")

    log(f"REDUCE bucket {bucket}: found {len(task_dirs)} task directories.")

    total_rows = 0
    nonempty_shards = 0

    for task_dir in task_dirs:
        counts_path = task_dir / "counts.npy"
        if not counts_path.exists():
            raise FileNotFoundError(f"Missing {counts_path}")

        counts = np.load(counts_path)
        bucket_rows = int(counts[bucket])
        total_rows += bucket_rows

        if bucket_rows > 0:
            nonempty_shards += 1

    log(
        f"REDUCE bucket {bucket}: expected rows={total_rows:,}, "
        f"nonempty shards={nonempty_shards:,}"
    )

    final_path = output_dir / f"bucket_{bucket:05d}.npy"
    tmp_final_path = output_dir / f"bucket_{bucket:05d}.npy.tmp"

    if final_path.exists() and not args.overwrite_output:
        raise FileExistsError(
            f"{final_path} already exists. Use --overwrite-output to replace it."
        )

    if tmp_final_path.exists():
        tmp_final_path.unlink()

    out = np.lib.format.open_memmap(
        tmp_final_path,
        mode="w+",
        dtype=np.uint8,
        shape=(total_rows, args.row_bytes),
    )

    pos = 0
    last_log_time = time.time()
    last_log_rows = 0

    for shard_idx, task_dir in enumerate(task_dirs, 1):
        bin_path = task_dir / f"bucket_{bucket:05d}.bin"

        if not bin_path.exists():
            continue

        n_bytes = bin_path.stat().st_size

        if n_bytes % args.row_bytes != 0:
            raise ValueError(
                f"{bin_path} size {n_bytes} is not divisible by row width "
                f"{args.row_bytes}"
            )

        n_rows = n_bytes // args.row_bytes

        data = np.memmap(
            bin_path,
            mode="r",
            dtype=np.uint8,
            shape=(n_rows, args.row_bytes),
        )

        for start in range(0, n_rows, args.copy_rows):
            stop = min(start + args.copy_rows, n_rows)
            rows_this_chunk = stop - start
            out[pos : pos + rows_this_chunk] = data[start:stop]
            pos += rows_this_chunk

            current_time = time.time()
            if (
                current_time - last_log_time >= args.log_interval_seconds
                or pos == total_rows
            ):
                elapsed = current_time - t0
                interval_rows = pos - last_log_rows
                interval_elapsed = current_time - last_log_time

                log(
                    f"REDUCE bucket {bucket}: progress "
                    f"rows {pos:,}/{total_rows:,}, "
                    f"shard {shard_idx}/{len(task_dirs)}, "
                    f"elapsed {elapsed / 60:.1f} min, "
                    f"overall rate {format_rate(pos, elapsed)}, "
                    f"recent rate {format_rate(interval_rows, interval_elapsed)}"
                )

                last_log_time = current_time
                last_log_rows = pos

    if pos != total_rows:
        raise RuntimeError(
            f"Bucket {bucket}: wrote {pos} rows but expected {total_rows}"
        )

    out.flush()
    del out

    os.replace(tmp_final_path, final_path)

    elapsed = time.time() - t0

    log(
        f"REDUCE bucket {bucket} done. "
        f"Wrote {final_path}, shape=({total_rows:,}, {args.row_bytes}), "
        f"size={total_rows * args.row_bytes / 1e9:.3f} GB, "
        f"time={elapsed / 60:.1f} min, "
        f"rate={format_rate(total_rows, elapsed)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_map = sub.add_parser("map")
    p_map.add_argument("--input-dir", required=True)
    p_map.add_argument("--tmp-dir", required=True)
    p_map.add_argument("--n-buckets", type=int, required=True)
    p_map.add_argument("--task-id", type=int, required=True)
    p_map.add_argument("--n-tasks", type=int, required=True)
    p_map.add_argument("--row-bytes", type=int, default=128)
    p_map.add_argument("--chunk-rows", type=int, default=200_000)
    p_map.add_argument("--seed", type=int, default=0)
    p_map.add_argument("--overwrite-task-tmp", action="store_true")
    p_map.add_argument("--log-interval-seconds", type=int, default=60)
    p_map.set_defaults(func=map_stage)

    p_reduce = sub.add_parser("reduce")
    p_reduce.add_argument("--tmp-dir", required=True)
    p_reduce.add_argument("--output-dir", required=True)
    p_reduce.add_argument("--n-buckets", type=int, required=True)
    p_reduce.add_argument("--bucket-id", type=int, required=True)
    p_reduce.add_argument("--row-bytes", type=int, default=128)
    p_reduce.add_argument("--copy-rows", type=int, default=1_000_000)
    p_reduce.add_argument("--overwrite-output", action="store_true")
    p_reduce.add_argument("--log-interval-seconds", type=int, default=60)
    p_reduce.set_defaults(func=reduce_stage)

    args = parser.parse_args()

    try:
        args.func(args)
    except Exception as e:
        log(f"ERROR: {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
