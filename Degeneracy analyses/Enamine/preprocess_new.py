#!/usr/bin/env python3
import os
import re
import sys
import csv
import hashlib
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from rdkit import Chem

def normalize_smiles_batch(smiles_list):
    """
    Normalize a batch of raw CXSMILES to canonical isomeric SMILES.
    Returns list of normalized SMILES (or None for invalid rows).
    """
    out = []
    for raw in smiles_list:
        if not raw:
            out.append(None)
            continue

        mol = Chem.MolFromSmiles(raw)
        if mol is None:
            out.append(None)
            continue

        # mol = Chem.RemoveHs(mol)

        # Writing back strips CXCSMILES annotations. Canonicalize.
        norm = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        out.append(norm)

    return out

def iter_firstcol_tsv(path, chunk_size_lines: int):
    """
    Read the first column of the tsv file in chunks.
    """
    with open(path, "r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        chunk = []
        for row in reader:
            if not row:
                continue
            chunk.append(row[0])
            if len(chunk) >= chunk_size_lines:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

def shard_index(s: str, n_shards: int) -> int:
    """
    Hashes a smiles string into an integer and assign to a
    shard index. Ensures identical canonicalized smiles map
    to the same shard.
    """
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little") % n_shards

def append_lines(path: Path, lines):
    """
    Append lines to the given txt file.
    """
    if not lines:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")

def write_shards(normalized_list, shard_dir: Path, n_shards: int) -> int:
    """
    Hashes SMILES into buckets, ensuring that identical SMILES
    are in the same bucket and writes shards.
    """
    buckets = [[] for _ in range(n_shards)]
    written = 0
    for s in normalized_list:
        if not s:
            continue
        idx = shard_index(s, n_shards)
        buckets[idx].append(s)
        written += 1

    for i, lines in enumerate(buckets):
        if lines:
            append_lines(shard_dir / f"shard_{i:05d}.txt", lines)
    return written

def phase_a_normalize_and_shard(in_tsv: Path,
    shard_dir: Path,
    n_shards: int,
    chunk_size_lines: int,
    max_workers: int,
    batches_in_flight: int,
    expected_lines: int = 0,
    log_every_chunks: int = 10,
):
    """
    Reads in a set of shard files containing SMILES strings
    from a given directory. Strips enhanced stereochemistry
    annotations and writes to shards.
    """
    shard_dir.mkdir(parents=True, exist_ok=True)

    for p in shard_dir.glob("shard_*.txt"):
        p.unlink()

    futures          = []
    total_in         = 0
    total_out        = 0
    chunks_submitted = 0
    chunks_done      = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for chunk in iter_firstcol_tsv(in_tsv, chunk_size_lines):
            total_in += len(chunk)
            chunks_submitted += 1
            futures.append(ex.submit(normalize_smiles_batch, chunk))

            if len(futures) >= batches_in_flight:
                for fut in as_completed(futures):
                    done = fut.result()
                    total_out += write_shards(done, shard_dir, n_shards)
                    futures.remove(fut)
                    chunks_done += 1

                    if chunks_done % log_every_chunks == 0:
                        elapsed = time.time() - t0
                        rate = total_in / elapsed if elapsed > 0 else 0.0
                        if expected_lines > 0:
                            pct = 100.0 * total_in / expected_lines
                            print(
                                f"[Phase A] {total_in:,}/{expected_lines:,} lines read "
                                f"({pct:.2f}%) | chunks_done={chunks_done:,} "
                                f"| normalized_written={total_out:,} "
                                f"| rate={rate:,.0f} lines/s",
                                file=sys.stderr,
                                flush=True,
                            )
                        else:
                            print(
                                f"[Phase A] {total_in:,} lines read "
                                f"| chunks_done={chunks_done:,} "
                                f"| normalized_written={total_out:,} "
                                f"| rate={rate:,.0f} lines/s",
                                file=sys.stderr,
                                flush=True,
                            )
                    break

        for fut in as_completed(futures):
            done = fut.result()
            total_out += write_shards(done, shard_dir, n_shards)
            chunks_done += 1

            if chunks_done % log_every_chunks == 0 or chunks_done == chunks_submitted:
                elapsed = time.time() - t0
                rate = total_in / elapsed if elapsed > 0 else 0.0
                if expected_lines > 0:
                    pct = 100.0 * total_in / expected_lines
                    print(
                        f"[Phase A] {total_in:,}/{expected_lines:,} lines read "
                        f"({pct:.2f}%) | chunks_done={chunks_done:,} "
                        f"| normalized_written={total_out:,} "
                        f"| rate={rate:,.0f} lines/s",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    print(
                        f"[Phase A] {total_in:,} lines read "
                        f"| chunks_done={chunks_done:,} "
                        f"| normalized_written={total_out:,} "
                        f"| rate={rate:,.0f} lines/s",
                        file=sys.stderr,
                        flush=True,
                    )

    print(
        f"[Phase A] Done. Read {total_in:,} lines, wrote {total_out:,} normalized SMILES.",
        file=sys.stderr,
        flush=True,
    )

def dedupe_one_shard(shard_path: str, out_path: str, sort_dedupe: bool):
    """
    Deduplicate one SMILES shard file. Returns the count of unique
    molecules in the shard.
    """
    shard_path = Path(shard_path)
    out_path   = Path(out_path)
    seen       = set()

    with open(shard_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.rstrip("\n")
            if s:
                seen.add(s)

    uniq = sorted(seen) if sort_dedupe else list(seen)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        if uniq:
            out.write("\n".join(uniq))
            out.write("\n")

    return shard_path.name, len(uniq)

def phase_b_dedupe_shards(shard_dir: Path, uniq_dir: Path, max_workers: int, sort_dedupe: bool):
    """
    Second phase. Once the SMILES shards have been written we deduplicate 
    identical canonicalized SMILES.
    """
    uniq_dir.mkdir(parents=True, exist_ok=True)
    shard_files = sorted(shard_dir.glob("shard_*.txt"))
    if not shard_files:
        raise RuntimeError(f"No shard files found in {shard_dir}")

    for p in uniq_dir.glob("uniq_*.txt"):
        p.unlink()

    total_shards = len(shard_files)
    done_shards  = 0
    total_uniq   = 0

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for sf in shard_files:
            uf = uniq_dir / sf.name.replace("shard_", "uniq_")
            futures.append(ex.submit(dedupe_one_shard, str(sf), str(uf), sort_dedupe))

        for fut in as_completed(futures):
            name, n = fut.result()
            total_uniq += n
            done_shards += 1

            if done_shards % 100 == 0 or done_shards == total_shards:
                pct = 100.0 * done_shards / total_shards
                print(
                    f"[Phase B] {done_shards:,}/{total_shards:,} shards deduped "
                    f"({pct:.2f}%) | uniq_total={total_uniq:,} | last={name}",
                    file=sys.stderr,
                    flush=True,
                )

    print(f"[Phase B] Wrote {total_uniq:,} unique normalized SMILES across shards.", file=sys.stderr, flush=True)

def concat_uniqs(uniq_dir: Path, out_file: Path):
    """
    Concatenate all unique shards back into one 
    file.
    """
    uniq_files = sorted(uniq_dir.glob("uniq_*.txt"))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as out:
        for uf in uniq_files:
            with open(uf, "r", encoding="utf-8") as f:
                for line in f:
                    out.write(line)
    print(f"[Done] Concatenated to {out_file}", file=sys.stderr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_tsv", required=True, help="Input TSV path (SMILES in column 1)")
    ap.add_argument("--out", required=True, help="Output file of unique normalized SMILES")
    ap.add_argument("--workdir", required=True, help="Working directory for shards")
    ap.add_argument("--n_shards", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=200_000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--inflight", type=int, default=4)
    ap.add_argument("--sort_dedupe", action="store_true")
    ap.add_argument("--expected_lines", type=int, default=0)
    ap.add_argument("--log_every_chunks", type=int, default=10)
    args = ap.parse_args()

    in_tsv    = Path(args.in_tsv)
    out_file  = Path(args.out)
    workdir   = Path(args.workdir)
    shard_dir = workdir / "shards"
    uniq_dir  = workdir / "uniq"

    phase_a_normalize_and_shard(
        in_tsv=in_tsv,
        shard_dir=shard_dir,
        n_shards=args.n_shards,
        chunk_size_lines=args.chunk,
        max_workers=args.workers,
        batches_in_flight=args.inflight,
        expected_lines=args.expected_lines,
        log_every_chunks=args.log_every_chunks,
    )
    phase_b_dedupe_shards(
        shard_dir=shard_dir,
        uniq_dir=uniq_dir,
        max_workers=args.workers,
        sort_dedupe=args.sort_dedupe,
    )
    concat_uniqs(uniq_dir=uniq_dir, out_file=out_file)

if __name__ == "__main__":
    main()
