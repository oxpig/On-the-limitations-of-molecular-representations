#!/usr/bin/env python3

# ---------- Imports and global variables ----------

import argparse
import os
import multiprocessing as mp
from typing import Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

_FPGEN = None  # Global fingerprint generator object.


# ---------- IO ----------

def count_data_rows_tsv(path: str) -> int:
    """
    Count number of rows in the file.
    Assumes no header row.
    """
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def read_smiles_chunks(path: str, chunk_rows: int) -> Iterable[List[str]]:
    """
    Yield SMILES from the first TSV column in chunks. 
    Assumes no header.
    """
    for df in pd.read_csv(
        path,
        sep="\t",
        header=None,
        usecols=[0],
        dtype=str,
        chunksize=chunk_rows,
    ):
        yield df.iloc[:, 0].fillna("").tolist()


def rebatch(smiles_chunks: Iterable[List[str]], batch_size: int) -> Iterable[List[str]]:
    """
    Convert larger chunks into fixed-size batches.
    """
    buf: List[str] = []

    for chunk in smiles_chunks:
        buf.extend(chunk)

        while len(buf) >= batch_size:
            yield buf[:batch_size]
            buf = buf[batch_size:]

    if buf:
        yield buf


# ---------- Fingerprinting: worker side ----------

def init_morgan_worker(fpsize: int, radius: int, chiral: bool) -> None:
    """ 
    Initialize a Morgan fingerprint worker.
    """
    global _FPGEN
    _FPGEN = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=fpsize,
        includeChirality=chiral,
    )


def init_atompair_worker(fpsize: int, chiral: bool) -> None:
    """
    Initialize an atom pair fingerprint worker.
    """
    global _FPGEN
    _FPGEN = rdFingerprintGenerator.GetAtomPairGenerator(
        fpSize=fpsize,
        includeChirality=chiral,
    )


def init_torsion_worker(fpsize: int, chiral: bool) -> None:
    """ 
    Initialize a topological torsion fingerprint worker.
    """
    global _FPGEN
    _FPGEN = rdFingerprintGenerator.GetTopologicalTorsionGenerator(
        fpSize=fpsize,
        includeChirality=chiral,
    )


def init_rdkit_worker(fpsize: int) -> None:
    """ 
    Initialize an rdkit fingerprint worker.
    """
    global _FPGEN
    _FPGEN = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=fpsize)


def init_featmorgan_worker(fpsize: int, radius: int, chiral: bool) -> None:
    """ 
    Initialize a feature Morgan fingerprint worker.
    """
    global _FPGEN
    _FPGEN = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=fpsize,
        atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
        includeChirality=chiral,
    )


def fingerprint_one_smiles(smiles: str) -> np.ndarray:
    """ 
    Converts one SMILES into an RDKit Mol object and generates the 
    selected fingerprint. 
    """
    global _FPGEN

    mol = Chem.MolFromSmiles(smiles)
    fp  = _FPGEN.GetFingerprintAsNumPy(mol).astype(np.uint8, copy=False)
    return fp


def fingerprint_batch(smiles_batch: List[str], fpsize: int) -> np.ndarray:
    """
    Return an unpacked uint8 array of shape:
    (batch_size, fpsize) with values 0/1.

    Packing is done later in the parent process before writing.
    """
    out     = np.zeros((len(smiles_batch), fpsize), dtype=np.uint8)

    for i, s in enumerate(smiles_batch):
        fp        = fingerprint_one_smiles(s)
        out[i, :] = fp

    return out


def fingerprint_batch_worker(smiles_batch: List[str]) -> np.ndarray:
    """
    Worker for the fingerprint_batch method.
    """
    global _FPGEN

    fpsize = _FPGEN.GetOptions().fpSize
    return fingerprint_batch(smiles_batch, fpsize)


# ---------- Output ----------

def ensure_out_dir(out_prefix: str) -> None:
    """
    Ensure the output directory exists by making it.
    """
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)


def packed_fpsize_from_fpsize(fpsize: int) -> int:
    """
    Number of uint8 columns needed to store fpsize bits.
    + 7 allows us to take the ceiling if the selected bit-size 
    isn't a multiple of 8. We're using 1024 in this pipeline, 
    so there's no change.
    """
    return (fpsize + 7) // 8


def open_fps_memmap(out_prefix: str, n_rows: int, fpsize: int) -> Tuple[str, np.memmap]:
    """
    Open output .npy memmap.

    Filename is intentionally unchanged: <out_prefix>_fps.npy
    But the stored array is bitpacked uint8 with shape: (n_rows, ceil(fpsize / 8))
    """
    out_path      = out_prefix + "_fps.npy"
    packed_fpsize = packed_fpsize_from_fpsize(fpsize)

    mm = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.uint8,
        shape=(n_rows, packed_fpsize),
    )

    return out_path, mm


def pack_fps_batch(fps_batch: np.ndarray) -> np.ndarray:
    """
    Bitpack a batch of 0/1 fingerprints.

    bitorder="big" preserves lexicographic order of the original bit vectors
    when comparing packed uint8 rows byte-by-byte.
    """
    return np.packbits(fps_batch, axis=1, bitorder="big",)


def write_batch(mm: np.memmap, offset: int, fps_batch: np.ndarray) -> int:
    """
    Pack and write one fingerprint batch.

    Input fps_batch shape: (batch_size, fpsize)
    Written packed shape:  (batch_size, ceil(fpsize / 8))
    """
    n      = fps_batch.shape[0]
    packed = pack_fps_batch(fps_batch)

    mm[offset: offset + n, :] = packed

    return offset + n


# ---------- Orchestration ----------

def make_pool(nprocs: int, initializer, initargs):
    """
    Make a pool of worker processes to parallelize
    fingerprint generation.
    """
    ctx = mp.get_context("spawn")

    return ctx.Pool(
        processes=nprocs,
        initializer=initializer,
        initargs=initargs,
    )


def stream_fps_batches(in_path: str, chunk_rows: int, batch_size: int, pool,) -> Iterable[np.ndarray]:
    """
    Split the processing of fingerprint chunks among multiple processes and retain
    the order.
    """
    smiles_chunks = read_smiles_chunks(in_path, chunk_rows=chunk_rows)
    batches       = rebatch(smiles_chunks, batch_size=batch_size)

    # imap preserves input order.
    yield from pool.imap(fingerprint_batch_worker, batches, chunksize=1)


def build_fps_memmap(
    in_path: str,
    out_prefix: str,
    fpsize: int,
    nprocs: int,
    chunk_rows: int,
    batch_size: int,
    init_worker,
    init_args,
) -> str:
    """ Set up the fingerprint nunpy memory maps"""
    n_rows = count_data_rows_tsv(in_path)

    if n_rows == 0:
        raise RuntimeError(f"No rows found in: {in_path}")

    ensure_out_dir(out_prefix)

    fps_path, mm = open_fps_memmap(
        out_prefix=out_prefix,
        n_rows=n_rows,
        fpsize=fpsize,
    )

    offset = 0
    with make_pool(
        nprocs=nprocs,
        initializer=init_worker,
        initargs=init_args,
    ) as pool:
        for fps_batch in stream_fps_batches(
            in_path=in_path,
            chunk_rows=chunk_rows,
            batch_size=batch_size,
            pool=pool,
        ):
            offset = write_batch(mm, offset, fps_batch)

    if offset != n_rows:
        raise RuntimeError(
            f"Wrote {offset} rows but expected {n_rows}. "
            "Check parsing/chunking."
        )

    mm.flush()

    return fps_path


def str2bool(v):
    """
    Parse arguments into bools for the chirality
    flag.
    """
    if isinstance(v, bool):
        return v

    v = str(v).strip().lower()

    if v in {"true", "t", "1", "yes", "y"}:
        return True

    if v in {"false", "f", "0", "no", "n"}:
        return False

    raise argparse.ArgumentTypeError(f"Expected true/false, got: {v}")


def parse_args() -> argparse.Namespace:
    """
    Argument parsing.
    """
    ap = argparse.ArgumentParser()

    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument(
        "--out",
        dest="out_path",
        required=True,
        help="Output prefix. Writes <prefix>_fps.npy",
    )

    ap.add_argument(
        "--fp",
        dest="fingerprint",
        required=True,
        choices=["morgan", "atompair", "torsion", "featmorgan", "rdkit"],
    )

    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--fpsize", type=int, default=1024)
    ap.add_argument("--chirality", required=True, type=str2bool)

    ap.add_argument(
        "--nprocs",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )

    ap.add_argument("--chunk-rows", type=int, default=200_000)
    ap.add_argument("--batch-size", type=int, default=10_000)

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.fingerprint == "morgan":
        init_worker = init_morgan_worker
        init_args   = (args.fpsize, args.radius, args.chirality)

    elif args.fingerprint == "atompair":
        init_worker = init_atompair_worker
        init_args   = (args.fpsize, args.chirality)

    elif args.fingerprint == "torsion":
        init_worker = init_torsion_worker
        init_args   = (args.fpsize, args.chirality)

    elif args.fingerprint == "featmorgan":
        init_worker = init_featmorgan_worker
        init_args   = (args.fpsize, args.radius, args.chirality)

    elif args.fingerprint == "rdkit":
        init_worker = init_rdkit_worker
        init_args   = (args.fpsize,)

    else:
        raise ValueError(f"Unsupported fingerprint type: {args.fingerprint}")

    packed_fpsize = packed_fpsize_from_fpsize(args.fpsize)

    print(
        "[args] "
        f"fp={args.fingerprint} "
        f"fpsize={args.fpsize} "
        f"packed_fpsize={packed_fpsize} "
        f"radius={args.radius} "
        f"chirality={args.chirality}"
    )

    fps_path = build_fps_memmap(
        in_path=args.in_path,
        out_prefix=args.out_path,
        fpsize=args.fpsize,
        nprocs=max(1, args.nprocs),
        chunk_rows=args.chunk_rows,
        batch_size=args.batch_size,
        init_worker=init_worker,
        init_args=init_args,
    )

    print(f"[ok] wrote bitpacked fingerprints: {fps_path}")


if __name__ == "__main__":
    main()
