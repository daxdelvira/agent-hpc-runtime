#!/usr/bin/env python3
"""Cut a size-matched subset out of a large FASTA.

TWO MODES, AND THE OBVIOUS ONE IS BIASED FOR UNIREF50
-----------------------------------------------------
`head` takes a prefix and trims back to the last record boundary. It is cheap,
it is what the first run used, and on UniRef50 it is BIASED -- which the
measurement itself caught:

    whole uniref50.fasta   16.94 GB / 38,794,121 records =  437 B/record
    its first 2 GB          2.00 GB /  1,000,810 records = 1998 B/record

The file is not in random order, so a prefix is drawn from a region whose
sequences are ~4.6x longer than the database average. That matters more here
than it would elsewhere, because pyhmmer's expansion ratio is driven by
PER-RECORD overhead: the same prefix measured 1.148x expansion where the whole
database measures 2.13x. A prefix is a different workload wearing the same name.

`stride` takes every Nth record instead, so the subset's length distribution
matches the whole file, at the cost of reading all of it. Use it for anything
whose answer depends on record structure -- which, for this consumer, is
everything.
"""
from __future__ import annotations

import os
import sys


def head_subset(src: str, dst: str, target: int) -> None:
    with open(src, "rb") as fi, open(dst, "w+b") as fo:
        left = target
        while left > 0:
            b = fi.read(min(1 << 24, left))
            if not b:
                break
            fo.write(b)
            left -= len(b)
        fo.flush()
        size = fo.tell()
        tail = 1 << 20
        start = max(0, size - tail)
        fo.seek(start)
        chunk = fo.read()
        cut = chunk.rfind(b"\n>")
        if cut >= 0:
            fo.truncate(start + cut + 1)


def stride_subset(src: str, dst: str, target: int, stride: int) -> None:
    """Every `stride`-th record, until `target` bytes are written.

    Streams line by line rather than materialising records, so it is O(1) in
    memory over a 17 GB input. `stride` should be about
    total_bytes / target_bytes, so the pass over the input finishes at roughly
    the moment the output fills. If the input runs out first the subset is
    simply smaller, and its true size is printed -- never silently padded.
    """
    written = 0
    keep = False
    idx = -1
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        for line in fi:
            if line.startswith(b">"):
                idx += 1
                keep = (idx % stride == 0)
                if keep and written >= target:
                    break
            if keep:
                fo.write(line)
                written += len(line)


def main(argv) -> int:
    if len(argv) < 4:
        print("usage: make_fasta_subset.py SRC DST GB [STRIDE]", file=sys.stderr)
        print("       STRIDE omitted or 1 -> head mode (biased on UniRef50)",
              file=sys.stderr)
        return 2
    src, dst, gb = argv[1], argv[2], float(argv[3])
    stride = int(argv[4]) if len(argv) > 4 else 1
    target = int(gb * 1e9)
    if stride > 1:
        stride_subset(src, dst, target, stride)
    else:
        head_subset(src, dst, target)
    mode = f"stride {stride}" if stride > 1 else "head"
    print(f"{dst} {os.path.getsize(dst)} ({mode})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
