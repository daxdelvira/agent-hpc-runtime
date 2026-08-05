#!/usr/bin/env python3
"""Record-aligned truncation of a FASTA. Standalone because the inline version
in job_p1_real_uniprot.sh opened the destination "wb" and then read from it,
which raises io.UnsupportedOperation and left the 8 GB subset uncreated."""
import os
import sys

src, dst, gb = sys.argv[1], sys.argv[2], float(sys.argv[3])
target = int(gb * 1e9)
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
print(dst, os.path.getsize(dst), flush=True)
