#!/usr/bin/env python3
"""Python port of the dataviz skill's validate_palette.js.

The bundled validator is an ES module and this machine has no node, so the six
checks are reimplemented here rather than skipped. Thresholds, the Machado CVD
matrices and the OKLab math are copied from the JS source so the two agree.

    python3 scripts/validate_palette.py "#076678,#af3a03,..." --surface "#ffffff"
"""
from __future__ import annotations

import argparse
import math

BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}   # OKLCH L
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 8.0, 6.0                        # OKLab dE x100, adjacent
NORMAL_FLOOR = 15.0                                     # worst pair, normal vision
CONTRAST_MIN = 3.0                                      # WCAG vs surface

MACHADO = {
    "protan": [[0.152286, 1.052583, -0.204868],
               [0.114503, 0.786281, 0.099216],
               [-0.003882, -0.048116, 1.051998]],
    "deutan": [[0.367322, 0.860646, -0.227968],
               [0.280085, 0.672501, 0.047413],
               [-0.011820, 0.042940, 0.968881]],
    "tritan": [[1.255528, -0.076749, -0.178779],
               [-0.078411, 0.930809, 0.147602],
               [0.004733, 0.691367, 0.303900]],
}


def _s2lin(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def lin(h: str):
    h = h.strip().lstrip("#")
    return [_s2lin(int(h[i:i + 2], 16) / 255) for i in (0, 2, 4)]


def rel_lum(h: str) -> float:
    r, g, b = lin(h)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((rel_lum(a), rel_lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def oklab_from_lin(rgb):
    r, g, b = rgb
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def oklch(h: str):
    L, a, b = oklab_from_lin(lin(h))
    return L, math.hypot(a, b)


def simulate(h: str, kind: str):
    r, g, b = lin(h)
    M = MACHADO[kind]
    return [min(1.0, max(0.0, M[i][0] * r + M[i][1] * g + M[i][2] * b))
            for i in range(3)]


def delta_e(h1: str, h2: str, kind: str | None = None) -> float:
    a = oklab_from_lin(simulate(h1, kind) if kind else lin(h1))
    b = oklab_from_lin(simulate(h2, kind) if kind else lin(h2))
    return 100 * math.dist(a, b)


def validate(palette, mode="light", surface=None, pairs="adjacent", quiet=False):
    surface = surface or ("#fcfcfb" if mode == "light" else "#1a1a19")
    lo, hi = BAND[mode]
    ok = True
    out = []

    off = [(c, round(oklch(c)[0], 3)) for c in palette
           if not (lo <= oklch(c)[0] <= hi)]
    ok &= not off
    out.append(("lightness band", not off,
                f"outside [{lo}, {hi}]: {off}" if off else f"all {len(palette)} in band"))

    lowc = [(c, round(oklch(c)[1], 3)) for c in palette if oklch(c)[1] < CHROMA_FLOOR]
    ok &= not lowc
    out.append(("chroma floor", not lowc,
                f"below {CHROMA_FLOOR} (reads gray): {lowc}" if lowc
                else f"all {len(palette)} >= {CHROMA_FLOOR}"))

    idx = ([(i, i + 1) for i in range(len(palette) - 1)] if pairs == "adjacent"
           else [(i, j) for i in range(len(palette)) for j in range(i + 1, len(palette))])
    worst_cvd, worst_pair = float("inf"), None
    for i, j in idx:
        d = min(delta_e(palette[i], palette[j], "protan"),
                delta_e(palette[i], palette[j], "deutan"))
        if d < worst_cvd:
            worst_cvd, worst_pair = d, (palette[i], palette[j])
    state = "pass" if worst_cvd >= CVD_TARGET else ("floor" if worst_cvd >= CVD_FLOOR else "fail")
    ok &= state != "fail"
    out.append((f"CVD separation ({pairs})", state,
                f"worst {worst_cvd:.1f} on {worst_pair} (target >= {CVD_TARGET}, floor {CVD_FLOOR})"))

    worst_n, worst_np = float("inf"), None
    for i, j in idx:
        d = delta_e(palette[i], palette[j])
        if d < worst_n:
            worst_n, worst_np = d, (palette[i], palette[j])
    ok &= worst_n >= NORMAL_FLOOR
    out.append(("normal-vision floor", worst_n >= NORMAL_FLOOR,
                f"worst {worst_n:.1f} on {worst_np} (floor {NORMAL_FLOOR})"))

    low = [(c, round(contrast(c, surface), 2)) for c in palette
           if contrast(c, surface) < CONTRAST_MIN]
    out.append(("contrast vs surface", "relief" if low else True,
                f"below {CONTRAST_MIN}:1 -> needs visible labels: {low}" if low
                else f"all {len(palette)} >= {CONTRAST_MIN}:1"))

    if not quiet:
        glyph = {True: "PASS", False: "FAIL", "pass": "PASS", "floor": "WARN",
                 "fail": "FAIL", "relief": "WARN"}
        print(f"  surface {surface}  mode {mode}  n={len(palette)}")
        for name, st, msg in out:
            print(f"  [{glyph[st]:>4}] {name:<26} {msg}")
    return ok, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("palette")
    ap.add_argument("--mode", default="light", choices=("light", "dark"))
    ap.add_argument("--surface", default=None)
    ap.add_argument("--pairs", default="adjacent", choices=("adjacent", "all"))
    a = ap.parse_args()
    good, _ = validate([c.strip() for c in a.palette.split(",") if c.strip()],
                       a.mode, a.surface, a.pairs)
    raise SystemExit(0 if good else 1)
