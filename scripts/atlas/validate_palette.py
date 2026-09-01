#!/usr/bin/env python3
"""Colorblind-separation + contrast checks for a categorical palette.

The dataviz skill ships a Node validator; this cluster has no node, so the same
six checks are implemented here rather than skipped. Distances are Euclidean in
OKLab x100, matching that validator's scale: CVD >= 8 is the target, the
normal-vision floor is 15 (a hard fail below), and contrast is WCAG against the
chart surface.
"""
import sys, math, itertools

def srgb_to_lin(c):
    c /= 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

def hex_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def oklab(h):
    r, g, b = (srgb_to_lin(v) for v in hex_rgb(h))
    l = 0.4122214708*r + 0.5363325363*g + 0.0514459929*b
    m = 0.2119034982*r + 0.6806995451*g + 0.1073969566*b
    s = 0.0883024619*r + 0.2817188376*g + 0.6299787005*b
    l_, m_, s_ = (v ** (1/3) if v > 0 else -((-v) ** (1/3)) for v in (l, m, s))
    return (0.2104542553*l_ + 0.7936177850*m_ - 0.0040720468*s_,
            1.9779984951*l_ - 2.4285922050*m_ + 0.4505937099*s_,
            0.0259040371*l_ + 0.7827717662*m_ - 0.8086757660*s_)

# Vienot-Brettel-Mollon dichromat simulation, applied in linear LMS.
_CVD = {
 'protan': ((0.0,2.02344,-2.52581),(0,1,0),(0,0,1)),
 'deutan': ((1,0,0),(0.494207,0.0,1.24827),(0,0,1)),
 'tritan': ((1,0,0),(0,1,0),(-0.395913,0.801109,0.0)),
}
def simulate(h, kind):
    r, g, b = (srgb_to_lin(v) for v in hex_rgb(h))
    L = 17.8824*r + 43.5161*g + 4.11935*b
    M = 3.45565*r + 27.1554*g + 3.86714*b
    S = 0.0299566*r + 0.184309*g + 1.46709*b
    m = _CVD[kind]
    L2 = m[0][0]*L + m[0][1]*M + m[0][2]*S
    M2 = m[1][0]*L + m[1][1]*M + m[1][2]*S
    S2 = m[2][0]*L + m[2][1]*M + m[2][2]*S
    r2 =  0.080944*L2 - 0.130504*M2 + 0.116721*S2
    g2 = -0.010248*L2 + 0.054019*M2 - 0.113614*S2
    b2 = -0.000365*L2 - 0.004121*M2 + 0.693513*S2
    def enc(v):
        v = max(0.0, min(1.0, v))
        v = 12.92*v if v <= 0.0031308 else 1.055*(v ** (1/2.4)) - 0.055
        return int(round(max(0, min(1, v)) * 255))
    return '#%02x%02x%02x' % (enc(r2), enc(g2), enc(b2))

def de(a, b):
    x, y = oklab(a), oklab(b)
    return 100 * math.dist(x, y)

def lum(h):
    r, g, b = (srgb_to_lin(v) for v in hex_rgb(h))
    return 0.2126*r + 0.7152*g + 0.0722*b

def contrast(a, b):
    la, lb = lum(a), lum(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)

pal = sys.argv[1].split(',')
surface = sys.argv[2] if len(sys.argv) > 2 else '#ffffff'
print(f"palette {pal}  surface {surface}")
ok = True
for a, b in itertools.combinations(pal, 2):
    n = de(a, b)
    row = [f"  {a} vs {b}  normal {n:6.1f}"]
    worst = n
    for kind in _CVD:
        d = de(simulate(a, kind), simulate(b, kind))
        worst = min(worst, d)
        row.append(f"{kind} {d:5.1f}")
    verdict = "PASS"
    if n < 15:
        verdict, ok = "FAIL normal<15", False
    elif worst < 8:
        verdict, ok = "FAIL cvd<8", False
    print("  ".join(row) + f"   -> {verdict}")
print()
for c in pal:
    cr = contrast(c, surface)
    print(f"  {c} vs surface contrast {cr:.2f}" + ("  WARN <3" if cr < 3 else ""))
print("\nRESULT:", "PASS" if ok else "FAIL")
