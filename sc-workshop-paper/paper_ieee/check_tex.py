#!/usr/bin/env python3
"""Static consistency checker for paper_ieee/.

There is no LaTeX toolchain on this machine, so this stands in for a compile.
It checks the failure modes that a split-file LaTeX document actually hits:

  1. brace balance (per file, ignoring comments and \\{ \\})
  2. \\begin/\\end pairing (per file and across the whole document)
  3. \\label/\\ref closure: dangling refs, duplicate labels, unused labels
  4. non-ASCII characters (the preamble loads no inputenc/fontenc)
  5. tabular body column counts vs the column spec
  6. suspicious unescaped % (e.g. "50%") and unescaped _ outside math/texttt
  7. \\figspec call arity (macro is [opt]{}{} -- every figure depends on it)
  8. figures/tables missing \\caption or \\label
  9. commands used vs packages the authoritative preamble actually loads

Run:  python3 check_tex.py            (from paper_ieee/, or anywhere)
Exit code 0 = clean, 1 = at least one ERROR.
"""

import os
import re
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))

# Order matters only for readability of the report.
FILES = [
    "main.tex",
    "sections/00_macros.tex",
    "sections/00_abstract.tex",
    "sections/01_introduction.tex",
    "sections/02_background.tex",
    "sections/03_problem.tex",
    "sections/04_opportunities.tex",
    "sections/04b_evaluation.tex",
    "sections/05_conclusion.tex",
]

ERRORS = []
WARNINGS = []
INFO = []


def err(f, ln, msg):
    ERRORS.append("ERROR  %s:%s  %s" % (f, ln, msg))


def warn(f, ln, msg):
    WARNINGS.append("WARN   %s:%s  %s" % (f, ln, msg))


def info(msg):
    INFO.append("INFO   " + msg)


# --------------------------------------------------------------------------
# comment stripping: '%' starts a comment unless preceded by an odd number of
# backslashes. Newline is preserved so line numbers stay meaningful.
# --------------------------------------------------------------------------
def strip_comments(text):
    out = []
    for line in text.split("\n"):
        i = 0
        cut = None
        while i < len(line):
            if line[i] == "\\":
                i += 2
                continue
            if line[i] == "%":
                cut = i
                break
            i += 1
        out.append(line if cut is None else line[:cut])
    return "\n".join(out)


def lineno(text, pos):
    return text.count("\n", 0, pos) + 1


def unescaped(text, pos):
    """True if the char at pos is not preceded by an odd run of backslashes."""
    n = 0
    j = pos - 1
    while j >= 0 and text[j] == "\\":
        n += 1
        j -= 1
    return n % 2 == 0


# --------------------------------------------------------------------------
CMD_PACKAGE = {
    # command / environment  ->  package that must be loaded
    r"\xspace": "xspace",
    r"\includegraphics": "graphicx",
    r"\graphicspath": "graphicx",
    r"\toprule": "booktabs",
    r"\midrule": "booktabs",
    r"\bottomrule": "booktabs",
    r"\cmidrule": "booktabs",
    r"\text{": "amsmath",
    r"\textcolor": "xcolor",
    r"\url": "hyperref/url",
    r"\href": "hyperref",
    r"\subcaptionbox": "subcaption",
    r"\State": "algpseudocode",
    r"\EndFor": "algpseudocode",
    r"\EndIf": "algpseudocode",
    r"\EndWhile": "algpseudocode",
    r"\Comment": "algpseudocode",
    r"\Procedure": "algpseudocode",
    r"\STATE": "algorithmic",
    r"\ENDFOR": "algorithmic",
    r"\COMMENT": "algorithmic",
}
ENV_PACKAGE = {
    "algorithm": "algorithm(float)",
    "algorithmic": "algorithmic",
    "tabularx": "tabularx",
    "subfigure": "subcaption",
    "align": "amsmath",
    "equation": "latex-kernel",
    "figure": "latex-kernel",
    "figure*": "latex-kernel",
    "table": "latex-kernel",
    "tabular": "latex-kernel",
    "itemize": "latex-kernel",
    "enumerate": "latex-kernel",
    "abstract": "latex-kernel",
    "IEEEkeywords": "IEEEtran",
    "document": "latex-kernel",
}
LOADED = {
    "cite", "amsmath", "amssymb", "amsfonts", "algorithmic", "graphicx",
    "textcomp", "xcolor", "subcaption", "booktabs", "tabularx", "ragged2e",
    "latex-kernel", "IEEEtran",
}

labels = {}          # label -> (file, line)
refs = []            # (label, file, line)
env_stack_global = []
raw_by_file = {}
src_by_file = {}

# --------------------------------------------------------------------------
for rel in FILES:
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        err(rel, 0, "file listed in FILES does not exist")
        continue
    raw = open(path, encoding="utf-8", errors="replace").read()
    raw_by_file[rel] = raw
    src = strip_comments(raw)
    src_by_file[rel] = src

    # ---- 4. non-ASCII (checked on the RAW text, comments included) ---------
    for m in re.finditer(r"[^\x09\x0a\x20-\x7e]", raw):
        err(rel, lineno(raw, m.start()),
            "non-ASCII char U+%04X (%r); preamble loads no inputenc/fontenc"
            % (ord(m.group()), m.group()))

    # ---- 1. brace balance -------------------------------------------------
    depth = 0
    open_positions = []
    for i, ch in enumerate(src):
        if ch in "{}" and unescaped(src, i):
            if ch == "{":
                depth += 1
                open_positions.append(i)
            else:
                depth -= 1
                if depth < 0:
                    err(rel, lineno(src, i), "unmatched closing brace")
                    depth = 0
                else:
                    open_positions.pop()
    if depth != 0:
        for p in open_positions:
            err(rel, lineno(src, p), "unclosed opening brace")

    # ---- 2. environments --------------------------------------------------
    stack = []
    for m in re.finditer(r"\\(begin|end)\s*\{([^}]*)\}", src):
        kind, name = m.group(1), m.group(2)
        ln = lineno(src, m.start())
        if name not in ENV_PACKAGE:
            warn(rel, ln, "environment '%s' not in the known-environment table"
                 % name)
        else:
            pkg = ENV_PACKAGE[name]
            if pkg not in LOADED:
                err(rel, ln, "environment '%s' requires package '%s', which the "
                    "authoritative preamble does NOT load" % (name, pkg))
        if kind == "begin":
            stack.append((name, ln))
            env_stack_global.append((rel, name, ln))
        else:
            if not stack:
                err(rel, ln, "\\end{%s} with no matching \\begin in this file"
                    % name)
            else:
                nm, oln = stack.pop()
                if nm != name:
                    err(rel, ln, "\\end{%s} closes \\begin{%s} opened at line %d"
                        % (name, nm, oln))
            if env_stack_global and env_stack_global[-1][1] == name:
                env_stack_global.pop()
    for nm, oln in stack:
        err(rel, oln, "\\begin{%s} never closed in this file" % nm)

    # ---- 3. labels and refs ----------------------------------------------
    for m in re.finditer(r"\\label\s*\{([^}]*)\}", src):
        lab = m.group(1)
        ln = lineno(src, m.start())
        if lab in labels:
            err(rel, ln, "duplicate \\label{%s} (first at %s:%d)"
                % (lab, labels[lab][0], labels[lab][1]))
        else:
            labels[lab] = (rel, ln)
    for m in re.finditer(r"\\(?:ref|eqref|autoref|pageref|cref|Cref)\s*\{([^}]*)\}",
                         src):
        refs.append((m.group(1), rel, lineno(src, m.start())))

    # ---- 6a. suspicious unescaped % --------------------------------------
    # every unescaped % is a comment start; flag the ones that look like a
    # percent sign that should have been \%
    # Only the FIRST unescaped % on a line actually starts a comment; any %
    # after it is already inside the comment and is harmless.
    off = 0
    for lnum, line in enumerate(raw.split("\n"), start=1):
        k = 0
        while k < len(line):
            if line[k] == "\\":
                k += 2
                continue
            if line[k] == "%":
                if k > 0 and (line[k - 1].isdigit() or line[k - 1] in ")]"):
                    err(rel, lnum,
                        "'%s%%' starts a comment mid-line; should it be \\%%? "
                        "(rest of line dropped: %r)"
                        % (line[k - 1], line[k + 1:][:40]))
                break
            k += 1
        off += len(line) + 1

    # ---- 6b. unescaped _ outside math / \texttt / file arguments ----------
    # build a mask of protected spans
    protected = [False] * len(src)

    def protect(a, b):
        for k in range(a, min(b, len(src))):
            protected[k] = True

    # $...$ and $$...$$
    i = 0
    while i < len(src):
        if src[i] == "$" and unescaped(src, i):
            j = i + 1
            if j < len(src) and src[j] == "$":
                j += 1
            while j < len(src):
                if src[j] == "$" and unescaped(src, j):
                    break
                j += 1
            protect(i, j + 1)
            i = j + 1
        else:
            i += 1
    # \[...\], equation/align environments
    for m in re.finditer(r"\\\[.*?\\\]", src, re.S):
        protect(m.start(), m.end())
    for envname in ("equation", "align", "algorithmic"):
        for m in re.finditer(r"\\begin\{%s\*?\}.*?\\end\{%s\*?\}"
                             % (envname, envname), src, re.S):
            protect(m.start(), m.end())
    # \texttt{...}, \verb, and file-name arguments
    for m in re.finditer(r"\\(?:texttt|verb|input|include|bibliography|"
                         r"graphicspath|label|ref|eqref|cite)\s*[\{|][^\}|]*[\}|]",
                         src):
        protect(m.start(), m.end())

    for i, ch in enumerate(src):
        if ch == "_" and unescaped(src, i) and not protected[i]:
            err(rel, lineno(src, i),
                "unescaped '_' outside math/texttt (context: %r)"
                % src[max(0, i - 25):i + 25].replace("\n", " "))

    # ---- 5. tabular column counts ----------------------------------------
    for m in re.finditer(r"\\begin\{tabular\}\s*(?:\[[^\]]*\])?\s*\{",
                         src):
        # find the matching close brace of the column spec
        s = m.end() - 1
        d = 0
        for k in range(s, len(src)):
            if src[k] == "{" and unescaped(src, k):
                d += 1
            elif src[k] == "}" and unescaped(src, k):
                d -= 1
                if d == 0:
                    spec_end = k
                    break
        spec = src[m.end():spec_end]
        body_start = spec_end + 1
        endm = re.compile(r"\\end\{tabular\}").search(src, body_start)
        body = src[body_start:endm.start()]
        start_ln = lineno(src, m.start())

        # count columns in the spec
        t = spec
        t = re.sub(r"@\{[^{}]*\}", "", t)
        t = re.sub(r"!\{[^{}]*\}", "", t)
        t = re.sub(r"[pmb]\{[^{}]*\}", "P", t)
        t = re.sub(r">\{[^{}]*\}", "", t)
        t = re.sub(r"<\{[^{}]*\}", "", t)
        t = t.replace("|", "").replace(" ", "")
        ncols = sum(1 for c in t if c in "lcrPXS")

        # split body into rows on top-level \\
        rows = []
        cur = []
        d = 0
        k = 0
        while k < len(body):
            if body[k] == "{" and unescaped(body, k):
                d += 1
            elif body[k] == "}" and unescaped(body, k):
                d -= 1
            if body.startswith("\\\\", k) and d == 0:
                rows.append("".join(cur))
                cur = []
                k += 2
                continue
            cur.append(body[k])
            k += 1
        if "".join(cur).strip():
            rows.append("".join(cur))

        for r in rows:
            bare = r.strip()
            stripped = re.sub(
                r"\\(toprule|midrule|bottomrule|hline|cmidrule|addlinespace)"
                r"(\[[^\]]*\])?(\{[^}]*\})?", "", bare).strip()
            if not stripped:
                continue
            d = 0
            amps = 0
            for k, ch in enumerate(stripped):
                if ch == "{" and unescaped(stripped, k):
                    d += 1
                elif ch == "}" and unescaped(stripped, k):
                    d -= 1
                elif ch == "&" and d == 0 and unescaped(stripped, k):
                    amps += 1
            got = amps + 1
            if got != ncols:
                err(rel, start_ln,
                    "tabular spec {%s} declares %d columns but row has %d: %r"
                    % (spec, ncols, got, stripped[:70]))
        info("%s:%d tabular {%s} -> %d cols, %d body rows checked"
             % (rel, start_ln, spec, ncols, len(
                 [r for r in rows if re.sub(
                     r"\\(toprule|midrule|bottomrule|hline)", "", r).strip()])))

    # ---- 7. \figspec arity ------------------------------------------------
    for m in re.finditer(r"\\figspec", src):
        k = m.end()
        if k < len(src) and src[k] == "[":
            d = 0
            while k < len(src):
                if src[k] == "[":
                    d += 1
                elif src[k] == "]":
                    d -= 1
                    if d == 0:
                        k += 1
                        break
                k += 1
        groups = 0
        while groups < 2:
            while k < len(src) and src[k] in " \n\t":
                k += 1
            if k >= len(src) or src[k] != "{":
                break
            d = 0
            while k < len(src):
                if src[k] == "{" and unescaped(src, k):
                    d += 1
                elif src[k] == "}" and unescaped(src, k):
                    d -= 1
                    if d == 0:
                        k += 1
                        break
                k += 1
            groups += 1
        if groups != 2 or "newcommand" in src[max(0, m.start() - 20):m.start()]:
            if "newcommand" not in src[max(0, m.start() - 20):m.start()]:
                err(rel, lineno(src, m.start()),
                    "\\figspec has %d mandatory args, expected 2" % groups)

    # ---- 8. floats without caption/label ---------------------------------
    for m in re.finditer(r"\\begin\{(figure\*?|table\*?)\}(.*?)\\end\{\1\}",
                         src, re.S):
        body = m.group(2)
        ln = lineno(src, m.start())
        if "\\caption" not in body:
            warn(rel, ln, "%s float has no \\caption" % m.group(1))
        if "\\label" not in body:
            warn(rel, ln, "%s float has no \\label" % m.group(1))

    # ---- 9. command/package gap ------------------------------------------
    for cmd, pkg in CMD_PACKAGE.items():
        for m in re.finditer(re.escape(cmd) + r"(?![a-zA-Z])"
                             if not cmd.endswith("{") else re.escape(cmd), src):
            if pkg not in LOADED:
                err(rel, lineno(src, m.start()),
                    "'%s' requires package '%s', NOT loaded by the "
                    "authoritative preamble" % (cmd, pkg))

# --------------------------------------------------------------------------
# 3. cross-file label closure
# --------------------------------------------------------------------------
for lab, f, ln in refs:
    if lab not in labels:
        err(f, ln, "dangling \\ref{%s} -- no \\label anywhere in the document"
            % lab)
referenced = set(l for l, _, _ in refs)
for lab, (f, ln) in sorted(labels.items()):
    if lab not in referenced:
        warn(f, ln, "\\label{%s} is never referenced" % lab)

if env_stack_global:
    for f, n, ln in env_stack_global:
        err(f, ln, "environment '%s' left open across file boundary" % n)

# --------------------------------------------------------------------------
print("=" * 74)
print("paper_ieee static check -- %d files" % len(raw_by_file))
print("=" * 74)
print("labels defined: %d   refs made: %d   distinct refs: %d"
      % (len(labels), len(refs), len(referenced)))
print()
for line in INFO:
    print(line)
print()
for line in WARNINGS:
    print(line)
print()
for line in ERRORS:
    print(line)
print()
print("-" * 74)
print("ERRORS: %d    WARNINGS: %d" % (len(ERRORS), len(WARNINGS)))
print("-" * 74)
sys.exit(1 if ERRORS else 0)
