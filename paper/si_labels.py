# Copy labels from the full arXiv build so split versions print the full version's numbers.
# Usage: python3 si_labels.py neurips_appendix_body.tex neurips_arxiv.aux > neurips_si_labels.tex
#        python3 si_labels.py --main neurips_appendix_body.tex neurips_arxiv.aux > neurips_main_labels.tex
# Default: the \newlabel entries for labels defined in the appendix (for the appendix-free version).
# --main: the entries for labels defined in the main text (for the standalone SI), plus \SIcounters,
# which starts the figure, table, equation, and assumption counters where the main text leaves them.
# Hyperlink anchors are blanked, so the references print "Section C.2" without links to absent pages.
import re, sys

def close(s, i):
    """Index of the brace closing the group that opens at s[i]."""
    depth, j = 0, i
    while j < len(s):
        if s[j] == "\\": j += 2; continue
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        if depth == 0: return j
        j += 1
    raise ValueError(s[i:i + 80])

main = sys.argv[1] == "--main"
body, aux = (open(f).read() for f in sys.argv[1 + main:3 + main])
si = set(re.findall(r"\\label\{([^}]*)\}", re.sub(r"(?<!\\)%.*", "", body)))
found, start = set(), {"figure": 0, "table": 0, "equation": 0, "assumption": 0}
for line in aux.splitlines():
    m = re.match(r"\\newlabel\{([^}]*)\}\{", line)
    if not m or (m.group(1) in si) == main: continue
    k = m.end() - 1; inner = line[k + 1:close(line, k)]
    parts, i = [], 0
    while i < len(inner) and inner[i] == "{":
        e = close(inner, i); parts.append(inner[i + 1:e]); i = e + 1
    if len(parts) >= 4:
        kind = parts[3].split(".")[0]
        if kind in start and parts[0].isdigit(): start[kind] = max(start[kind], int(parts[0]))
        parts[3] = ""
    print("\\newlabel{%s}{%s}" % (m.group(1), "".join("{%s}" % p for p in parts)))
    found.add(m.group(1))
if main:
    print("\\newcommand{\\SIcounters}{%s}" % "".join("\\setcounter{%s}{%d}" % kv for kv in start.items()))
elif si - found:
    sys.exit(f"labels missing from {sys.argv[2]}: {sorted(si - found)}")
