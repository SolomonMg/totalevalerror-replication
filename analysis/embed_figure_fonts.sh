#!/usr/bin/env bash
# embed_figure_fonts.sh
# Embed all fonts in every figure PDF the paper includes. R's pdf() device writes
# Helvetica as an unembedded base-14 font; the NeurIPS template asks for Type 1 or
# embedded TrueType fonts only, and embedding removes any doubt. Ghostscript substitutes
# the metric-compatible Nimbus Sans, so layout does not change.
#
# Re-run after any figure script rewrites a PDF (run_all.sh does this at the end of Stage 5).
# Usage: bash analysis/embed_figure_fonts.sh   (from project root; needs `gs`)
set -euo pipefail
command -v gs >/dev/null || { echo "Ghostscript (gs) not found: brew install ghostscript"; exit 1; }

figs=$(grep -ho '\\includegraphics\(\[[^]]*\]\)\?{[^}]*}' paper/neurips_arxiv.tex paper/neurips_appendix_body.tex \
       | sed -E 's/.*\{([^}]*)\}/\1/' | sort -u)
n=0
for f in $figs; do
  path="figures/${f}"; [[ "$path" == *.pdf ]] || path="${path}.pdf"
  [ -f "$path" ] || { echo "missing: $path"; continue; }
  if pdffonts "$path" 2>/dev/null | awk 'NR > 2 && $(NF-4) == "no" {bad = 1} END {exit !bad}'; then
    tmp="$(mktemp -t embed).pdf"
    # /prepress plus an empty NeverEmbed list: by default gs never embeds the 14 standard
    # PDF fonts (Helvetica, Symbol, ...) even with -dEmbedAllFonts=true.
    gs -q -dNOPAUSE -dBATCH -dSAFER -sDEVICE=pdfwrite -dCompatibilityLevel=1.5 -dPDFSETTINGS=/prepress \
       -dEmbedAllFonts=true -dSubsetFonts=true -sOutputFile="$tmp" \
       -c "<</NeverEmbed [ ]>> setdistillerparams" -f "$path"
    mv "$tmp" "$path"; n=$((n + 1)); echo "embedded: $path"
  fi
  if pdffonts "$path" 2>/dev/null | awk 'NR > 2 && $(NF-4) == "no" {bad = 1} END {exit !bad}'; then
    echo "STILL UNEMBEDDED: $path"; exit 1
  fi
done
echo "done: $n figure(s) rewritten; all included figures have embedded fonts"
