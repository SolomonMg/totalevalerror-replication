TEXBIN = /Library/TeX/texbin
TEX = $(TEXBIN)/pdflatex
BIB = $(TEXBIN)/bibtex
LATEXMK = /usr/bin/perl $(TEXBIN)/latexmk
MAIN = main

export PATH := $(TEXBIN):$(PATH)

PNAS = pnas_main
NEURIPS = neurips_main
# Figures the paper includes (via \graphicspath{{../figures/}}); a changed figure must trigger a rebuild.
FIGS = $(wildcard ../figures/*.pdf)
ARXIV = neurips_arxiv
DROPBOX = /Users/ns/Library/CloudStorage/Dropbox/drafts_to_share/Total_Eval_Error.pdf

.PHONY: all latexmk clean share neurips arxiv sync-main

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib
	-$(TEX) -synctex=1 -interaction=nonstopmode $(MAIN)
	-$(BIB) $(MAIN)
	$(TEX) -synctex=1 -interaction=nonstopmode $(MAIN)
	$(TEX) -synctex=1 -interaction=nonstopmode $(MAIN)

$(PNAS).pdf: $(PNAS).tex references.bib
	-$(TEX) -synctex=1 -interaction=nonstopmode $(PNAS)
	-$(BIB) $(PNAS)
	$(TEX) -synctex=1 -interaction=nonstopmode $(PNAS)
	$(TEX) -synctex=1 -interaction=nonstopmode $(PNAS)

neurips: $(NEURIPS).pdf

# neurips_main.tex is generated: camera-ready options, checklist after the appendix.
# Edit neurips_arxiv.tex, never neurips_main.tex.
$(NEURIPS).tex: $(ARXIV).tex neurips_checklist.tex
	{ echo '% GENERATED from neurips_arxiv.tex by make. Edit neurips_arxiv.tex, not this file.'; \
	  sed -e 's/\\usepackage\[preprint\]{neurips\/neurips_2026}/\\usepackage[main, final]{neurips\/neurips_2026}/' \
	      -e 's/^\\end{document}$$/\\newpage\n\\input{neurips_checklist}\n\n\\end{document}/' $(ARXIV).tex; } > $@

sync-main: $(NEURIPS).tex

$(NEURIPS).pdf: $(NEURIPS).tex references.bib neurips/neurips_2026.sty neurips_appendix_body.tex neurips_checklist.tex $(FIGS)
	-$(TEX) -synctex=1 -interaction=nonstopmode $(NEURIPS)
	-$(BIB) $(NEURIPS)
	$(TEX) -synctex=1 -interaction=nonstopmode $(NEURIPS)
	$(TEX) -synctex=1 -interaction=nonstopmode $(NEURIPS)

arxiv: $(ARXIV).pdf

$(ARXIV).pdf: $(ARXIV).tex references.bib neurips/neurips_2026.sty neurips_appendix_body.tex $(FIGS)
	-$(TEX) -synctex=1 -interaction=nonstopmode $(ARXIV)
	-$(BIB) $(ARXIV)
	$(TEX) -synctex=1 -interaction=nonstopmode $(ARXIV)
	$(TEX) -synctex=1 -interaction=nonstopmode $(ARXIV)

latexmk: $(MAIN).tex references.bib
	$(LATEXMK) -pdf $(MAIN).tex

share: $(PNAS).pdf
	cp $(PNAS).pdf $(DROPBOX)
	@echo "Copied to Dropbox: $(DROPBOX)"

clean:
	rm -f $(MAIN).{aux,bbl,blg,log,out,pdf,toc,lof,lot,fls,fdb_latexmk,synctex.gz}
