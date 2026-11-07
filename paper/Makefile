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
NOAPP = neurips_arxiv_noapp
SI = neurips_si
SIURL = https://solomonmg.github.io/pdf/totalevalerror_si.pdf
DROPBOX = /Users/ns/Library/CloudStorage/Dropbox/drafts_to_share/Total_Eval_Error.pdf

.PHONY: all latexmk clean share neurips arxiv noapp si sync-main

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

# arXiv version without the appendix (generated; edit neurips_arxiv.tex), with the NeurIPS footer. References into the SI keep
# the full version's numbers: si_labels.py copies the SI labels from the full build's .aux.
noapp: $(NOAPP).pdf

$(NOAPP).tex: $(ARXIV).tex
	{ echo '% GENERATED from neurips_arxiv.tex by make. Edit neurips_arxiv.tex, not this file.'; \
	  sed -e '/^\\appendix$$/d' -e '/^\\input{neurips_appendix_body}$$/d' \
	      -e 's/\\usepackage\[preprint\]{neurips\/neurips_2026}/\\usepackage[main, final]{neurips\/neurips_2026}/' \
	      -e 's#^\\title{\(.*\)}$$#\\title{\1\\thanks{The SI Appendix is available at \\url{$(SIURL)}.}}#' \
	      -e 's/^\\begin{document}$$/\\makeatletter\\input{neurips_si_labels}\\makeatother\n\\begin{document}/' $(ARXIV).tex; } > $@

neurips_si_labels.tex: $(ARXIV).pdf neurips_appendix_body.tex si_labels.py
	python3 si_labels.py neurips_appendix_body.tex $(ARXIV).aux > $@

$(NOAPP).pdf: $(NOAPP).tex neurips_si_labels.tex references.bib neurips/neurips_2026.sty $(FIGS)
	-$(TEX) -synctex=1 -interaction=nonstopmode $(NOAPP)
	-$(BIB) $(NOAPP)
	$(TEX) -synctex=1 -interaction=nonstopmode $(NOAPP)
	$(TEX) -synctex=1 -interaction=nonstopmode $(NOAPP)

# Standalone SI Appendix for the website (SIURL), generated, with the NeurIPS footer. Same section, figure, table, and equation
# numbers as the full version; references into the main text print its numbers (si_labels.py --main).
si: $(SI).pdf

$(SI).tex: $(ARXIV).tex
	{ echo '% GENERATED from neurips_arxiv.tex by make. Edit neurips_arxiv.tex, not this file.'; \
	  awk '/^\\begin{document}$$/{print "\\makeatletter\\input{neurips_main_labels}\\makeatother"} {print} /^\\maketitle/{exit}' $(ARXIV).tex \
	    | sed -e 's/^\\title{/\\title{Supplementary Information for: /' \
	          -e 's/\\usepackage\[preprint\]{neurips\/neurips_2026}/\\usepackage[main, final]{neurips\/neurips_2026}/'; \
	  printf '%s\n' '\appendix' '\SIcounters' '\input{neurips_appendix_body}' '\clearpage' '\bibliographystyle{plainnat}' '\bibliography{references}' '' '\end{document}'; } > $@

neurips_main_labels.tex: $(ARXIV).pdf neurips_appendix_body.tex si_labels.py
	python3 si_labels.py --main neurips_appendix_body.tex $(ARXIV).aux > $@

$(SI).pdf: $(SI).tex neurips_main_labels.tex neurips_appendix_body.tex references.bib neurips/neurips_2026.sty $(FIGS)
	-$(TEX) -synctex=1 -interaction=nonstopmode $(SI)
	-$(BIB) $(SI)
	$(TEX) -synctex=1 -interaction=nonstopmode $(SI)
	$(TEX) -synctex=1 -interaction=nonstopmode $(SI)

latexmk: $(MAIN).tex references.bib
	$(LATEXMK) -pdf $(MAIN).tex

share: $(PNAS).pdf
	cp $(PNAS).pdf $(DROPBOX)
	@echo "Copied to Dropbox: $(DROPBOX)"

clean:
	rm -f $(MAIN).{aux,bbl,blg,log,out,pdf,toc,lof,lot,fls,fdb_latexmk,synctex.gz}
