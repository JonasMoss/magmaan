TEX = convergence-note.tex
PDF = convergence-note.pdf
JOB = convergence-note

.PHONY: pdf clean distclean

pdf: $(PDF)

$(PDF): $(TEX) convergence-note.bib
	@if command -v latexmk >/dev/null 2>&1; then \
		latexmk -pdf -interaction=nonstopmode -halt-on-error $(TEX); \
	else \
		pdflatex -interaction=nonstopmode -halt-on-error $(TEX); \
		bibtex $(JOB); \
		pdflatex -interaction=nonstopmode -halt-on-error $(TEX); \
		pdflatex -interaction=nonstopmode -halt-on-error $(TEX); \
	fi

clean:
	@if command -v latexmk >/dev/null 2>&1; then \
		latexmk -c $(TEX); \
	else \
		rm -f $(JOB).aux $(JOB).bbl $(JOB).blg $(JOB).fdb_latexmk \
			$(JOB).fls $(JOB).log $(JOB).out $(JOB).toc; \
	fi

distclean: clean
	rm -f $(PDF)
