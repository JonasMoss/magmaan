.PHONY: all sims textbook tables clean

all: sims textbook tables

sims:
	Rscript scripts/run_convergence_sims.R

textbook:
	Rscript scripts/run_textbook_subset.R

tables:
	Rscript scripts/build_tables.R

clean:
	rm -f results/*.csv tables/*.csv tables/*.tex report/pilot-report.tex
