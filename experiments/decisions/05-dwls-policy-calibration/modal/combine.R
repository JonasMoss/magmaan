#!/usr/bin/env Rscript
# Combine Modal cell outputs into the study's frozen summaries.
#   Rscript modal/combine.R --run-dir /vol/<run-id> --mode production --git-head <sha>
# Reads <run-dir>/cells/cell_*/raw.rds, recomputes the (cheap, deterministic)
# population targets, and runs the same dwls_summarize() and gates as
# run_experiment.R, writing <run-dir>/final.
args <- commandArgs(TRUE)
opt <- function(key) {
  at <- match(key, args)
  if (is.na(at) || at == length(args)) stop("Missing ", key)
  args[at + 1L]
}
run_dir <- opt("--run-dir"); mode <- opt("--mode"); git_head <- opt("--git-head")
script <- normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
study <- dirname(dirname(script))
source(file.path(study, "..", "..", "_support", "R", "helpers.R"))
set_single_threaded_math()
source(file.path(study, "R", "compute.R"))
source(file.path(study, "R", "summarize.R"))

if(!mode %in% c('smoke','pilot','production','explore','confirm')) stop('Unknown mode')
family <- if('--family' %in% args) opt('--family') else 'all'
lane <- if('--lane' %in% args) opt('--lane') else 'dwls-policy'
if(lane %in% c('exact-first-stage','latent-nonnormal')) {
  source(file.path(study,'modal','exact_combine.R'))
  exact_combine(run_dir,mode,git_head,family,opt,study,lane)
  quit(save='no')
}
if(lane=='mixed') {
  source(file.path(study,'modal','mixed_combine.R'))
  mixed_combine(run_dir,mode,git_head,family,opt,study)
  quit(save='no')
}
if(lane!='dwls-policy') stop('Unknown lane')
cells <- dwls_mode_cells(mode,family)
files <- sort(list.files(file.path(run_dir, "cells"), pattern = "^raw[.]rds$",
                         recursive = TRUE, full.names = TRUE))
if (length(files) != nrow(cells))
  stop("Expected ", nrow(cells), " cell outputs, found ", length(files))
raw <- do.call(rbind, lapply(files, readRDS))
raw <- raw[order(raw$cell_id, raw$replicate), , drop = FALSE]
rownames(raw) <- NULL
if(!setequal(unique(raw$cell_id),cells$cell_id)) stop('Cell IDs do not match mode')

out <- file.path(run_dir, "final")
if (dir.exists(out)) stop("Final directory exists; use a fresh run id")
dir.create(out, recursive = TRUE)
seed_base <- c(smoke = 817130001L, pilot = 817140001L, production = 817150001L, explore = 817150001L, confirm = 817160001L)[[mode]]
if(any(raw$seed != seed_base+10000L*raw$cell_id+raw$replicate)) stop('Seed mismatch')
saveRDS(raw,file.path(out,'raw.rds'))
write_metadata(file.path(out, "metadata.csv"), list(mode = mode, executor = "modal",
  cells = nrow(cells), family = family, seed_base = seed_base, git_head = git_head,
  population_n_per_group = 100000L, population_seed_base = 817120001L, pols_gamma = 4, spectrum_truncate_negative = TRUE),
  packages = c("magmaanlab", "lavaan"))
write_csv(cells, file.path(out, "cells.csv"))
invisible(dwls_population(cells, out))
dwls_summarize(raw, cells, out)
if (any(is.finite(raw$policy_gap) & raw$policy_gap > 1e-7)) stop("Policy equivalence gate failed")
bad <- grepl("Policy unavailable|Policy covariance unavailable", raw$error) &
  !grepl("not_converged|numeric_failure|boundary", raw$error)
if (any(bad)) stop("Required policy component structurally unavailable; see failures.csv")
cat("Combined ", nrow(raw), " rows from ", length(files), " cells into ", out, "\n", sep = "")
