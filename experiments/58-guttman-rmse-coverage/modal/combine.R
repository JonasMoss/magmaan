#!/usr/bin/env Rscript
# Reduce a Modal cell fan-out into experiment 58's final result CSVs. Each cell
# wrote its own <out-dir>/cells/<slug>/ directory holding the same per-cell
# summary CSVs a single-process run writes. Because the runner seeds each design
# point from its coordinates (cell_code, not a grid-row counter), the cells are
# disjoint slices of one full sweep, so the reduction is a plain row-bind: every
# summary here is already aggregated within its design cell. The bulky per-rep
# diagnostics.csv is left in the cell dirs and not re-bound.
#
# Usage: Rscript combine.R --out-dir PATH

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default) {
  i <- match(flag, args)
  if (is.na(i) || i == length(args)) default else args[[i + 1L]]
}
out_dir <- get_arg("--out-dir", ".")
cells_root <- file.path(out_dir, "cells")
if (!dir.exists(cells_root)) stop("no cells/ under ", out_dir, call. = FALSE)

bind_glob <- function(base) {
  fs <- list.files(cells_root, pattern = paste0("^", base, "\\.csv$"),
                   recursive = TRUE, full.names = TRUE)
  if (!length(fs)) return(NULL)
  parts <- lapply(fs, function(f)
    tryCatch(read.csv(f, stringsAsFactors = FALSE, check.names = FALSE),
             error = function(e) { warning("unreadable: ", f); NULL }))
  parts <- parts[!vapply(parts, is.null, logical(1))]
  if (!length(parts)) return(NULL)
  list(df = do.call(rbind, parts), n = length(parts))
}

# Row-bound across cells. These are the report's inputs plus the other summaries.
row_bound <- c("design", "population", "estimate_summary", "estimate_joint_summary",
               "whole_summary", "whole_joint_summary", "diagnostic_summary",
               "timing_summary", "progress")
for (base in row_bound) {
  s <- bind_glob(base)
  if (is.null(s)) { cat(sprintf("[%-22s] no cells\n", base)); next }
  write.csv(s$df, file.path(out_dir, paste0(base, ".csv")), row.names = FALSE)
  cat(sprintf("[%-22s] %d rows from %d cells\n", base, nrow(s$df), s$n))
}

# metadata is cell-invariant except the fanned design-dimension fields (each cell
# carries singletons); the report only reads `reps` from it, and design.csv above
# is the authoritative design record. Copy one cell's metadata.
meta1 <- list.files(cells_root, pattern = "^metadata\\.csv$",
                    recursive = TRUE, full.names = TRUE)
if (length(meta1)) {
  file.copy(meta1[[1]], file.path(out_dir, "metadata.csv"), overwrite = TRUE)
  cat("[metadata            ] copied from first cell\n")
}
cat(sprintf("\ncombined into %s\n", out_dir))
