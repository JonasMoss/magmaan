#!/usr/bin/env Rscript
# Results-independent grid from criteria/fit_indices.md; run from any directory.
args <- commandArgs(trailingOnly = TRUE)
if (length(args)) {
  if (identical(args, "--help")) {
    cat("Rscript scripts/production_cells.R\nWrites production/cells_{all,ml,fiml}.csv from the registered design, without fitting.\n")
    quit(status = 0)
  }
  stop("Unknown option")
}
file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
leaf <- dirname(dirname(normalizePath(sub("^--file=", "", file_arg[1]))))
root <- normalizePath(file.path(leaf, "../../.."))
source(file.path(root, "experiments/_support/R/helpers.R"))
source(support_path("R", "basis.R"))
source(file.path(leaf, "R", "config.R"))
blocks <- read.csv(support_path("basis", "sem_score", "misspecification.csv"))
signs <- unique(blocks[, c("id", "sign")])
models <- basis_ids()
stopifnot(length(models) == 12L, !anyDuplicated(signs$id),
          setequal(models, signs$id), all(signs$sign %in% c(-1, 1)))
cells <- expand.grid(model = models, delta = c(0, .1, .2, .4),
  N = c(200L, 500L, 1000L, 4000L),
  generator = c("normal", "vm", "independent", "discretized"),
  missing_rate = c(0, .15, .30), stringsAsFactors = FALSE)
cells$delta <- cells$delta * signs$sign[match(cells$model, signs$id)]
cells$estimator <- ifelse(cells$missing_rate == 0, "ML", "FIML")
cells$target_rmsea <- NA_real_
cells$skewness <- 1; cells$kurtosis <- 2; cells$categories <- 5L
# Empty fields select the registered first-variable cause and last-half targets.
cells$cause <- ""; cells$missing_targets <- ""
# Semantic IDs stay stable when subsets or execution order change.
cells$id <- paste(cells$model, sprintf("d%02d", round(abs(cells$delta)*100)),
  paste0("n", cells$N), cells$generator, tolower(cells$estimator),
  sprintf("mar%02d", round(cells$missing_rate*100)), sep = "_")
cells <- validate_config(cells)
stopifnot(nrow(cells) == 2304L, sum(cells$estimator == "ML") == 768L)
out <- file.path(leaf, "production")
dir.create(out, showWarnings = FALSE)
for (setting in c("all", "ml", "fiml")) {
  selected <- if (setting == "all") cells else cells[tolower(cells$estimator) == setting, ]
  write_csv(selected, file.path(out, paste0("cells_", setting, ".csv")))
}
cat("Wrote 2304 cells (768 ML, 1536 FIML) to", out, "\n")
