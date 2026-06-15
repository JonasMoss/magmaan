#!/usr/bin/env Rscript
# MLR validation: is the MLR (Yuan-Bentler) over-rejection real, or a magmaan
# artifact? The main run only checks magmaan vs lavaan MLR on rep 1 of each cell.
# This fits magmaan FIML+MLR AND lavaan MLR (estimator="MLR", missing="ml",
# defaults) on the SAME data every replicate, on the hardest cells, and confirms:
#   (1) per-replicate agreement across ALL reps (not just rep 1),
#   (2) lavaan's own MLR rejection rate equals magmaan's, and
#   (3) the variance-inflation factor df*Sum(l^2)/Sum(l)^2 predicts that a
#       mean-scaled statistic must over-reject (it is >> 1 under non-normality).
# Writes results/mlr_validation.csv. Run after the main grid; needs lavaan.
#
# Usage: Rscript scripts/validate_mlr.R [--reps N] [--dists CSV] [--rate R]

.support <- function() {
  fa <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  s <- normalizePath(sub("^--file=", "", fa[[1L]]), mustWork = TRUE)
  file.path(dirname(dirname(dirname(s))), "_support", "R", "helpers.R")
}
source(.support())
source(file.path(dirname(.support()), "missingness.R"))
exp_root <- dirname(dirname(normalizePath(
  sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1L]]))))
for (f in c("population", "generators", "tests", "oracle")) {
  source(file.path(exp_root, "R", paste0(f, ".R")))
}
suppressMessages(library(magmaan)); set_single_threaded_math()
core <- magmaan::magmaan_core

a <- commandArgs(trailingOnly = TRUE)
getarg <- function(flag, default) {
  i <- which(a == flag); if (length(i)) a[[i + 1L]] else default
}
reps  <- as.integer(getarg("--reps", "150"))
dists <- parse_csv_arg(getarg("--dists", "vm2,pl2,ig2"))
rate  <- as.numeric(getarg("--rate", "0.30"))
mech  <- "MCAR"
require_pkg("lavaan")

pop <- build_invariance_population(); n_groups <- c(500L, 350L)
rows <- list()
for (dist in dists) {
  sampler <- build_cell_sampler(pop, dist, NULL, n_groups, reps, 99000L, core = core)
  pm <- pl <- vif <- rep(NA_real_, reps)
  for (i in seq_len(reps)) {
    mm <- apply_missingness(sampler$draw(i), pop$ov, mech, rate, seed = 99000L + i)
    fit <- fit_fiml_level("metric", mm$df)
    if (is.null(fit)) next
    m <- mlr_test(fit); g <- fmg_gof(fit)
    lav <- tryCatch(lavaan_mlr_fit(invariance_syntax("metric"), mm$df, "metric", "ml"),
                    error = function(e) NULL)
    if (is.null(m) || is.null(lav)) next
    pm[i] <- m$p; pl[i] <- lavaan_mlr_measures(lav)$p_scaled
    if (!is.null(g$spectrum)) vif[i] <- length(g$spectrum) *
        sum(g$spectrum^2) / sum(g$spectrum)^2
  }
  ok <- is.finite(pm) & is.finite(pl)
  rows[[dist]] <- data.frame(
    dist = dist, mech = mech, rate = rate, reps_ok = sum(ok),
    mlr_reject_magmaan = mean(pm[ok] < .05),
    mlr_reject_lavaan  = mean(pl[ok] < .05),
    max_abs_p_diff = max(abs(pm[ok] - pl[ok])),
    mean_abs_p_diff = mean(abs(pm[ok] - pl[ok])),
    variance_inflation = mean(vif[ok]), stringsAsFactors = FALSE)
  cat(sprintf("%-4s: magmaan=%.3f lavaan=%.3f  max|dp|=%.1e  vif=%.2f\n",
              dist, rows[[dist]]$mlr_reject_magmaan, rows[[dist]]$mlr_reject_lavaan,
              rows[[dist]]$max_abs_p_diff, rows[[dist]]$variance_inflation))
}
out <- do.call(rbind, rows)
out_path <- file.path(exp_root, "results", "mlr_validation.csv")
write_csv(out, out_path)
cat(sprintf("\nWrote %s\n", out_path))
