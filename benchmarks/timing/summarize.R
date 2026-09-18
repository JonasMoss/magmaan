#!/usr/bin/env Rscript
# Summarize magmaan_timing_bench CSV output. Base R only, no dependencies.
#
#   Rscript benchmarks/timing/summarize.R results/timing_pladder_*.csv
#
# Prints, per input file and whichever axes actually vary in it:
#   * stage medians in microseconds, by p (the scaling table)
#   * a log-log scaling exponent per stage, d log(time) / d log(p)
#   * n-sensitivity, as the ratio of the largest n to the smallest
#   * parameterization arms side by side, with optimizer evaluation counts
#   * the staged-vs-composite reconstruction gap

args <- commandArgs(trailingOnly = TRUE)
if (!length(args)) stop("usage: summarize.R <timing csv> [...]")

us <- function(ns) round(ns / 1000, 2)

fmt_table <- function(m, digits = 2) {
  m <- round(m, digits)
  print(m, quote = FALSE, na.print = ".")
}

for (path in args) {
  d <- utils::read.csv(path, stringsAsFactors = FALSE)
  cat("\n=====", basename(path), "=====\n")
  cat(sprintf("%d rows | models: %s\n", nrow(d),
              paste(unique(d$model), collapse = ", ")))

  # ---- scaling in p -------------------------------------------------------
  if (length(unique(d$p)) > 1L) {
    for (mod in unique(d$model)) {
      s <- d[d$model == mod & d$parameterization == "marker", ]
      if (length(unique(s$p)) < 2L) next
      # One n only, so stages are comparable across p.
      s <- s[s$n == min(s$n), ]
      tab <- tapply(s$median_ns, list(s$stage, s$p), function(x) us(x[1]))
      tab <- matrix(unlist(tab), nrow = nrow(tab), dimnames = dimnames(tab))
      slope <- apply(tab, 1, function(y) {
        pv <- as.numeric(colnames(tab))
        ok <- is.finite(y) & y > 0
        if (sum(ok) < 2L) return(NA_real_)
        unname(stats::coef(stats::lm(log(y[ok]) ~ log(pv[ok])))[2])
      })
      cat(sprintf("\n-- %s : stage medians (us) by p, with log-log slope --\n", mod))
      out <- cbind(tab, slope = round(slope, 2))
      fmt_table(out[order(-out[, ncol(out) - 1L]), , drop = FALSE])
    }
  }

  # ---- sensitivity to n ---------------------------------------------------
  if (length(unique(d$n)) > 1L) {
    cat("\n-- n-sensitivity: median at max n / median at min n --\n")
    cat("   (complete data: only sample_stats and info_cross_products should move)\n")
    for (pp in sort(unique(d$p))) {
      s <- d[d$p == pp, ]
      lo <- s[s$n == min(s$n), ]
      hi <- s[s$n == max(s$n), ]
      m <- merge(lo[, c("stage", "median_ns")], hi[, c("stage", "median_ns")],
                 by = "stage", suffixes = c(".lo", ".hi"))
      m$ratio <- round(m$median_ns.hi / m$median_ns.lo, 2)
      m <- m[order(-m$ratio), c("stage", "ratio")]
      cat(sprintf("\n   p = %d, n %d -> %d\n", pp, min(s$n), max(s$n)))
      print(utils::head(m, 6), row.names = FALSE)
      flat <- m[m$ratio > 0.8 & m$ratio < 1.25, "stage"]
      cat(sprintf("   flat in n (%d stages): %s\n", length(flat),
                  paste(utils::head(flat, 12), collapse = " ")))
    }
  }

  # ---- parameterization arms ---------------------------------------------
  if (length(unique(d$parameterization)) > 1L) {
    cat("\n-- parameterization arms --\n")
    key <- c("optimize_lbfgs", "fit_ml_end_to_end", "objective_call",
             "constraints_eq", "staged_fit_ml", "pipeline_total")
    s <- d[d$stage %in% key, ]
    for (mod in unique(s$model)) for (pp in sort(unique(s$p))) {
      t <- s[s$model == mod & s$p == pp, ]
      if (!nrow(t)) next
      tab <- tapply(t$median_ns, list(t$stage, t$parameterization),
                    function(x) us(x[1]))
      tab <- matrix(unlist(tab), nrow = nrow(tab), dimnames = dimnames(tab))
      ev <- tapply(d$fit_g_evals[d$model == mod & d$p == pp],
                   d$parameterization[d$model == mod & d$p == pp],
                   function(x) x[1])
      cat(sprintf("\n   %s, p = %d  (us; g_evals: %s)\n", mod, pp,
                  paste(names(ev), unlist(ev), sep = "=", collapse = " ")))
      fmt_table(tab)
    }
  }

  # ---- staged vs composite ------------------------------------------------
  a <- d[d$stage == "staged_fit_ml", ]
  b <- d[d$stage == "fit_ml_end_to_end", ]
  if (nrow(a) && nrow(b)) {
    k <- c("model", "parameterization", "p", "n")
    m <- merge(a[, c(k, "median_ns")], b[, c(k, "median_ns")], by = k,
               suffixes = c(".staged", ".composite"))
    m$gap_pct <- round(100 * (m$median_ns.staged / m$median_ns.composite - 1), 1)
    cat("\n-- staged reconstruction vs composite fit_ml (gap %) --\n")
    cat(sprintf("   median |gap| = %.1f%%, max |gap| = %.1f%%\n",
                stats::median(abs(m$gap_pct)), max(abs(m$gap_pct))))
    worst <- m[order(-abs(m$gap_pct)), c(k, "gap_pct")]
    print(utils::head(worst, 5), row.names = FALSE)
  }
}
