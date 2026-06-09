#!/usr/bin/env Rscript
# FIML measurement-invariance FMG audit, magmaan vs lavaan.
#
# Question: is magmaan's Foldnes-Moss-Gronneberg robust-test family correct and
# feature-complete under FIML (missing data) across the cases that matter for
# applied measurement invariance — multiple groups, cross-group loading/intercept
# equality, and mean structure — for both goodness-of-fit and nested difference
# tests? The grid is deliberately built to surface a gap if one of those
# combinations is half-implemented.
#
# It crosses an invariance ladder (configural -> metric -> scalar) with normal /
# heavy-tailed data and complete / MCAR-missing data, fits FIML, and computes
# FMG GOF + nested tests. The decisive correctness check is the FIML UGamma
# spectrum against lavaan's lavInspect("UGamma") eigenvalues (unstructured H1).
# The complete-data ML path is checked the same way and is expected to deviate
# for cross-group-equality cells (a separate, documented build_u_factor bug).
#
# Usage:
#   Rscript run_experiment.R [--reps N] [--n N] [--seed-base S]
#                            [--conditions CSV] [--dists CSV] [--miss CSV]
#                            [--lavaan-parity] [--smoke]

.support_helpers <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg)) {
    script <- normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE)
  } else {
    ofile <- tryCatch(sys.frames()[[1L]]$ofile, error = function(e) NULL)
    script <- normalizePath(if (is.null(ofile)) "run_experiment.R" else ofile,
                            mustWork = FALSE)
  }
  file.path(dirname(dirname(script)), "_support", "R", "helpers.R")
}
source(.support_helpers())
rm(.support_helpers)
source(experiment_path("R", "invariance_population.R"))
source(experiment_path("R", "fiml_oracle.R"))

parse_args <- function(args) {
  out <- list(reps = 20L, n = 500L, seed_base = 20260609L,
              conditions = c("H0", "H1_metric", "H1_scalar"),
              dists = c("norm", "t5"), miss = c(0.0, 0.15),
              lavaan_parity = FALSE, smoke = FALSE)
  i <- 1L
  take <- function(i) args[[i + 1L]]
  while (i <= length(args)) {
    a <- args[[i]]
    if (a %in% c("-h", "--help")) {
      cat("Usage: Rscript run_experiment.R [--reps N] [--n N] [--seed-base S]\n",
          "       [--conditions H0,H1_metric,H1_scalar] [--dists norm,t5]\n",
          "       [--miss 0,0.15] [--lavaan-parity] [--smoke]\n\n", sep = "")
      cat("  Fits a 1-factor/6-indicator 2-group invariance ladder (configural,\n",
          "  metric, scalar) under FIML and computes FMG GOF + nested tests.\n",
          "  --lavaan-parity adds the lavaan UGamma / chi-square / nested oracle\n",
          "  (rep 1 of each cell). --smoke runs one tiny cell.\n", sep = "")
      quit(save = "no", status = 0L)
    } else if (a == "--reps") { i <- i + 1L; out$reps <- as.integer(take(i - 1L))
    } else if (startsWith(a, "--reps=")) { out$reps <- as.integer(sub("^--reps=", "", a))
    } else if (a == "--n") { i <- i + 1L; out$n <- as.integer(take(i - 1L))
    } else if (startsWith(a, "--n=")) { out$n <- as.integer(sub("^--n=", "", a))
    } else if (a == "--seed-base") { i <- i + 1L; out$seed_base <- as.integer(take(i - 1L))
    } else if (startsWith(a, "--seed-base=")) { out$seed_base <- as.integer(sub("^--seed-base=", "", a))
    } else if (a == "--conditions") { i <- i + 1L; out$conditions <- parse_csv_arg(take(i - 1L))
    } else if (startsWith(a, "--conditions=")) { out$conditions <- parse_csv_arg(sub("^--conditions=", "", a))
    } else if (a == "--dists") { i <- i + 1L; out$dists <- parse_csv_arg(take(i - 1L))
    } else if (startsWith(a, "--dists=")) { out$dists <- parse_csv_arg(sub("^--dists=", "", a))
    } else if (a == "--miss") { i <- i + 1L; out$miss <- parse_csv_numeric(take(i - 1L))
    } else if (startsWith(a, "--miss=")) { out$miss <- parse_csv_numeric(sub("^--miss=", "", a))
    } else if (a == "--lavaan-parity") { out$lavaan_parity <- TRUE
    } else if (a == "--smoke") { out$smoke <- TRUE
    } else { stop("unknown argument: ", a, call. = FALSE) }
    i <- i + 1L
  }
  if (out$smoke) {
    out$reps <- 3L; out$n <- 400L
    out$conditions <- "H0"; out$dists <- "norm"; out$miss <- 0.0
    out$lavaan_parity <- TRUE
  }
  out
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
set_single_threaded_math()
require_pkg("magmaan")
core <- magmaan::magmaan_core
pop  <- build_invariance_population()
levels_all <- invariance_levels()
pairs_all  <- nested_pairs()
have_lav   <- args$lavaan_parity && requireNamespace("lavaan", quietly = TRUE)

CTRL_FIML <- list(max_iter = 16000L, ftol = 1e-13, gtol = 1e-9)
CTRL_ML   <- list(max_iter = 10000L)

# Fit one invariance level under FIML; returns NULL on failure.
fit_fiml_level <- function(level, df) {
  syntax <- invariance_syntax(level)
  spec <- magmaan::model_spec(syntax, group = "school",
                              group_labels = c("A", "B"), meanstructure = TRUE)
  tryCatch({
    fit <- core$fit_fiml(spec, magmaan::df_to_fiml_data(df, spec, group = "school"),
                         control = CTRL_FIML)
    if (!isTRUE(fit$converged)) return(NULL)
    fit
  }, error = function(e) NULL)
}
fit_ml_level <- function(level, df) {
  syntax <- invariance_syntax(level)
  spec <- magmaan::model_spec(syntax, group = "school",
                              group_labels = c("A", "B"), meanstructure = TRUE)
  tryCatch({
    fit <- core$fit_ml(spec, magmaan::df_to_data(df, spec, group = "school"),
                       control = CTRL_ML)
    if (!isTRUE(fit$converged)) return(NULL)
    fit
  }, error = function(e) NULL)
}

alpha <- 0.05
gof_rows <- list(); nested_rows <- list(); parity_rows <- list(); invariant_rows <- list()
gk <- nk <- pk <- ik <- 0L

cells <- expand.grid(condition = args$conditions, dist = args$dists,
                     miss = args$miss, stringsAsFactors = FALSE)
violate_for <- function(condition) switch(condition,
  H0 = NULL, H1_metric = "metric", H1_scalar = "scalar", NULL)

t_start <- proc.time()[["elapsed"]]
for (ci in seq_len(nrow(cells))) {
  cell <- cells[ci, ]
  violate <- violate_for(cell$condition)
  cell_tag <- sprintf("%s/%s/miss=%.2f", cell$condition, cell$dist, cell$miss)
  message(sprintf("[cell %d/%d] %s", ci, nrow(cells), cell_tag))

  for (rep in seq_len(args$reps)) {
    set.seed(args$seed_base + 1000L * ci + rep)
    df_complete <- draw_complete(pop, args$n, dist = cell$dist, violate = violate)
    df <- inject_mcar(df_complete, pop$ov, cell$miss)

    fits <- lapply(levels_all, fit_fiml_level, df = df)
    names(fits) <- levels_all
    if (any(vapply(fits, is.null, logical(1)))) next   # skip rep on any failure

    # ---- GOF FMG at each level -------------------------------------------
    for (lev in levels_all) {
      tab <- tryCatch(magmaan::fmg_tests(fits[[lev]]), error = function(e) NULL)
      sp  <- tryCatch(magmaan:::infer_fiml_fmg_spectrum(fits[[lev]]),
                      error = function(e) NULL)
      if (is.null(tab) || is.null(sp)) next
      for (r in seq_len(nrow(tab))) {
        gk <- gk + 1L
        gof_rows[[gk]] <- data.frame(
          condition = cell$condition, dist = cell$dist, miss = cell$miss,
          rep = rep, level = lev, test = tab$label[r],
          p_value = tab$p_value[r], base_stat = tab$base_statistic[r],
          df = sp$df, reject = as.integer(tab$p_value[r] < alpha),
          eig_min = min(sp$biased), eig_max = max(sp$biased),
          trace_xcheck = sp$trace_xcheck, eig_sum = sum(sp$biased),
          stringsAsFactors = FALSE)
      }
    }

    # ---- Nested FMG for each adjacent ladder pair -------------------------
    for (pr in pairs_all) {
      h1 <- fits[[pr[["h1"]]]]; h0 <- fits[[pr[["h0"]]]]
      nt <- tryCatch(magmaan::nestedTest(h1, h0, method = "satorra.2000",
                                         A.method = "exact"),
                     error = function(e) NULL)
      if (is.null(nt)) next
      nk <- nk + 1L
      nested_rows[[nk]] <- data.frame(
        condition = cell$condition, dist = cell$dist, miss = cell$miss,
        rep = rep, pair = paste(pr[["h1"]], pr[["h0"]], sep = "_vs_"),
        T_diff = nt$T_diff, df_diff = nt$df_diff, scale_c = nt$scale_c,
        p_unscaled = nt$p_unscaled, p_scaled = nt$p_scaled,
        p_adjusted = nt$p_adjusted, p_mixture = nt$p_mixture,
        reject_unscaled = as.integer(nt$p_unscaled < alpha),
        reject_scaled   = as.integer(nt$p_scaled < alpha),
        reject_mixture  = as.integer(nt$p_mixture < alpha),
        stringsAsFactors = FALSE)
    }

    # ---- Parity + invariants on rep 1 only --------------------------------
    if (rep == 1L && have_lav) {
      for (lev in levels_all) {
        sp <- tryCatch(magmaan:::infer_fiml_fmg_spectrum(fits[[lev]]),
                       error = function(e) NULL)
        if (is.null(sp)) next
        miss_lav <- if (cell$miss > 0) "fiml" else "listwise"
        lav <- tryCatch(lavaan_fit_level(invariance_syntax(lev), df, lev, miss_lav),
                        error = function(e) NULL)
        if (is.null(lav)) next
        lu <- lavaan_ugamma(lav)
        # FIML LRT chi-square vs lavaan chisq.
        pk <- pk + 1L
        parity_rows[[pk]] <- data.frame(
          condition = cell$condition, dist = cell$dist, miss = cell$miss,
          level = lev, metric = "fiml_chisq",
          magmaan = sp$chi2_lrt, lavaan = lu$chisq,
          abs_diff = abs(sp$chi2_lrt - lu$chisq), note = "", stringsAsFactors = FALSE)
        # FIML UGamma trace vs lavaan trace.
        pk <- pk + 1L
        parity_rows[[pk]] <- data.frame(
          condition = cell$condition, dist = cell$dist, miss = cell$miss,
          level = lev, metric = "fiml_ugamma_trace",
          magmaan = sp$trace_xcheck, lavaan = lu$trace,
          abs_diff = abs(sp$trace_xcheck - lu$trace), note = "", stringsAsFactors = FALSE)
        # FIML UGamma full spectrum vs lavaan eigenvalues (complete data only).
        if (cell$miss == 0 && !is.null(lu$eigenvalues)) {
          pk <- pk + 1L
          parity_rows[[pk]] <- data.frame(
            condition = cell$condition, dist = cell$dist, miss = cell$miss,
            level = lev, metric = "fiml_ugamma_spectrum_maxabs",
            magmaan = NA_real_, lavaan = NA_real_,
            abs_diff = spectrum_maxabs(sp$biased, lu$eigenvalues),
            note = "FIML vs lavaan unstructured UGamma eigenvalues",
            stringsAsFactors = FALSE)
          # Complete-data ML (structured) spectrum vs lavaan structured UGamma:
          # demonstrates the build_u_factor cross-group-equality deviation.
          fml <- fit_ml_level(lev, df)
          lav_s <- tryCatch(lavaan::cfa(invariance_syntax(lev), df, group = "school",
                              group.equal = lavaan_group_equal(lev),
                              estimator = "ML", test = "satorra.bentler",
                              meanstructure = TRUE, h1.information = "structured"),
                            error = function(e) NULL)
          if (!is.null(fml) && !is.null(lav_s)) {
            ml_tab <- tryCatch(magmaan::fmg_tests(fml, tests = "pEBA4_RLS"),
                               error = function(e) NULL)
            lus <- lavaan_ugamma(lav_s)
            if (!is.null(ml_tab) && !is.null(lus$eigenvalues)) {
              pk <- pk + 1L
              parity_rows[[pk]] <- data.frame(
                condition = cell$condition, dist = cell$dist, miss = cell$miss,
                level = lev, metric = "ml_ugamma_spectrum_maxabs",
                magmaan = NA_real_, lavaan = NA_real_,
                abs_diff = spectrum_maxabs(as.numeric(ml_tab$eigenvalues[[1]]),
                                           lus$eigenvalues),
                note = "complete-data ML (structured) vs lavaan structured UGamma",
                stringsAsFactors = FALSE)
            }
          }
        }
      }

      # Internal invariants (complete H0 cells): trace identity + group-permute.
      if (cell$miss == 0 && cell$condition == "H0") {
        for (lev in levels_all) {
          sp <- tryCatch(magmaan:::infer_fiml_fmg_spectrum(fits[[lev]]),
                         error = function(e) NULL)
          if (is.null(sp)) next
          ik <- ik + 1L
          invariant_rows[[ik]] <- data.frame(
            condition = cell$condition, dist = cell$dist, level = lev,
            invariant = "trace_identity",
            value = abs(sp$trace_xcheck - sum(sp$biased)), tol = 1e-6,
            stringsAsFactors = FALSE)
        }
        # Group-permutation invariance of the metric GOF spectrum.
        df_swap <- df
        df_swap$school <- ifelse(df$school == "A", "B", "A")
        f_swap <- fit_fiml_level("metric", df_swap)
        sp0 <- tryCatch(magmaan:::infer_fiml_fmg_spectrum(fits[["metric"]]),
                        error = function(e) NULL)
        sp1 <- if (!is.null(f_swap))
          tryCatch(magmaan:::infer_fiml_fmg_spectrum(f_swap), error = function(e) NULL)
        else NULL
        if (!is.null(sp0) && !is.null(sp1)) {
          ik <- ik + 1L
          invariant_rows[[ik]] <- data.frame(
            condition = cell$condition, dist = cell$dist, level = "metric",
            invariant = "group_permutation",
            value = spectrum_maxabs(sp0$biased, sp1$biased), tol = 1e-6,
            stringsAsFactors = FALSE)
        }
      }
    }
  }
}
elapsed <- proc.time()[["elapsed"]] - t_start

# ---- Aggregate rejection rates ---------------------------------------------
gof_df    <- if (length(gof_rows))    do.call(rbind, gof_rows)    else data.frame()
nested_df <- if (length(nested_rows)) do.call(rbind, nested_rows) else data.frame()

summary_rows <- list(); sk <- 0L
if (nrow(nested_df)) {
  grp <- split(nested_df, list(nested_df$condition, nested_df$dist,
                               nested_df$miss, nested_df$pair), drop = TRUE)
  for (g in grp) {
    sk <- sk + 1L
    summary_rows[[sk]] <- data.frame(
      kind = "nested", condition = g$condition[1], dist = g$dist[1],
      miss = g$miss[1], key = g$pair[1], n_rep = nrow(g),
      reject_unscaled = mean(g$reject_unscaled),
      reject_scaled   = mean(g$reject_scaled),
      reject_mixture  = mean(g$reject_mixture),
      stringsAsFactors = FALSE)
  }
}
if (nrow(gof_df)) {
  gof_sb <- gof_df[gof_df$test %in% c("sb_ml", "pall_ml", "peba4_ml"), , drop = FALSE]
  grp <- split(gof_sb, list(gof_sb$condition, gof_sb$dist, gof_sb$miss,
                            gof_sb$level, gof_sb$test), drop = TRUE)
  for (g in grp) {
    sk <- sk + 1L
    summary_rows[[sk]] <- data.frame(
      kind = "gof", condition = g$condition[1], dist = g$dist[1],
      miss = g$miss[1], key = paste(g$level[1], g$test[1], sep = ":"),
      n_rep = nrow(g), reject_unscaled = NA_real_,
      reject_scaled = mean(g$reject), reject_mixture = NA_real_,
      stringsAsFactors = FALSE)
  }
}
summary_df <- if (length(summary_rows)) do.call(rbind, summary_rows) else data.frame()

# ---- Write outputs ----------------------------------------------------------
out_dir <- ensure_results_dir()
write_csv(cells, file.path(out_dir, "cells.csv"))
write_csv(gof_df, file.path(out_dir, "gof_fits.csv"))
write_csv(nested_df, file.path(out_dir, "nested_fits.csv"))
write_rows(parity_rows, file.path(out_dir, "lavaan_parity.csv"))
write_rows(invariant_rows, file.path(out_dir, "invariants.csv"))
write_csv(summary_df, file.path(out_dir, "summary_rejection.csv"))
write_metadata(
  file.path(out_dir, "metadata.csv"),
  values = list(reps = args$reps, n_per_group = args$n, seed_base = args$seed_base,
                conditions = args$conditions, dists = args$dists, miss = args$miss,
                lavaan_parity = have_lav, elapsed_sec = round(elapsed, 1)),
  packages = c("magmaan", "lavaan"))

cat(sprintf("\nDone in %.1fs. Wrote:\n", elapsed))
for (f in c("cells.csv", "gof_fits.csv", "nested_fits.csv", "lavaan_parity.csv",
            "invariants.csv", "summary_rejection.csv", "metadata.csv")) {
  cat("  ", file.path(out_dir, f), "\n", sep = "")
}
if (length(parity_rows)) {
  pj <- do.call(rbind, parity_rows)
  cat("\nParity (max |abs_diff| by metric):\n")
  agg <- aggregate(abs_diff ~ metric, pj, function(x) max(x, na.rm = TRUE))
  print(agg, row.names = FALSE)
}
