## magmaan R bindings -- nested ordinal measurement-invariance test via the
## `group.equal` keyword: Satorra-2000 scaled LRTs across the ordinal theta
## measurement-invariance ladder and the FMG eigenvalue-tail diagnostics on the
## difference spectrum, cross-checked against lavaan.
##
## Run from the repo root after installing the package:
##     Rscript r-package/examples/group_equal_nested_ordinal.R
##
## The Wu-Estabrook release can make these pairs non-nested in raw parameter
## space, so -- like lavaan -- the nested test uses A.method = "delta".
## nestedTest() returns the difference spectrum, and fmg_nested_ordinal() applies
## the FMG transforms to it the same way fmg_tests_ordinal() does for a
## single-model spectrum. The oracle is lavaan's
## `lavTestLRT(method = "satorra.2000")` on robust (`se = "robust.sem"`)
## DWLS-theta fits.
##
## magmaan's LS chi-square is on the 2N*fmin scale; lavaan rescales by (N-G)/N,
## so the scaled difference is compared after that rescaling.

suppressMessages({ library(magmaan); library(lavaan) })
core <- magmaan::magmaan_core
ctrl <- list(max_iter = 4000, ftol = 1e-13, gtol = 1e-8)

make_ord_df_scaled <- function(n, cuts_by_var, seed = 1L,
                               sd_scale = 1, mean_shift = 0) {
  set.seed(seed)
  p <- length(cuts_by_var)
  z <- matrix(rnorm(n * p), n, p)
  z[, 2] <- 0.55 * z[, 1] + sqrt(1 - 0.55^2) * z[, 2]
  z[, 3] <- 0.40 * z[, 1] + 0.25 * z[, 2] + sqrt(0.78) * z[, 3]
  z[, 4] <- 0.30 * z[, 1] + 0.20 * z[, 2] + sqrt(0.87) * z[, 4]
  z <- z * sd_scale + mean_shift
  out <- data.frame(row.names = seq_len(n))
  for (j in seq_len(p))
    out[[paste0("x", j)]] <- ordered(cut(z[, j], c(-Inf, cuts_by_var[[j]], Inf),
                                         labels = FALSE),
                                     levels = seq_len(length(cuts_by_var[[j]]) + 1L))
  out
}

model   <- "f =~ x1 + x2 + x3 + x4"
ordered <- paste0("x", 1:4)
cuts    <- list(c(-0.70, 0.35), c(-0.55, 0.60), c(-0.85, 0.20), c(-0.45, 0.75))
d1 <- make_ord_df_scaled(420, cuts, seed = 211L, sd_scale = 1.0, mean_shift = 0.00)
d2 <- make_ord_df_scaled(360, cuts, seed = 223L, sd_scale = 1.5, mean_shift = 0.45)
d1$school <- "Pasteur"; d2$school <- "Grant-White"
df       <- rbind(d1, d2)
glabels  <- c("Pasteur", "Grant-White")
ge_metric <- c("thresholds", "loadings")
ge_scalar <- c("thresholds", "loadings", "intercepts")

## ---- ordinal DWLS-theta fits (one shared stats object) ---------------------
spec_cfg <- magmaan::model_spec(model, ordered = ordered, parameterization = "theta",
                                group = "school", group_labels = glabels)
spec_met <- magmaan::model_spec(model, ordered = ordered, parameterization = "theta",
                                group = "school", group_labels = glabels,
                                group_equal = ge_metric)
spec_sca <- magmaan::model_spec(model, ordered = ordered, parameterization = "theta",
                                group = "school", group_labels = glabels,
                                group_equal = ge_scalar)
stats   <- core$data_ordinal_stats_from_df(df, spec_cfg)
fit_cfg <- core$fit_dwls_ordinal(spec_cfg, stats, control = ctrl)
fit_met <- core$fit_dwls_ordinal(spec_met, stats, control = ctrl)
fit_sca <- core$fit_dwls_ordinal(spec_sca, stats, control = ctrl)

lavaan_fit <- function(ge) {
  args <- list(model = model, data = df, ordered = ordered, estimator = "DWLS",
               parameterization = "theta", group = "school",
               se = "robust.sem", test = "satorra.bentler")
  if (!is.null(ge)) args$group.equal <- ge
  do.call(lavaan::cfa, args)
}

check_nested <- function(label, fit_h1, fit_h0, ge_h1, ge_h0, scaled_tol = 5e-3) {
  res <- magmaan::nestedTest(fit_h1, fit_h0, data = stats,
                             method = "satorra.2000", A.method = "delta",
                             weight = "DWLS")
  N <- fit_h1$ntotal; G <- fit_h1$ngroups
  mg_scaled <- res$T_scaled * (N - G) / N # 2N*fmin -> lavaan convention

  lav_h1 <- lavaan_fit(ge_h1)
  lav_h0 <- lavaan_fit(ge_h0)
  lr <- lavaan::lavTestLRT(lav_h1, lav_h0, method = "satorra.2000")
  lav_scaled <- as.numeric(lr[["Chisq diff"]][2])
  lav_df     <- as.integer(lr[["Df diff"]][2])
  lav_p      <- as.numeric(lr[["Pr(>Chisq)"]][2])

  cat(sprintf("== %s Satorra-2000 (delta, theta) ==\n", label))
  cat(sprintf("  scaled diff: magmaan = %.4f, lavaan = %.4f  (|diff| = %.2e)\n",
              mg_scaled, lav_scaled, abs(mg_scaled - lav_scaled)))
  cat(sprintf("  df diff:     magmaan = %d, lavaan = %d\n", res$df_diff, lav_df))
  cat(sprintf("  p (scaled):  magmaan = %.4f, lavaan = %.4f\n", res$p_scaled, lav_p))
  cat(sprintf("  diff spectrum: %s\n",
              paste(sprintf("%.4f", res$eigenvalues), collapse = ", ")))

  stopifnot(identical(as.integer(res$df_diff), lav_df))
  stopifnot(abs(mg_scaled - lav_scaled) < scaled_tol)
  stopifnot(abs(res$p_scaled - lav_p) < 5e-3)

  ## ---- FMG eigenvalue-tail diagnostics on the difference spectrum ----------
  fr <- magmaan::fmg_nested_ordinal(fit_h1, fit_h0, stats, A.method = "delta")
  cat("\n== fmg_nested_ordinal (default tests) ==\n")
  print(fr[, c("label", "base_statistic", "df", "p_value", "chi2_equiv")])

  # Same shape as fmg_tests_ordinal: 5 default rows, all the difference rank,
  # with the unscaled LS difference as the common base statistic.
  stopifnot(nrow(fr) == 5L)
  stopifnot(all(fr$df == res$df_diff))
  stopifnot(all(is.finite(fr$p_value)), all(fr$p_value >= 0), all(fr$p_value <= 1))
  stopifnot(max(abs(fr$base_statistic - res$T_diff)) < 1e-8)
  # The SB row IS the Satorra-Bentler mean scaling of the difference triple, so
  # it reproduces nestedTest()'s satorra.2000 scaled statistic.
  sb <- fr[fr$label == "sb_ls", ]
  stopifnot(nrow(sb) == 1L, abs(sb$chi2_equiv - res$T_scaled) < 1e-6)
  cat("\n")
}

check_nested("configural -> metric", fit_cfg, fit_met, NULL, ge_metric)
check_nested("metric -> scalar", fit_met, fit_sca, ge_metric, ge_scalar,
             scaled_tol = 1.5e-2)

cat("\nnested ordinal group.equal: satorra.2000 + fmg_nested_ordinal vs lavaan: ok\n")
