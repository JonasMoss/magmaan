## magmaan R bindings -- ordinal DWLS Satorra-2000 nested-model test.
##
## Run from the repo root after installing the R package:
##     Rscript r-package/examples/nested_test_ordinal.R
##
## The lavaan oracle is WLSMV estimation with
## `lavTestLRT(..., method = "satorra.2000", A.method = "exact",
## scaled.shifted = FALSE)`. magmaan fits the same polychoric DWLS objective
## and computes the nested spectrum from the ordinal moment Jacobian, DWLS
## weight, polychoric NACOV, and the exact parameter restriction map.
## For ordinal WLSMV and m > 1, lavaan's displayed row is the
## mean/variance-adjusted statistic (`T_adjusted`, `d0`), not magmaan's
## integer-rank mean-scaled `T_scaled` row.

suppressMessages({ library(magmaan); library(lavaan) })
core <- magmaan::magmaan_core

make_ord_cfa <- function(n, lambda, cuts, seed) {
  set.seed(seed)
  eta <- rnorm(n)
  out <- data.frame(row.names = seq_len(n))
  for (j in seq_along(lambda)) {
    y <- lambda[[j]] * eta + sqrt(1 - lambda[[j]]^2) * rnorm(n)
    out[[paste0("x", j)]] <- ordered(cut(y, c(-Inf, cuts[[j]], Inf),
                                         labels = FALSE))
  }
  out
}

fit_mg_pair <- function(df, h1_model, h0_model, ordered, group = NULL,
                        group_labels = character()) {
  spec_args <- list(ordered = ordered, parameterization = "delta")
  if (!is.null(group)) {
    spec_args$group <- group
    spec_args$group_labels <- group_labels
  }
  spec_H1 <- do.call(magmaan::model_spec, c(list(syntax = h1_model), spec_args))
  spec_H0 <- do.call(magmaan::model_spec, c(list(syntax = h0_model), spec_args))
  stats <- core$data_ordinal_stats_from_df(df, spec_H1)
  fit_H1 <- core$fit_dwls_ordinal(
    spec_H1, stats, control = list(max_iter = 4000, ftol = 1e-13,
                                   gtol = 1e-8))
  fit_H0 <- core$fit_dwls_ordinal(
    spec_H0, stats, control = list(max_iter = 4000, ftol = 1e-13,
                                   gtol = 1e-8))
  list(stats = stats, fit_H1 = fit_H1, fit_H0 = fit_H0)
}

check_lrt <- function(label, mg, lav_H1, lav_H0,
                      stat_tol = 8e-2, shape_tol = 2e-3,
                      p_tol = 1e-2) {
  res <- magmaan::nestedTest(
    mg$fit_H1, mg$fit_H0, data = mg$stats, method = "satorra.2000",
    A.method = "exact", weight = "DWLS")
  lav <- lavaan::lavTestLRT(
    lav_H1, lav_H0, method = "satorra.2000", A.method = "exact",
    scaled.shifted = FALSE)

  lav_df <- as.numeric(lav[2, "Df diff"])
  lav_T <- as.numeric(lav[2, "Chisq diff"])
  lav_p <- as.numeric(lav[2, "Pr(>Chisq)"])
  lav_scale <- as.numeric(attr(lav, "scale")[[2]])
  spectrum_scale <- sum(res$eigenvalues) / res$adjust_d0

  cat("\n== ", label, " ==\n", sep = "")
  cat(sprintf("rank m:         magmaan = %d\n", res$df_diff))
  cat(sprintf("adjusted df d0: magmaan = %.8f, lavaan = %.8f\n",
              res$adjust_d0, lav_df))
  cat(sprintf("adjusted diff:  magmaan = %.8f, lavaan = %.8f\n",
              res$T_adjusted, lav_T))
  cat(sprintf("scale trace/d0: magmaan = %.8f, lavaan = %.8f\n",
              spectrum_scale, lav_scale))
  cat(sprintf("adjusted p:     magmaan = %.8g, lavaan = %.8g\n",
              res$p_adjusted, lav_p))
  cat("spectrum:       ",
      paste(format(res$eigenvalues, digits = 8), collapse = ", "), "\n",
      sep = "")

  stopifnot(isTRUE(all.equal(res$adjust_d0, lav_df, tolerance = shape_tol)))
  stopifnot(isTRUE(all.equal(res$T_adjusted, lav_T, tolerance = stat_tol)))
  stopifnot(isTRUE(all.equal(spectrum_scale, lav_scale,
                             tolerance = shape_tol)))
  stopifnot(isTRUE(all.equal(res$p_adjusted, lav_p, tolerance = p_tol)))
  if (length(res$eigenvalues) == 1L) {
    stopifnot(isTRUE(all.equal(unname(res$eigenvalues[[1]]), lav_scale,
                               tolerance = shape_tol)))
  }
  invisible(res)
}

ordered <- paste0("x", 1:4)
cuts <- list(c(-0.75, 0.05, 0.85), c(-0.65, 0.10, 0.75),
             c(-0.80, 0.00, 0.70), c(-0.70, 0.15, 0.90))
h1 <- "f =~ x1 + x2 + x3 + x4"

## --- single group: one loading-equality restriction ------------------------
single <- make_ord_cfa(650, c(0.82, 0.70, 0.55, 0.64), cuts, seed = 101L)
h0_single <- "f =~ x1 + eq*x2 + eq*x3 + x4"
mg_single <- fit_mg_pair(single, h1, h0_single, ordered)
lav_single_H1 <- lavaan::cfa(h1, data = single, ordered = ordered,
                             estimator = "WLSMV",
                             parameterization = "delta")
lav_single_H0 <- lavaan::cfa(h0_single, data = single, ordered = ordered,
                             estimator = "WLSMV",
                             parameterization = "delta")
check_lrt("single-group ordinal loading equality", mg_single,
          lav_single_H1, lav_single_H0)

## --- two groups: configural versus metric invariance -----------------------
g1 <- make_ord_cfa(520, c(0.82, 0.70, 0.55, 0.64), cuts, seed = 202L)
g2 <- make_ord_cfa(480, c(0.82, 0.58, 0.72, 0.50), cuts, seed = 303L)
g1$grp <- factor("g1", levels = c("g1", "g2"))
g2$grp <- factor("g2", levels = c("g1", "g2"))
two_group <- rbind(g1, g2)
h0_metric <- "f =~ x1 + L2*x2 + L3*x3 + L4*x4"
mg_metric <- fit_mg_pair(two_group, h1, h0_metric, ordered,
                         group = "grp", group_labels = c("g1", "g2"))
lav_cfg <- lavaan::cfa(h1, data = two_group, ordered = ordered,
                       group = "grp", estimator = "WLSMV",
                       parameterization = "delta")
lav_met <- lavaan::cfa(h1, data = two_group, ordered = ordered,
                       group = "grp", group.equal = "loadings",
                       estimator = "WLSMV", parameterization = "delta")
check_lrt("two-group ordinal configural vs metric", mg_metric,
          lav_cfg, lav_met)

cat("\nordinal nestedTest() Satorra-2000 workflow: ok\n")
