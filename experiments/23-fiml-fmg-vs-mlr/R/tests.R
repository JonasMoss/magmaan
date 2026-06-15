# FIML fit plus the competing goodness-of-fit / nested test statistics, all over
# magmaan's public surface. The headline competitor is MLR (the Yuan-Bentler
# mean-scaled FIML test, lavaan's `estimator = "MLR", missing = "ml"` default);
# the proposal is the FMG eigenvalue family (SS plus pEBA / penalized-all). The
# naive FIML LRT is kept only as the un-robustified reference.

`%||%` <- function(a, b) if (is.null(a)) b else a

# Fit one invariance level under FIML. Returns the fit, or NULL on any failure.
fit_fiml_level <- function(level, df, control = NULL) {
  control <- control %||% list(max_iter = 16000L, ftol = 1e-13, gtol = 1e-9)
  spec <- magmaan::model_spec(invariance_syntax(level), group = "school",
                              group_labels = c("A", "B"), meanstructure = TRUE)
  tryCatch({
    fit <- magmaan::magmaan_core$fit_fiml(
      spec, magmaan::df_to_fiml_data(df, spec, group = "school"),
      control = control)
    if (!isTRUE(fit$converged)) return(NULL)
    fit
  }, error = function(e) NULL)
}

# MLR / Yuan-Bentler scaled FIML test — the dominant applied default. The C++
# `fiml_robust_mlr` corner builds the Huber-White sandwich and returns the scaled
# statistic; the p-value references it to chi-square(df). NULL on failure.
mlr_test <- function(fit) {
  m <- tryCatch(magmaan::magmaan_core$estimate_fiml_robust_mlr(fit),
                error = function(e) NULL)
  if (is.null(m) || !is.finite(m$chisq_scaled) || m$df <= 0L) return(NULL)
  list(chisq_scaled = m$chisq_scaled, df = m$df,
       scaling_factor = m$scaling_factor,
       p = stats::pchisq(m$chisq_scaled, m$df, lower.tail = FALSE))
}

# FMG goodness-of-fit methods requested from magmaan, mapped to short column
# names. `fmg_tests()` returns FIML labels suffixed `_ml`; we strip that.
fmg_gof_methods <- function() {
  c(SB = "sb", SS = "ss", pEBA2 = "peba2", pEBA4 = "peba4", pEBA6 = "peba6",
    pall = "pall", all = "all")
}

# FMG goodness-of-fit p-values for one FIML fit. Returns the named FMG p-value
# vector, the naive LRT p-value, the df, the base FIML LRT statistic, and the
# biased UGamma spectrum. NULL on failure.
fmg_gof <- function(fit, methods = fmg_gof_methods()) {
  tab <- tryCatch(magmaan::fmg_tests(fit, tests = names(methods)),
                  error = function(e) NULL)
  if (is.null(tab) || !nrow(tab)) return(NULL)
  base <- tab$base_statistic[1L]
  df   <- tab$df[1L]
  key  <- sub("_ml$", "", tab$label)
  p_fmg <- vapply(methods, function(m) {
    hit <- which(key == m)
    if (length(hit)) tab$p_value[hit[1L]] else NA_real_
  }, numeric(1))
  names(p_fmg) <- names(methods)
  p_naive <- stats::pchisq(base, df, lower.tail = FALSE)
  spectrum <- tryCatch(as.numeric(tab$eigenvalues[[1L]]), error = function(e) NULL)
  list(p_naive = p_naive, p_fmg = p_fmg, df = df, base_stat = base,
       spectrum = spectrum,
       trace_xcheck = attr(tab, "trace_xcheck") %||% NA_real_)
}

# Satorra-2000 nested difference test for H1 (less restricted) over H0.
# p-values: naive / SB-scaled / mean-var-adjusted / exact-mixture. NULL on fail.
fmg_nested <- function(fit_h1, fit_h0) {
  nt <- tryCatch(magmaan::nestedTest(fit_h1, fit_h0, method = "satorra.2000",
                                     A.method = "exact"),
                 error = function(e) NULL)
  if (is.null(nt)) return(NULL)
  list(p = c(naive = nt$p_unscaled, SB = nt$p_scaled,
             adjusted = nt$p_adjusted, mixture = nt$p_mixture),
       T_diff = nt$T_diff, df_diff = nt$df_diff, scale_c = nt$scale_c)
}

# Draw one replicate, inject MCAR, fit configural + metric under FIML, and
# compute the goodness-of-fit (metric model: naive, MLR, FMG family) and the
# configural-vs-metric nested test. Returns list(gof, nested) of long-form one-
# row-per-method data frames, or NULL if any fit or test fails.
run_one_rep <- function(pop, sampler, rep_i, miss, mcar_seed) {
  df <- sampler$draw(rep_i)
  if (miss > 0) { set.seed(mcar_seed); df <- inject_mcar(df, pop$ov, miss) }
  cfg <- fit_fiml_level("configural", df)
  met <- fit_fiml_level("metric", df)
  if (is.null(cfg) || is.null(met)) return(NULL)

  g   <- fmg_gof(met)
  mlr <- mlr_test(met)
  nst <- fmg_nested(cfg, met)
  if (is.null(g) || is.null(mlr) || is.null(nst)) return(NULL)

  gof_p <- c(naive = g$p_naive, MLR = mlr$p, g$p_fmg)
  gof <- data.frame(outcome = "gof", method = names(gof_p),
                    p_value = unname(gof_p), df = g$df, base_stat = g$base_stat,
                    stringsAsFactors = FALSE)
  nested <- data.frame(outcome = "nested", method = names(nst$p),
                       p_value = unname(nst$p), df = nst$df_diff,
                       base_stat = nst$T_diff, stringsAsFactors = FALSE)
  list(gof = gof, nested = nested)
}

# Rejection rate at level alpha over a numeric p-value vector (NA-dropping).
rejection_rate <- function(p, alpha = 0.05) {
  p <- p[is.finite(p)]
  if (!length(p)) return(NA_real_)
  mean(p < alpha)
}
