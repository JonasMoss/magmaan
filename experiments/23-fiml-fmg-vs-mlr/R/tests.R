# FIML fit, the missingness mechanism, and the competing test battery, all over
# magmaan's public surface. The headline competitor is MLR (the Yuan-Bentler
# mean-scaled FIML test, lavaan's `estimator="MLR", missing="ml"` default); the
# proposal is the FMG eigenvalue family; naive is the un-robustified reference.
# The battery is deliberately wide ("all the crappy statistics too") so future
# p-value work can pick from it; the per-replicate UGamma spectrum and base
# statistic are persisted so any new transform is recomputable without refitting.

`%||%` <- function(a, b) if (is.null(a)) b else a

# ── Missingness ──────────────────────────────────────────────────────────────
# Canonical Savalei-Bentler 2005 mechanisms from experiments/_support (the same
# generators exp 08/09 use); applied to the observed columns only, the grouping
# column passed through. MCAR keeps x1,x2 intact; MAR makes missingness on
# x3..x6 depend on the always-observed x1,x2 (so x1,x2 are the MAR cause).
apply_missingness <- function(df, ov, mechanism, rate, seed = NULL) {
  if (identical(mechanism, "complete") || rate <= 0) {
    return(list(df = df, realized = 0.0, mechanism = "complete"))
  }
  X <- df[, ov, drop = FALSE]
  res <- switch(mechanism,
    MCAR = sb2005_mcar(X, rate = rate, intact = 1:2, seed = seed),
    MAR  = sb2005_mar(X, rate = rate, predictors = 1:2, seed = seed,
                      calibrate = TRUE),
    stop("unknown missingness mechanism: ", mechanism, call. = FALSE))
  df[, ov] <- res$data
  list(df = df, realized = res$summary$overall_rate, mechanism = res$mechanism)
}

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

# MLR / Yuan-Bentler scaled FIML test -- the dominant applied default.
mlr_test <- function(fit) {
  m <- tryCatch(magmaan::magmaan_core$estimate_fiml_robust_mlr(fit),
                error = function(e) NULL)
  if (is.null(m) || !is.finite(m$chisq_scaled) || m$df <= 0L) return(NULL)
  list(chisq_scaled = m$chisq_scaled, df = m$df,
       scaling_factor = m$scaling_factor,
       trace = m$trace_ugamma,
       trace_h1 = m$trace_ugamma_h1,
       trace_h0 = m$trace_ugamma_h0,
       p = stats::pchisq(m$chisq_scaled, m$df, lower.tail = FALSE))
}

# The full FIML-valid FMG battery, mapped short-name -> semTests-style test name.
# Under FIML only the biased Gamma + ML base are defined, so no `_ug`/`_rls`.
# Includes the deliberately-imperfect ones (SB/SF/EBA) alongside the better
# spectrum matchers (SS/all) and the penalized family (pEBA/pall/pOLS).
fmg_gof_methods <- function() {
  c(SB = "sb", SS = "ss", SF = "sf",
    EBA2 = "eba2", EBA4 = "eba4", EBA6 = "eba6",
    pEBA2 = "peba2", pEBA4 = "peba4", pEBA6 = "peba6",
    pall = "pall", pOLS = "pols2", all = "all")
}

# FMG goodness-of-fit p-values for one FIML fit, plus the base FIML LRT, df, the
# UGamma spectrum (the sufficient statistic for every eigenvalue transform), and
# its trace. NULL on failure.
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
  spectrum <- tryCatch(as.numeric(tab$eigenvalues[[1L]]), error = function(e) NULL)
  list(p_naive = stats::pchisq(base, df, lower.tail = FALSE),
       p_fmg = p_fmg, df = df, base_stat = base,
       spectrum = spectrum, trace = if (!is.null(spectrum)) sum(spectrum) else NA_real_)
}

# Satorra-2000 nested difference test for H1 (less restricted) over H0.
# p-values: naive / SB-scaled / mean-var-adjusted / exact-mixture, plus the
# difference spectrum (stored for later nested-FMG work). NULL on fail.
fmg_nested <- function(fit_h1, fit_h0) {
  nt <- tryCatch(magmaan::nestedTest(fit_h1, fit_h0, method = "satorra.2000",
                                     A.method = "exact"),
                 error = function(e) NULL)
  if (is.null(nt)) return(NULL)
  spectrum <- tryCatch(as.numeric(nt$eigenvalues), error = function(e) NULL)
  list(p = c(naive = nt$p_unscaled, SB = nt$p_scaled,
             adjusted = nt$p_adjusted, mixture = nt$p_mixture),
       T_diff = nt$T_diff, df_diff = nt$df_diff, scale_c = nt$scale_c,
       spectrum = spectrum)
}

# Draw one replicate, apply the missingness mechanism, fit configural + metric
# under FIML, and compute the GOF (metric model: naive, MLR, full FMG battery)
# and configural-vs-metric nested tests. Returns a rich list including the
# spectra and base statistics (sufficient statistics for offline p-value work),
# or NULL if any fit or test fails.
run_one_rep <- function(pop, sampler, rep_i, mechanism, rate, mask_seed) {
  draw <- sampler$draw(rep_i)
  mm <- apply_missingness(draw, pop$ov, mechanism, rate, seed = mask_seed)
  df <- mm$df
  cfg <- fit_fiml_level("configural", df)
  met <- fit_fiml_level("metric", df)
  if (is.null(cfg) || is.null(met)) return(NULL)

  g    <- fmg_gof(met)
  mlr  <- mlr_test(met)
  nst  <- fmg_nested(cfg, met)
  if (is.null(g) || is.null(mlr) || is.null(nst)) return(NULL)

  yb_exact <- unname(g$p_fmg[["SB"]])
  sat_p <- c(naive = g$p_naive, MLR = mlr$p, YB_mplus = mlr$p,
             YB_exact = yb_exact, g$p_fmg)
  sat_method <- names(sat_p)
  sat_trace <- rep(g$trace, length(sat_p))
  sat_trace[sat_method %in% c("MLR", "YB_mplus")] <- mlr$trace
  sat_scale <- sat_trace / g$df
  sat_trace_h1 <- rep(NA_real_, length(sat_p))
  sat_trace_h0 <- rep(NA_real_, length(sat_p))
  sat_trace_h1[sat_method %in% c("MLR", "YB_mplus")] <- mlr$trace_h1
  sat_trace_h0[sat_method %in% c("MLR", "YB_mplus")] <- mlr$trace_h0
  gof_sat <- data.frame(
    outcome = "gof",
    method = sat_method,
    p_value = unname(sat_p),
    base_stat = g$base_stat,
    df = g$df,
    trace = sat_trace,
    scaling_factor = sat_scale,
    trace_h1 = sat_trace_h1,
    trace_h0 = sat_trace_h0,
    h1_information = "saturated",
    realized_rate = mm$realized,
    stringsAsFactors = FALSE)
  # The structured-H1 FMG variant was removed from magmaan (2026-06-24): the
  # model-implied curvature is not guaranteed PD off H0 and never beat the
  # saturated convention here. Saturated H1 is the only path now.
  gof <- gof_sat
  nested <- data.frame(outcome = "nested", method = names(nst$p),
                       p_value = unname(nst$p), base_stat = nst$T_diff,
                       df = nst$df_diff, trace = if (!is.null(nst$spectrum))
                         sum(nst$spectrum) else NA_real_,
                       scaling_factor = nst$scale_c,
                       trace_h1 = NA_real_,
                       trace_h0 = NA_real_,
                       h1_information = NA_character_,
                       realized_rate = mm$realized, stringsAsFactors = FALSE)
  list(gof = gof, nested = nested,
       gof_spectrum = g$spectrum,
       nested_spectrum = nst$spectrum,
       mlr_scaled = mlr$chisq_scaled, mlr_factor = mlr$scaling_factor)
}

# Rejection rate at level alpha over a numeric p-value vector (NA-dropping).
rejection_rate <- function(p, alpha = 0.05) {
  p <- p[is.finite(p)]
  if (!length(p)) return(NA_real_)
  mean(p < alpha)
}
