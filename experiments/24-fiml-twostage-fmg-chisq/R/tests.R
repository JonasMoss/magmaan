# One replicate: fit the correct one-factor model by FIML and by two-stage ML
# (ML2S), then run the full goodness-of-fit-statistic battery on each. Every
# statistic is a transform of the same (base chi-square, df, UGamma spectrum)
# triple, so the battery is one fmg_tests() call per estimator. Because the model
# is correct, every rejection is a Type-I error.

`%||%` <- function(a, b) if (is.null(a)) b else a

# The statistic battery, by camp:
#   Savalei / low-moment matches : std (naive), sb (mean-scaled = T_RML),
#                                  ss (scaled-shifted), sf (scaled-F)
#   FMG full-spectrum            : all (exact Imhof mixture), pall (penalized),
#                                  pEBA2/4/6 (penalized eigenvalue-block),
#                                  pOLS (penalized OLS tail)
fmg_battery_tests <- function() {
  c("std", "sb", "ss", "sf", "all", "pall", "peba2", "peba4", "peba6", "pols")
}

# ── Missingness ──────────────────────────────────────────────────────────────
# Canonical Savalei-Bentler 2005 mechanisms from experiments/_support. MCAR keeps
# x1,x2 intact; MAR makes missingness on x3..x6 depend on the always-observed
# x1,x2 (so x1,x2 are the MAR cause).
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

# ── Fits ─────────────────────────────────────────────────────────────────────
fiml_control <- function() list(max_iter = 16000L, ftol = 1e-13, gtol = 1e-9)

fit_fiml_model <- function(spec, df, control = NULL) {
  control <- control %||% fiml_control()
  tryCatch({
    fit <- magmaan::magmaan_core$fit_fiml(
      spec, magmaan::df_to_fiml_data(df, spec), control = control)
    if (!isTRUE(fit$converged)) return(NULL)
    fit
  }, error = function(e) NULL)
}

fit_ml2s_model <- function(spec, df, control = NULL) {
  control <- control %||% fiml_control()
  tryCatch({
    fit <- magmaan::magmaan_core$fit_ml2s(
      spec, magmaan::df_to_fiml_data(df, spec), control = control)
    if (!isTRUE(fit$converged)) return(NULL)
    fit
  }, error = function(e) NULL)
}

# ── Battery ──────────────────────────────────────────────────────────────────
# Run the eigenvalue-tail battery on one fit. The base statistic is the FIML LRT
# (FIML) or the Stage-2 ML chi-square (ML2S); the UGamma spectrum is the same for
# every method, so its sum (the reference-law mean = the consistency anchor) and
# the spectrum itself are recorded once.
fmg_battery <- function(fit, tests = fmg_battery_tests()) {
  tab <- tryCatch(magmaan::fmg_tests(fit, tests = tests),
                  error = function(e) NULL)
  if (is.null(tab) || !nrow(tab)) return(NULL)
  spectrum <- tryCatch(as.numeric(tab$eigenvalues[[1L]]), error = function(e) NULL)
  if (is.null(spectrum) || !all(is.finite(tab$p_value))) return(NULL)
  list(
    rows = data.frame(method = tab$input, p_value = tab$p_value,
                      base_stat = tab$base_statistic, df = tab$df,
                      stringsAsFactors = FALSE),
    trace = sum(spectrum),
    spectrum = spectrum)
}

# ── One replicate ────────────────────────────────────────────────────────────
# Paired design: both estimators see the SAME generated-and-masked dataset, so
# the FIML-vs-ML2S contrast is free of between-sample noise. A replicate counts
# only if BOTH fits converge and both batteries succeed (NULL otherwise).
run_one_rep <- function(spec, sampler, rep_i, mechanism, rate, mask_seed) {
  df <- sampler$draw(rep_i)
  ov <- colnames(df)
  mm <- apply_missingness(df, ov, mechanism, rate, seed = mask_seed)

  fits <- list(FIML = fit_fiml_model(spec, mm$df),
               ML2S = fit_ml2s_model(spec, mm$df))
  if (is.null(fits$FIML) || is.null(fits$ML2S)) return(NULL)

  bat <- lapply(fits, fmg_battery)
  if (is.null(bat$FIML) || is.null(bat$ML2S)) return(NULL)

  rows <- do.call(rbind, lapply(names(bat), function(est) {
    b <- bat[[est]]
    data.frame(estimator = est, b$rows, trace = b$trace,
               realized_rate = mm$realized, stringsAsFactors = FALSE)
  }))
  list(rows = rows,
       spectra = list(FIML = bat$FIML$spectrum, ML2S = bat$ML2S$spectrum))
}

# Empirical rejection rate at level alpha (NA-safe).
rejection_rate <- function(p, alpha = 0.05) mean(p < alpha, na.rm = TRUE)
