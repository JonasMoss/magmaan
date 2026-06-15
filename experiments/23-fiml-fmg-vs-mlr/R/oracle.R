# lavaan correctness oracle. The decisive check is that magmaan reproduces what
# an applied lavaan user actually gets from the dominant default,
# `estimator = "MLR", missing = "ml"` (Yuan-Bentler mean-scaled FIML test): the
# scaled chi-square and its p-value. We also confirm the FIML LRT base statistic
# and, on complete data, the UGamma spectrum that every FMG transform consumes.
# No magmaan-internal calls; the magmaan side comes from the public test wrappers.

# Dominant-default MLR fit: lavaan's own `estimator = "MLR"` conventions
# (h1.information = "structured", information = "observed", test =
# "yuan.bentler.mplus", se = "robust.huber.white"), left at their defaults on
# purpose -- that is literally what an applied user runs, and magmaan's
# `fiml_robust_mlr` reproduces this scaled statistic to machine precision.
# `missing` is "ml" (FIML) or "listwise" (complete data).
lavaan_mlr_fit <- function(syntax, df, level, missing = "ml") {
  lavaan::cfa(
    syntax, data = df, group = "school",
    group.equal = lavaan_group_equal(level),
    missing = missing, estimator = "MLR", meanstructure = TRUE)
}

# Separate fit for the FMG UGamma spectrum parity only: magmaan's FMG
# eigenvalues use the saturated (unstructured-h1) missing-data Gamma, so the
# spectrum is checked against lavaan's unstructured Yuan-Bentler UGamma, the same
# convention validated in experiment 21. This is NOT the statistic a user reads;
# it only gates the eigenvalues the FMG transforms consume.
lavaan_spectrum_fit <- function(syntax, df, level, missing = "ml") {
  lavaan::cfa(
    syntax, data = df, group = "school",
    group.equal = lavaan_group_equal(level),
    missing = missing, estimator = "ML", test = "yuan.bentler",
    meanstructure = TRUE, h1.information = "unstructured")
}

# Pull lavaan's naive and scaled chi-square / p-value for the fit.
lavaan_mlr_measures <- function(fit) {
  fm <- tryCatch(lavaan::fitMeasures(
    fit, c("chisq", "df", "pvalue", "chisq.scaled", "df.scaled",
           "pvalue.scaled")), error = function(e) NULL)
  if (is.null(fm)) return(NULL)
  list(chisq = unname(fm["chisq"]), df = unname(fm["df"]),
       p_naive = unname(fm["pvalue"]),
       chisq_scaled = unname(fm["chisq.scaled"]),
       df_scaled = unname(fm["df.scaled"]),
       p_scaled = unname(fm["pvalue.scaled"]))
}

# Sorted nonzero eigenvalues of lavaan's robust UGamma (complete data only).
lavaan_ugamma_spectrum <- function(fit) {
  ev <- tryCatch({
    ug <- lavaan::lavInspect(fit, "UGamma")
    sort(Re(eigen(ug, only.values = TRUE)$values), decreasing = TRUE)
  }, error = function(e) NULL)
  if (is.null(ev)) return(NULL)
  ev[ev > 1e-8]
}

# Max absolute elementwise difference of two spectra, aligned to shorter length.
spectrum_maxabs <- function(a, b) {
  if (is.null(a) || is.null(b) || !length(a) || !length(b)) return(NA_real_)
  a <- sort(a, decreasing = TRUE); b <- sort(b, decreasing = TRUE)
  k <- min(length(a), length(b))
  max(abs(a[seq_len(k)] - b[seq_len(k)]))
}

.parity_row <- function(base, metric, magmaan, lavaan) {
  data.frame(c(base, list(metric = metric, magmaan = magmaan, lavaan = lavaan,
                          abs_diff = abs(magmaan - lavaan))),
             stringsAsFactors = FALSE)
}

# Parity rows for one level: the magmaan side is the fmg_gof()/mlr_test() output
# for the metric (or configural) FIML fit; the lavaan side is the MLR fit on the
# same data. Compares the FIML LRT base, the MLR scaled statistic, and the MLR
# scaled p-value; on complete data, the UGamma spectrum too.
parity_rows_for_level <- function(g, mlr, df, level, miss, dist, truth) {
  miss_lav <- if (miss > 0) "ml" else "listwise"
  lav <- tryCatch(lavaan_mlr_fit(invariance_syntax(level), df, level, miss_lav),
                  error = function(e) NULL)
  if (is.null(lav)) return(list())
  lm <- lavaan_mlr_measures(lav)
  if (is.null(lm)) return(list())
  base <- list(condition = truth, dist = dist, miss = miss, level = level)
  rows <- list(
    .parity_row(base, "fiml_lrt_chisq",     g$base_stat,     lm$chisq),
    .parity_row(base, "mlr_chisq_scaled",   mlr$chisq_scaled, lm$chisq_scaled),
    .parity_row(base, "mlr_pvalue_scaled",  mlr$p,           lm$p_scaled))
  if (miss == 0) {
    lav_sp <- tryCatch(lavaan_spectrum_fit(invariance_syntax(level), df, level,
                                           miss_lav), error = function(e) NULL)
    lev_ev <- if (!is.null(lav_sp)) lavaan_ugamma_spectrum(lav_sp) else NULL
    if (!is.null(lev_ev)) {
      spec_row <- data.frame(c(base, list(
        metric = "fiml_ugamma_spectrum_maxabs", magmaan = NA_real_,
        lavaan = NA_real_, abs_diff = spectrum_maxabs(g$spectrum, lev_ev))),
        stringsAsFactors = FALSE)
      rows <- c(rows, list(spec_row))
    }
  }
  rows
}
