# lavaan oracle helpers for the FIML measurement-invariance FMG experiment.
# These produce the reference quantities magmaan is checked against: the FIML
# LRT chi-square, the robust UGamma spectrum (eigenvalues / trace), and the
# scaled nested difference test. No magmaan code here.

# Fit one invariance level with lavaan. `missing` is "fiml" or "listwise".
# h1.information = "unstructured" matches magmaan's FIML saturated-h1 convention.
# lavaan forces listwise under the satorra.bentler test, so missing data uses
# the Yuan-Bentler robust test (the missing-data analogue) to stay on FIML.
lavaan_fit_level <- function(syntax, df, level, missing = "listwise") {
  test <- if (identical(missing, "fiml")) "yuan.bentler" else "satorra.bentler"
  lavaan::cfa(
    syntax, data = df, group = "school",
    group.equal = lavaan_group_equal(level),
    missing = missing, estimator = "ML",
    test = test, meanstructure = TRUE,
    h1.information = "unstructured")
}

# Sorted nonzero eigenvalues of lavaan's robust UGamma, plus its trace and df.
# Returns NULL eigenvalues when lavInspect cannot form UGamma (some missing-data
# configurations), in which case only the trace (from the test slot) is used.
lavaan_ugamma <- function(fit) {
  ev <- tryCatch({
    ug <- lavaan::lavInspect(fit, "UGamma")
    sort(Re(eigen(ug, only.values = TRUE)$values), decreasing = TRUE)
  }, error = function(e) NULL)
  trace <- tryCatch({
    ts <- lavaan::lavInspect(fit, "test")
    ts[[2]]$trace.UGamma %||% sum(ev[ev > 1e-8])
  }, error = function(e) if (!is.null(ev)) sum(ev[ev > 1e-8]) else NA_real_)
  df <- tryCatch(unname(lavaan::fitMeasures(fit, "df")), error = function(e) NA_real_)
  if (!is.null(ev)) ev <- ev[ev > 1e-8]
  list(eigenvalues = ev, trace = trace, df = df,
       chisq = unname(lavaan::fitMeasures(fit, "chisq")))
}

# Scaled nested difference test (lavaan SB-2001 approximation) on two MLR fits.
# magmaan's restriction map is Satorra-2000; the two conventions differ slightly
# and the gap is reported, not asserted.
lavaan_nested <- function(fit_h1, fit_h0) {
  out <- tryCatch({
    lrt <- lavaan::lavTestLRT(fit_h0, fit_h1, method = "satorra.bentler.2001")
    list(chisq_diff = lrt[["Chisq diff"]][2],
         df_diff    = lrt[["Df diff"]][2],
         p_value    = lrt[["Pr(>Chisq)"]][2])
  }, error = function(e) list(chisq_diff = NA_real_, df_diff = NA_real_,
                              p_value = NA_real_))
  out
}

# Align two ascending/descending spectra to the shorter length and return the
# max absolute elementwise difference (both sorted descending first).
spectrum_maxabs <- function(a, b) {
  if (is.null(a) || is.null(b) || !length(a) || !length(b)) return(NA_real_)
  a <- sort(a, decreasing = TRUE); b <- sort(b, decreasing = TRUE)
  k <- min(length(a), length(b))
  max(abs(a[seq_len(k)] - b[seq_len(k)]))
}
