# Fit + test battery for the invariance difference-test study, all over
# magmaan's public surface. Per fit we harvest every eigenvalue-spectrum p-value
# (cheap once the spectrum is formed); per adjacent ladder pair we harvest the
# Satorra-2000 difference test plus its difference spectrum.

`%||%` <- function(a, b) if (is.null(a)) b else a

# ── Missingness (Savalei-Bentler 2005, from experiments/_support) ─────────────
# MCAR keeps x1,x2 intact; MAR makes missingness on x3..xp depend on the
# always-observed x1,x2 (the MAR cause).
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

fiml_control <- function() list(max_iter = 16000L, ftol = 1e-13, gtol = 1e-9)

# Fit one invariance level under the requested estimator. NULL on any failure.
# `em` is the shared saturated EM moments for this masked dataset (see
# run_one_rep): the saturated H1 is identical across all four rungs and both
# estimators, so it is built ONCE per replicate and threaded in here -- ML2S
# skips its Stage-1 EM (`stage1 = em`) and the FIML fit is stamped (`fit$stage1`)
# so its FMG battery + nested test reuse it instead of rebuilding. `em = NULL`
# falls back to the per-fit rebuild (correct, just slower).
fit_level <- function(level, df, pop, estimator = c("FIML", "ML2S"),
                      control = NULL, em = NULL) {
  estimator <- match.arg(estimator)
  control <- control %||% fiml_control()
  spec <- magmaan::model_spec(invariance_syntax(level, pop$ov),
                              group = "school", group_labels = c("A", "B"),
                              meanstructure = TRUE)
  tryCatch({
    fd <- magmaan::df_to_fiml_data(df, spec)
    if (estimator == "FIML") {
      fit <- magmaan::magmaan_core$fit_fiml(spec, fd, control = control)
      if (!is.null(em) && !is.null(fit)) fit$stage1 <- em
    } else {
      fit <- magmaan::magmaan_core$fit_ml2s(spec, fd, control = control,
                                            stage1 = em)
    }
    if (!isTRUE(fit$converged)) return(NULL)
    fit
  }, error = function(e) NULL)
}

# The eigenvalue-spectrum battery (semTests short names).
fmg_methods <- function() {
  c(SB = "sb", SS = "ss", SF = "sf",
    EBA2 = "eba2", EBA4 = "eba4", EBA6 = "eba6",
    pEBA2 = "peba2", pEBA4 = "peba4", pEBA6 = "peba6",
    pall = "pall", pOLS = "pols2", all = "all")
}

# MLR / Yuan-Bentler scaled FIML test (FIML only; the applied default).
mlr_p <- function(fit) {
  m <- tryCatch(magmaan::magmaan_core$estimate_fiml_robust_mlr(fit),
                error = function(e) NULL)
  if (is.null(m) || !is.finite(m$chisq_scaled) || m$df <= 0L) return(NULL)
  stats::pchisq(m$chisq_scaled, m$df, lower.tail = FALSE)
}

# GOF p-values for one fit -> long rows (one per method). `mlr` adds the MLR row.
gof_rows <- function(fit, estimator, rung, h1 = "saturated", add_mlr = FALSE) {
  methods <- fmg_methods()
  tab <- tryCatch(magmaan::fmg_tests(fit, tests = names(methods),
                                     h1_information = h1),
                  error = function(e) NULL)
  if (is.null(tab) || !nrow(tab)) return(NULL)
  # fmg_tests returns the canonical lowercase code in `label` (e.g. "ss_ml");
  # `input` echoes the names we passed. Match on the stripped lowercase code.
  key  <- if ("label" %in% names(tab)) sub("_ml$", "", tab$label) else tab$input
  base <- tab$base_statistic[1L]; df <- tab$df[1L]
  p_of <- function(code) { hit <- which(key == code); if (length(hit)) tab$p_value[hit[1L]] else NA_real_ }
  rows <- data.frame(
    estimator = estimator, rung = rung, outcome = "gof", h1_information = h1,
    method = c("naive", names(methods)),
    p_value = c(stats::pchisq(base, df, lower.tail = FALSE),
                vapply(unname(methods), p_of, numeric(1))),
    base_stat = base, df = df,
    stringsAsFactors = FALSE)
  if (add_mlr && identical(h1, "saturated")) {
    p <- mlr_p(fit)
    if (!is.null(p)) rows <- rbind(rows, data.frame(
      estimator = estimator, rung = rung, outcome = "gof",
      h1_information = h1, method = "MLR", p_value = p,
      base_stat = base, df = df, stringsAsFactors = FALSE))
  }
  attr(rows, "spectrum") <- tryCatch(as.numeric(tab$eigenvalues[[1L]]),
                                     error = function(e) NULL)
  rows
}

# Satorra-2000 nested difference test for a ladder step -> long rows. Returns the
# documented p-values (naive / SB / mean-var-adjusted / mixture); pEBA-on-the-
# difference is the wiring crux (see report.qmd) -- `peba_diff_available()`
# reports whether nestedTest already exposes it.
nested_rows <- function(fit_h1, fit_h0, estimator, step) {
  nt <- tryCatch(magmaan::nestedTest(fit_h1, fit_h0, method = "satorra.2000",
                                     A.method = "exact"),
                 error = function(e) NULL)
  if (is.null(nt)) return(NULL)
  p <- c(naive = nt$p_unscaled, SB = nt$p_scaled,
         adjusted = nt$p_adjusted, mixture = nt$p_mixture)
  rows <- data.frame(
    estimator = estimator, rung = step, outcome = "nested",
    h1_information = "saturated", method = names(p), p_value = unname(p),
    base_stat = nt$T_diff %||% NA_real_, df = nt$df_diff %||% NA_real_,
    stringsAsFactors = FALSE)
  attr(rows, "spectrum") <- tryCatch(as.numeric(nt$eigenvalues),
                                     error = function(e) NULL)
  rows
}

# One-time wiring probe: does nestedTest expose pEBA-family difference p-values?
peba_diff_available <- function(nt) {
  any(grepl("peba|eba|p_mixture", names(nt), ignore.case = TRUE))
}

# One replicate: draw + mask, then for each estimator fit the full ladder, harvest
# GOF per rung and the nested difference test per adjacent pair. Both estimators
# see the SAME masked dataset (paired design). TOLERANT: a single failed step does
# not nuke the replicate -- we skip that estimator if a fit fails, and skip an
# individual gof/nested piece if it fails, returning whatever computed. NULL only
# if nothing at all was produced.
#
# KNOWN GAP harvested by the tolerance: the metric->scalar ("strong") nested test
# currently fails because that step is NOT a strict parameter-subset (scalar ties
# intercepts AND frees the group-2 latent mean, so npar(scalar) > npar(metric));
# magmaan's exact Satorra-2000 route needs npar(H0) < npar(H1). This is the
# mean-structure nested-comparison wiring item (see report.qmd); the weak and
# strict steps are clean restrictions and work.
run_one_rep <- function(pop, sampler, rep_i, mechanism, rate, mask_seed,
                        estimators = c("FIML", "ML2S")) {
  df <- sampler$draw(rep_i)
  mm <- apply_missingness(df, pop$ov, mechanism, rate, seed = mask_seed)
  df <- mm$df
  # The saturated EM moments depend only on the masked data -- identical across
  # all four rungs AND both estimators -- so build them ONCE here and thread
  # them through every fit/battery/nested test (see fit_level). All four specs
  # share pop$ov order and the group structure, so one EM is valid for all; a
  # build failure leaves em = NULL and each fit falls back to its own rebuild.
  em <- tryCatch({
    spec0 <- magmaan::model_spec(invariance_syntax("configural", pop$ov),
                                 group = "school", group_labels = c("A", "B"),
                                 meanstructure = TRUE)
    magmaan::magmaan_core$estimate_saturated_em_moments(
      magmaan::df_to_fiml_data(df, spec0))
  }, error = function(e) NULL)
  pairs <- ladder_pairs()
  all_rows <- list()
  push <- function(x) if (!is.null(x)) all_rows[[length(all_rows) + 1L]] <<- x
  for (est in estimators) {
    fits <- lapply(c("configural", "metric", "scalar", "strict"),
                   fit_level, df = df, pop = pop, estimator = est, em = em)
    names(fits) <- c("configural", "metric", "scalar", "strict")
    if (any(vapply(fits, is.null, logical(1)))) next   # skip estimator, keep others
    for (lvl in c("metric", "scalar", "strict"))        # GOF of constrained models
      push(gof_rows(fits[[lvl]], est, lvl, add_mlr = (est == "FIML")))
    for (pr in pairs)                                   # nested difference tests
      push(nested_rows(fits[[pr$h1]], fits[[pr$h0]], est, pr$step))
  }
  if (!length(all_rows)) return(NULL)
  out <- do.call(rbind, c(all_rows, list(make.row.names = FALSE)))
  out$realized_rate <- mm$realized
  out
}

rejection_rate <- function(p, alpha = 0.05) mean(p < alpha, na.rm = TRUE)
