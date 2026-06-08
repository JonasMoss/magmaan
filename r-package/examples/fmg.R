library(magmaan)

utils::data("HolzingerSwineford1939", package = "lavaan")

df <- as.data.frame(HolzingerSwineford1939)
model <- "visual  =~ x1 + x2 + x3
          textual =~ x4 + x5 + x6
          speed   =~ x7 + x8 + x9"

fit <- magmaan(model, df, estimator = "ML", se = "none", test = "none")
stopifnot(inherits(fit$raw_data, "magmaan_complete_data"))

types <- c("std", "sb", "ss", "sf", "all", "pall",
           "eba2", "peba", "peba2", "peba4", "peba6", "pols")
grid <- expand.grid(type = types, ug = c(FALSE, TRUE), base = c("ml", "rls"),
                    stringsAsFactors = FALSE)
tests <- apply(grid, 1L, function(r) {
  paste(c(r[["type"]], if (identical(r[["ug"]], "TRUE")) "ug", r[["base"]]),
        collapse = "_")
})
expected_type <- sub("^peba$", "peba2", sub("^pols$", "pols2", types))
expected <- apply(expand.grid(type = expected_type, ug = c(FALSE, TRUE),
                              base = c("ml", "rls"),
                              stringsAsFactors = FALSE),
                  1L, function(r) {
                    paste(c(r[["type"]], if (identical(r[["ug"]], "TRUE")) "ug",
                            r[["base"]]), collapse = "_")
                  })

tab <- fmg_tests(fit, tests = tests)
stopifnot(inherits(tab, "magmaan_fmg_tests"))
stopifnot(identical(tab$label, expected))
stopifnot(all(c("df", "base_statistic", "method", "param", "ug",
                "eigenvalues", "lambdas", "lambdas_reference") %in% names(tab)))
stopifnot(all(tab$df == tab$df[[1L]]), tab$df[[1L]] > 0L)
stopifnot(all(tab$base %in% c("ml", "rls")))
stopifnot(all(is.finite(tab$p_value)), all(tab$p_value >= 0), all(tab$p_value <= 1))
stopifnot(all(lengths(tab$eigenvalues) >= tab$df))
stopifnot(all(lengths(tab$lambdas) == tab$df))

pv <- fmg_pvalues(fit, tests = tests)
stopifnot(identical(names(pv), tab$label))
stopifnot(isTRUE(all.equal(unname(pv), tab$p_value, tolerance = 1e-12)))

pv_explicit_data <- fmg_pvalues(fit, data = df, tests = tests[seq_len(6L)])
stopifnot(isTRUE(all.equal(unname(pv_explicit_data), tab$p_value[seq_len(6L)],
                           tolerance = 1e-12)))

fm <- fit_measures(fit, fmg = c("sb_rls", "peba4_rls"))
stopifnot(is.finite(fm$cfi), is.finite(fm$rmsea))
stopifnot(inherits(fm$fmg, "magmaan_fmg_tests"))
stopifnot(identical(fm$fmg$label, c("sb_rls", "peba4_rls")))

fit_group <- magmaan(model, df, estimator = "ML", groups = "school",
                     meanstructure = TRUE, se = "none", test = "none")
group_tests <- c("sb_ug_rls", "peba2_ug_rls", "peba4_rls", "pols2_rls")
tab_group <- fmg_tests(fit_group, tests = group_tests)
stopifnot(inherits(tab_group, "magmaan_fmg_tests"))
stopifnot(identical(tab_group$label, group_tests))
stopifnot(all(is.finite(tab_group$p_value)),
          all(tab_group$p_value >= 0), all(tab_group$p_value <= 1))
stopifnot(all(lengths(tab_group$eigenvalues) >= tab_group$df))
stopifnot(all(lengths(tab_group$lambdas) == tab_group$df))

pv_group_df <- fmg_pvalues(fit_group, data = df, tests = group_tests)
pv_group_list <- fmg_pvalues(fit_group, data = fit_group$raw_data$X,
                             tests = group_tests)
stopifnot(isTRUE(all.equal(unname(pv_group_df), tab_group$p_value,
                           tolerance = 1e-12)))
stopifnot(isTRUE(all.equal(unname(pv_group_list), tab_group$p_value,
                           tolerance = 1e-12)))

df_missing <- df
df_missing$x1[1L] <- NA_real_
err_missing <- tryCatch(fmg_tests(fit, data = df_missing), error = conditionMessage)
stopifnot(grepl("missing observed values", err_missing, fixed = TRUE))

fiml_data <- df_to_fiml_data(df_missing, model_spec(model))
err_fiml <- tryCatch(fmg_tests(fit, data = fiml_data), error = conditionMessage)
stopifnot(grepl("FIML/missing-data", err_fiml, fixed = TRUE))

# Oracle parity: magmaan's FMG p-values must match semTests::pvalues() value-for-
# value over the full test x gamma x base grid. semTests reads the UGamma spectra
# and base statistics off a lavaan robust fit; magmaan computes them itself, so an
# agreement to ~1e-8 exercises the entire chain (base stats, biased AND unbiased
# Du-Bentler gamma, every eigenvalue-tail transform). Guarded so the example still
# runs where semTests is absent.
if (requireNamespace("semTests", quietly = TRUE) &&
    requireNamespace("lavaan", quietly = TRUE)) {
  fit_l <- lavaan::sem(model, df, estimator = "ML", test = "satorra.bentler",
                       meanstructure = FALSE)
  # std ignores the gamma flavour and semTests names it std_<base> only, so omit
  # std_ug; every other method depends on the spectrum and is tested both ways.
  ug_types <- c("sb", "ss", "sf", "all", "pall", "eba2", "eba3", "eba4",
                "peba2", "peba4", "peba6", "pols2")
  g <- expand.grid(type = ug_types, ug = c(FALSE, TRUE), base = c("ml", "rls"),
                   stringsAsFactors = FALSE)
  parity_tests <- c("std_ml", "std_rls",
                    apply(g, 1L, function(r)
                      paste(c(r[["type"]], if (identical(r[["ug"]], "TRUE")) "ug",
                              r[["base"]]), collapse = "_")))
  pv_m <- fmg_pvalues(fit, tests = parity_tests)
  pv_s <- semTests::pvalues(fit_l, tests = as.list(parity_tests))
  common <- intersect(names(pv_m), names(pv_s))
  stopifnot(setequal(names(pv_m), names(pv_s)))
  stopifnot(max(abs(pv_m[common] - pv_s[common])) < 1e-6)
  cat(sprintf("FMG vs semTests parity: ok (%d cells, max|d| = %.1e)\n",
              length(common), max(abs(pv_m[common] - pv_s[common]))))
}

cat("FMG fit-measure/inference workflow: ok\n")
