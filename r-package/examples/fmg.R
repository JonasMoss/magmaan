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
                     se = "none", test = "none")
err_group <- tryCatch(fmg_tests(fit_group), error = conditionMessage)
stopifnot(grepl("single-group", err_group, fixed = TRUE))

df_missing <- df
df_missing$x1[1L] <- NA_real_
err_missing <- tryCatch(fmg_tests(fit, data = df_missing), error = conditionMessage)
stopifnot(grepl("missing observed values", err_missing, fixed = TRUE))

fiml_data <- df_to_fiml_data(df_missing, model_spec(model))
err_fiml <- tryCatch(fmg_tests(fit, data = fiml_data), error = conditionMessage)
stopifnot(grepl("FIML/missing-data", err_fiml, fixed = TRUE))

cat("FMG fit-measure/inference workflow: ok\n")
