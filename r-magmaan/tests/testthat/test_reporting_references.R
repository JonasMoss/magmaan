reference_names <- c("std", "sb", "ss", "mv", "scaled_f", "all", "pall",
                     "peba1", "peba2", "peba4", "peba6", "eba2", "eba4")
reference_columns <- c("test", "statistic", "df", "reference", "pvalue", "recommended", "reason")

reference_tail <- function(t, method) {
  block <- startsWith(method, "peba") || startsWith(method, "eba")
  magmaanlab::magmaan_core$robust_fmg_test(t$statistic, t$df, t$eigenvalues,
    if (method == "std") "standard" else if (block) sub("[0-9]+$", "", method) else method,
    if (block) as.numeric(sub("^p?eba", "", method)) else 0)$p_value
}

expect_reference_rows <- function(rows, t, label, defaults) {
  z <- rows[rows$test == label, , drop = FALSE]
  if (!isTRUE(t$available)) {
    expect_equal(nrow(z), 1L)
    expect_true(is.na(z$reference) && is.na(z$pvalue))
    expect_identical(z$reason, t$reason)
    expect_false(z$recommended)
    return(invisible(NULL))
  }
  expect_identical(z$reference, reference_names)
  expect_equal(z$statistic, rep(t$statistic, length(reference_names)), tolerance = 0)
  expect_equal(z$df, rep(t$df, length(reference_names)))
  expect_equal(z$pvalue, unname(vapply(reference_names, function(m) reference_tail(t, m), 0)),
               tolerance = 1e-12)
  expect_identical(z$recommended, reference_names %in% defaults)
  expect_true(all(is.na(z$reason)))
}

test_that("global and nested references reuse policy spectra across regimes and groups", {
  skip_if_not_installed("lavaan")
  tables <- list()
  for (regime in c("ML", "FIML", "ordinal", "mixed")) {
    for (grouped in c(FALSE, TRUE)) {
      d <- hs()
      ord <- if (regime == "ordinal") paste0("x", 1:6) else
        if (regime == "mixed") paste0("x", 1:3) else NULL
      if (length(ord)) for (v in ord) d[[v]] <- ordered(cut(d[[v]],
        quantile(d[[v]], c(0, 1/3, 2/3, 1)), include.lowest = TRUE))
      if (regime == "FIML") {
        set.seed(9595)
        # MAR: missingness depends on an always-observed indicator.
        for (v in paste0("x", 2:6)) d[[v]][runif(nrow(d)) < plogis(-4 + .2*d$x1)] <- NA_real_
      }
      syntax <- cfa
      null_syntax <- if (grouped) syntax else
        "visual =~ x1 + a*x2 + a*x3\ntextual =~ x4 + x5 + x6"
      spec <- function(s, null = FALSE) magmaan_model(s, prototype = d, ordered = ord,
        group = if (grouped) "school" else NULL,
        group.equal = if (grouped && null) "loadings" else NULL)
      estimator <- if (regime %in% c("ordinal", "mixed")) "DWLS" else regime
      h1 <- magmaan(spec(syntax), d, estimator = estimator)
      h0 <- magmaan(spec(null_syntax, TRUE), d, estimator = estimator)
      expect_true(h1$lab$converged && h0$lab$converged)
      global <- summary(h0, references = toupper(reference_names))$tests
      expect_identical(names(global), reference_columns)
      expect_identical(unique(global$test), if (estimator == "DWLS") c("fit_function", "lr") else c("score", "lr"))
      native_global <- magmaanlab::policy_inference(h0$lab)
      defaults <- if (estimator == "DWLS") "all" else c("sb", "peba4")
      expect_reference_rows(global, native_global$score,
        if (estimator == "DWLS") "fit_function" else "score", defaults)
      expect_reference_rows(global, native_global$lr, "lr", character())
      default_global <- summary(h0)$tests
      for (component in c("score", "lr")) {
        t <- native_global[[component]]
        if (!isTRUE(t$available)) next
        label <- if (component == "lr") "lr" else
          if (estimator == "DWLS") "fit_function" else "score"
        z <- default_global[default_global$test == label, ]
        expected <- if (identical(t$reference, "all")) t$p_all else c(t$p_sb, t$p_peba4)
        expect_identical(z$pvalue, expected)
      }
      nested <- anova(h1, h0, references = reference_names)
      expect_identical(names(nested), reference_columns)
      expect_identical(unique(nested$test), if (estimator == "DWLS") c("score", "fit_function_difference") else c("score", "lr"))
      native_nested <- magmaanlab::policy_nested(h1$lab, h0$lab)
      expect_true(native_nested$lr$available, info = native_nested$lr$detail)
      expect_reference_rows(nested, native_nested$score, "score", c("sb", "peba4"))
      expect_reference_rows(nested, native_nested$lr,
        if (estimator == "DWLS") "fit_function_difference" else "lr",
        if (estimator == "DWLS") c("sb", "peba4") else character())
      expect_identical(attr(nested, "spectra"), lapply(native_nested[c("score", "lr")], function(t) t$eigenvalues))
      default_nested <- anova(h1, h0)
      for (component in c("score", "lr")) {
        t <- native_nested[[component]]
        if (!isTRUE(t$available)) next
        label <- if (component == "score") "score" else
          if (estimator == "DWLS") "fit_function_difference" else "lr"
        expect_identical(default_nested$pvalue[default_nested$test == label], c(t$p_sb, t$p_peba4))
      }
      for (pair in list(list(global, default_global, native_global,
                            if (estimator == "DWLS") "fit_function" else "score"),
                       list(nested, default_nested, native_nested,
                            if (estimator == "DWLS") "fit_function_difference" else "score"))) {
        component <- if (pair[[4]] == "fit_function_difference") "lr" else "score"
        native <- pair[[3]][[component]]
        laws <- if (identical(native$reference, "all")) "all" else c("sb", "peba4")
        for (rows in pair[1:2]) {
          selected <- subset(rows, recommended)
          expect_identical(selected$test, rep(pair[[4]], length(laws)))
          expect_identical(selected$reference, laws)
          expect_identical(selected$pvalue, if (identical(laws, "all")) native$p_all else
            c(native$p_sb, native$p_peba4))
          expect_false(any(rows$recommended[rows$test == "lr"]))
        }
      }
      if (estimator != "DWLS") {
        expect_output(print(summary(h0)), "likelihood ratio")
        expect_output(print(default_nested), "likelihood ratio")
      }
      bundle <- if (estimator == "DWLS") "WLSMV" else "MLR"
      compat_summary <- summary(h0, lavaan_compat = bundle)
      compat <- compat_summary$tests
      expect_identical(compat$reference, compat_summary$inference$global_lr$method)
      expect_identical(compat$test, if (estimator == "DWLS") "fit_function" else "lr")
      compat_nested <- anova(h1, h0, lavaan_compat = if (estimator == "DWLS") "WLSMV" else "MLR")
      expect_identical(compat_nested$test, if (estimator == "DWLS") "fit_function_difference" else "lr")
      expect_identical(compat_nested$reference, magmaanlab::convention_nested(h1$lab, h0$lab, bundle)$test$method)
      for (z in list(compat, compat_nested)) {
        expect_identical(names(z), c(reference_columns, "unscaled.statistic", "scale", "shift"))
        expect_false(any(z$recommended))
        tables[[length(tables) + 1L]] <- z[reference_columns]
      }
      tables[[length(tables) + 1L]] <- global
      tables[[length(tables) + 1L]] <- as.data.frame(nested)
    }
  }
  expect_equal(nrow(do.call(rbind, tables)), sum(vapply(tables, nrow, 0L)))
})

test_that("references are validated even for unavailable tests and printing shows requested laws", {
  fit <- magmaan(cfa, hs())
  null <- magmaan("visual =~ x1 + a*x2 + a*x3\ntextual =~ x4 + x5 + x6", hs())
  for (bad in list(character(), NA_character_, 4, "peba0", "eba-1", "peba2.5", "peba", "pols", "all ", "peba2147483648")) {
    expect_error(summary(fit, references = bad), "std, sb, ss, mv, scaled_f, all, pall, peba<k>, eba<k>", fixed = TRUE)
    expect_error(anova(fit, null, references = bad), "integer k >= 1", fixed = TRUE)
  }
  expect_error(summary(fit, references = "all", lavaan_compat = "MLR"), "cannot be combined")
  expect_error(anova(fit, null, references = "all", lavaan_compat = "MLR"), "cannot be combined")
  expect_output(print(summary(fit, references = c("STD", "ALL"))), "std")
  expect_output(print(anova(fit, null, references = c("STD", "ALL"))), "all")
  saturated <- magmaan("x1 ~~ x1", hs())
  z <- summary(saturated, references = reference_names)$tests
  expect_equal(nrow(z), 2L)
  expect_identical(z$reason, rep("saturated", 2L))
  expect_false(any(z$recommended))
  compat <- summary(saturated, lavaan_compat = "MLR")$tests
  expect_equal(nrow(compat), 1L)
  expect_identical(compat$reason, "saturated")
})
