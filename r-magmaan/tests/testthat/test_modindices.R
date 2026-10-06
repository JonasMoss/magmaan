.mi_workers <- function(fit, data) {
  lab <- as_lab_fit(fit)
  ordinal <- isTRUE(lab$ordinal) || isTRUE(lab$mixed_ordinal)
  fixed <- magmaanlab::modification_indices_robust(lab, data = data,
    bread = "observed", information = if (ordinal) "expected" else "observed",
    estimated_weight = ordinal)
  release <- magmaanlab::score_tests_robust(lab, data = data,
    bread = "observed", estimated_weight = ordinal)
  list(fixed = fixed, release = release)
}
.mi_key <- function(x) paste(x$lhs, x$op, x$rhs, x$group)

test_that("ML candidates agree with lavaan and score rows agree with policy workers", {
  d <- hs()
  syntax <- paste(cfa, "speed =~ x7 + x8 + x9", sep = "\n")
  for (grouped in c(FALSE, TRUE)) {
    m <- if (grouped) magmaan_model(syntax, prototype = d, group = "school") else magmaan_model(syntax)
    fit <- magmaan(m, d, inference = FALSE)
    out <- modindices(fit, releases = FALSE)
    policy <- magmaanlab::policy_modification_indices(fit$lab, releases = FALSE)
    expect_equal(out, policy, ignore_attr = TRUE)
    w <- .mi_workers(fit, d)$fixed
    available <- out[out$reason == "available", ]
    expect_equal(available$statistic, w$mi.scaled)
    expect_equal(available$epc, w$epc)
    expect_equal(available$pvalue, pchisq(available$statistic, 1, lower.tail = FALSE))
    lav <- lavaan::cfa(syntax, d, meanstructure = TRUE, group = if (grouped) "school" else NULL)
    cand <- lavaan::modindices(lav)
    if (is.null(cand$group)) cand$group <- 1L
    expect_setequal(.mi_key(out), .mi_key(cand))
    selected <- modindices(fit, candidates = out[c(4,1), ])
    expect_equal(.mi_key(selected), .mi_key(out[c(1,4), ]))
  }
})

test_that("ML equality releases match robust workers for labels and explicit equalities", {
  d <- hs()
  for (syntax in c("f =~ x1 + a*x2 + a*x3 + x4", "f =~ x1 + a*x2 + b*x3 + x4\na == b")) {
    fit <- magmaan(magmaan_model(syntax), d, inference = FALSE)
    out <- modindices(fit)
    release <- out[out$kind == "release", ]
    w <- .mi_workers(fit, d)$release
    expect_equal(release$statistic, w$mi.scaled)
    expect_equal(release$pvalue, w$pvalue)
    expect_true(nrow(release) > 0)
    expect_identical(nrow(modindices(fit, releases = FALSE)), nrow(out) - nrow(release))
  }
  fit <- magmaan(magmaan_model(cfa, prototype = d, group = "school", group.equal = "loadings"), d, inference = FALSE)
  out <- modindices(fit)
  expect_equal(out$statistic[out$kind == "release"], .mi_workers(fit, d)$release$mi.scaled)
})

test_that("ML LR rows agree with explicitly augmented anova models", {
  d <- hs()
  fit <- magmaan(magmaan_model(cfa), d, inference = FALSE)
  row <- data.frame(lhs = "x2", op = "~~", rhs = "x3", group = 1L)
  lr <- modindices(fit, test = "lr", candidates = row, releases = FALSE)
  augmented <- magmaan(magmaan_model(paste(cfa, "x2 ~~ x3", sep = "\n")), d, inference = FALSE)
  compare <- anova(fit, augmented)
  compare <- compare[compare$test == "lr", ]
  expect_identical(lr$reason, "available")
  expect_equal(lr$statistic, compare$statistic, tolerance = 1e-6)
  expect_equal(lr$pvalue, compare$pvalue, tolerance = 1e-6)
  tied <- magmaan(magmaan_model("f =~ x1 + a*x2 + a*x3 + x4"), d, inference = FALSE)
  release <- modindices(tied, candidates = "==", test = "lr")
  free <- magmaan(magmaan_model("f =~ x1 + x2 + x3 + x4"), d, inference = FALSE)
  compare <- anova(tied, free)
  expect_equal(release$statistic, compare$statistic[compare$test == "lr"], tolerance = 1e-6)
  expect_equal(release$pvalue, compare$pvalue[compare$test == "lr"], tolerance = 1e-6)
})

test_that("FIML policy MI and releases use MAR pattern scores in one and two groups", {
  d <- hs()
  d$x2[d$x1 > median(d$x1)] <- NA_real_
  for (grouped in c(FALSE, TRUE)) {
    syntax <- "f =~ x1 + a*x2 + a*x3 + x4"
    m <- if (grouped) magmaan_model(syntax, prototype = d, group = "school") else magmaan_model(syntax)
    fit <- magmaan(m, d, estimator = "FIML", inference = FALSE)
    out <- modindices(fit)
    w <- .mi_workers(fit, d)
    expect_equal(out$statistic[out$kind == "fixed" & out$reason == "available"], w$fixed$mi.scaled)
    expect_equal(out$statistic[out$kind == "release"], w$release$mi.scaled)
    candidate <- out[which(out$lhs == "x1" & out$op == "~~" & out$rhs == "x2" & out$group == 1)[1], ]
    lr <- modindices(fit, test = "lr", candidates = candidate, releases = FALSE)
    expect_identical(lr$reason, "available")
    expect_true(is.finite(lr$pvalue))
    line <- if (grouped) paste0(candidate$lhs, " ", candidate$op, " c(NA,0)*", candidate$rhs) else
      paste(candidate$lhs, candidate$op, candidate$rhs)
    alt_model <- if (grouped) magmaan_model(paste(syntax, line, sep = "\n"), prototype = d, group = "school") else
      magmaan_model(paste(syntax, line, sep = "\n"))
    alt <- magmaan(alt_model, d, estimator = "FIML", inference = FALSE)
    alt$lab <- suppressWarnings(magmaanlab::refit_from_null(alt$lab, fit$lab))
    compare <- anova(fit, alt)
    compare <- compare[compare$test == "lr", ]
    expect_equal(lr$statistic, compare$statistic, tolerance = 1e-6)
    expect_equal(lr$pvalue, compare$pvalue, tolerance = 1e-6)
  }
})

test_that("ordinal and mixed DWLS policy MI use exact estimated-weight rows", {
  d <- hs()
  for (mixed in c(FALSE, TRUE)) for (grouped in c(FALSE, TRUE)) {
    ordered <- paste0("x", if (mixed) 1:3 else 1:6)
    dat <- d
    for (v in ordered) dat[[v]] <- ordered(cut(dat[[v]], 3))
    m <- magmaan_model(cfa, prototype = dat, ordered = ordered,
      group = if (grouped) "school" else NULL,
      group.equal = if (grouped) "loadings" else NULL)
    fit <- magmaan(m, dat, estimator = "DWLS", inference = FALSE)
    out <- modindices(fit)
    expect_true(any(out$reason == "available"))
    expect_equal(out, magmaanlab::policy_modification_indices(fit$lab), ignore_attr = TRUE)
    # Mixed workers already construct exact rows; all-ordinal lab comparators
    # deliberately retain their legacy OPG first-stage default.
    if (mixed) {
      w <- .mi_workers(fit, dat)
      expect_equal(out$statistic[out$kind == "fixed"], w$fixed$mi.scaled)
      expect_equal(out$statistic[out$kind == "release"], w$release$mi.scaled)
    }
    expect_equal(out$pvalue, pchisq(out$statistic, 1, lower.tail = FALSE))
    row <- out[which(out$lhs == "x2" & out$rhs == "x3" & out$op == "~~" & out$group == 1)[1], ]
    lr <- modindices(fit, test = "lr", candidates = row, releases = FALSE)
    expect_identical(lr$test, "fit_function_difference")
    expect_identical(lr$reason, "available")
    expect_true(is.finite(lr$pvalue))
    line <- if (grouped) "x2 ~~ c(NA,0)*x3" else "x2 ~~ x3"
    alt_model <- magmaan_model(paste(cfa, line, sep = "\n"), prototype = dat, ordered = ordered,
      group = if (grouped) "school" else NULL, group.equal = if (grouped) "loadings" else NULL)
    alt <- magmaan(alt_model, dat, estimator = "DWLS", inference = FALSE)
    compare <- anova(fit, alt)
    compare <- compare[compare$test == "fit_function_difference", ]
    expect_equal(lr$statistic, compare$statistic, tolerance = 1e-6)
    expect_equal(lr$pvalue, compare$pvalue, tolerance = 1e-6)
    expect_identical(compare$reference, "all")
  }
})

test_that("unsupported fits and failed refits keep typed rows", {
  d <- hs()
  f <- magmaan(magmaan_model(cfa), d, inference = FALSE)
  bad <- f; bad$lab$converged <- FALSE
  expect_identical(modindices(bad)$reason, "not_converged")
  bad <- f; bad$lab$penalty_inference <- "not_validated"; bad$lab$composition$penalty_weight <- 1
  expect_identical(modindices(bad)$reason, "penalized")
  bad <- f; bad$lab$estimator <- "ULS"
  expect_identical(modindices(bad)$reason, "unsupported_model")
  bad <- f; bad$lab$estimator <- "association_ml"
  expect_identical(modindices(bad)$reason, "unsupported_model")
  expect_identical(modindices(f, candidates = "~")$reason, "unsupported_model")
  expect_error(modindices(f, candidates = "invalid"), "operator")
  expect_true(nrow(magmaanlab::modification_indices_robust(f$lab, data = d)) > 0)
})

test_that("PoliticalDemocracy SEM candidate rows agree with lavaan", {
  skip_if_not_installed("lavaan")
  syntax <- paste("ind60 =~ x1 + x2 + x3", "dem60 =~ y1 + y2 + y3 + y4",
    "dem65 =~ y5 + y6 + y7 + y8", "dem60 ~ ind60", "dem65 ~ ind60 + dem60",
    "y1 ~~ y5", "y2 ~~ y4 + y6", "y3 ~~ y7", "y4 ~~ y8", "y6 ~~ y8", sep = "\n")
  d <- lavaan::PoliticalDemocracy
  fit <- magmaan(magmaan_model(syntax), d, inference = FALSE)
  out <- modindices(fit, releases = FALSE)
  lav <- lavaan::modindices(lavaan::sem(syntax, d, meanstructure = TRUE))
  lav$group <- 1L
  expect_setequal(.mi_key(out), .mi_key(lav))
})

test_that("failed LR refits preserve the candidate and typed reason", {
  d <- hs()
  f <- magmaan(magmaan_model(cfa), d, inference = FALSE)
  row <- data.frame(lhs = "x2", op = "~~", rhs = "x3", group = 1L)
  testthat::local_mocked_bindings(.policy_mi_refit = function(...) stop("solver failed"),
    .package = "magmaanlab")
  out <- modindices(f, test = "lr", candidates = row, releases = FALSE)
  expect_identical(out$reason, "refit_failed")
  expect_equal(out$lhs, row$lhs)
  expect_true(is.na(out$statistic))
  expect_true(is.na(out$pvalue))
})

test_that("two-group ML LR refits free only the requested group's parameter", {
  d <- hs()
  model <- magmaan_model(cfa, prototype = d, group = "school")
  fit <- magmaan(model, d, inference = FALSE)
  row <- data.frame(lhs = "x2", op = "~~", rhs = "x3", group = 1L)
  out <- modindices(fit, test = "lr", candidates = row, releases = FALSE)
  alt <- magmaan(magmaan_model(paste(cfa, "x2 ~~ c(NA,0)*x3", sep = "\n"),
    prototype = d, group = "school"), d, inference = FALSE)
  compare <- anova(fit, alt)
  compare <- compare[compare$test == "lr", ]
  expect_identical(out$reason, "available")
  expect_equal(out$statistic, compare$statistic, tolerance = 1e-6)
  expect_equal(out$pvalue, compare$pvalue, tolerance = 1e-6)
})
