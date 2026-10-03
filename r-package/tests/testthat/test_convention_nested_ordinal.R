# Whole-bundle gates against installed lavaan; doctest-relative tolerance.
ordinal_nested_data <- function() {
  d <- lavaan::HolzingerSwineford1939
  for (v in paste0("x", 1:6)) d[[v]] <- ordered(cut(d[[v]],
    quantile(d[[v]], c(0, 1/3, 2/3, 1)), include.lowest = TRUE,
    labels = FALSE), levels = 1:3)
  d
}
ordinal_nested_pair <- function(syntax, parameterization, convention,
                                equal = list(NULL, NULL), group = FALSE) {
  d <- ordinal_nested_data()
  ord <- paste0("x", if (grepl("x4", syntax[[1]])) 1:6 else 1:3)
  estimator <- switch(convention, WLSMV = "DWLS", ULSMV = "ULS", convention)
  fits <- refs <- vector("list", 2)
  for (i in 1:2) {
    spec <- model_spec(syntax[[i]], ordered = ord, meanstructure = TRUE,
      parameterization = parameterization, group = if (group) "school" else NULL,
      group_labels = if (group) levels(d$school) else NULL, group_equal = equal[[i]])
    fits[[i]] <- fit_model(spec, d, estimator = estimator)
    refs[[i]] <- lavaan::cfa(syntax[[i]], d, ordered = ord, estimator = convention,
      parameterization = parameterization, group = if (group) "school" else NULL,
      group.label = if (group) levels(d$school) else NULL, group.equal = equal[[i]])
    expect_true(fits[[i]]$converged)
    expect_true(lavaan::lavInspect(refs[[i]], "converged"))
  }
  list(fits = fits, refs = refs)
}
expect_ordinal_nested_bundle <- function(pair, convention) {
  ref <- lavaan::lavTestLRT(pair$refs[[1]], pair$refs[[2]])
  ours <- convention_nested(pair$fits[[1]], pair$fits[[2]], convention)$test
  expect_true(ours$available, info = ours$detail)
  relative <- function(x, y) expect_lte(abs(x-y), 1e-5 * max(abs(x), abs(y)))
  relative(ours$statistic, ref[2, "Chisq diff"])
  expect_equal(ours$df, ref[2, "Df diff"])
  if (convention %in% c("DWLS", "ULS")) expect_true(is.na(ours$pvalue))
  else relative(ours$pvalue, ref[2, "Pr(>Chisq)"])
  if (convention %in% c("WLSMV", "ULSMV")) {
    explicit <- lavaan::lavTestLRT(pair$refs[[1]], pair$refs[[2]],
      method = "satorra.2000", A.method = "delta", scaled.shifted = TRUE)
    expect_equal(ref, explicit)
    relative(ours$scale, 1/attr(ref, "scale")[[2]])
    # A one-dimensional shift is zero up to floating-point roundoff.
    expect_lte(abs(ours$shift - attr(ref, "shift")[[2]]),
      1e-5 * max(abs(ours$shift), abs(attr(ref, "shift")[[2]])) + 1e-14)
    expect_identical(ours$method, "satorra.2000")
  } else {
    expect_identical(ours$method, "standard")
    expect_equal(ours$scale, 1)
    expect_equal(ours$shift, 0)
  }
  reversed <- convention_nested(pair$fits[[2]], pair$fits[[1]], convention)$test
  expect_identical(reversed$reason, "not_nested")
}

for (convention in c("WLSMV", "ULSMV", "DWLS", "ULS", "WLS")) {
  test_that(paste(convention, "single-group ordinal nested bundles match lavaan"), {
    skip_if_not_installed("lavaan")
    base <- "visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6"
    for (p in c("delta", "theta")) {
      for (restriction in c("visual ~~ 0*textual", "visual =~ 0.8*x2")) {
        pair <- ordinal_nested_pair(list(base, paste(base, restriction, sep = "\n")), p, convention)
        expect_ordinal_nested_bundle(pair, convention)
      }
    }
  })
}
for (convention in c("WLSMV", "ULSMV")) {
  test_that(paste(convention, "theta invariance uses the default delta nested recipe"), {
    skip_if_not_installed("lavaan")
    base <- "visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6"
    for (eq in list(list(NULL, "loadings"),
                    list("thresholds", c("thresholds", "loadings")))) {
      pair <- ordinal_nested_pair(list(base, base), "theta", convention, eq, TRUE)
      expect_ordinal_nested_bundle(pair, convention)
    }
  })
  test_that(paste(convention, "retains a saturated ordinal alternative"), {
    skip_if_not_installed("lavaan")
    pair <- ordinal_nested_pair(list("f =~ x1 + x2 + x3",
      "f =~ x1 + a*x2 + a*x3"), "theta", convention)
    expect_equal(lavaan::fitMeasures(pair$refs[[1]], "df"), c(df = 0))
    expect_ordinal_nested_bundle(pair, convention)
  })
}

test_that("ordinal nested compatibility retains explicit refusal reasons", {
  skip_if_not_installed("lavaan")
  base <- "visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6"
  pair <- ordinal_nested_pair(list(base, paste(base, "visual ~~ 0*textual", sep = "\n")),
                             "theta", "WLSMV")
  a <- pair$fits[[1]]; b <- pair$fits[[2]]
  bad <- b; bad$converged <- FALSE
  expect_identical(convention_nested(a, bad, "WLSMV")$test$reason, "not_converged")
  bad <- fit_model(model_spec(base, ordered = paste0("x", 1:6),
    parameterization = "delta", meanstructure = TRUE), ordinal_nested_data(), estimator = "DWLS")
  expect_identical(convention_nested(a, bad, "WLSMV")$test$reason, "unsupported_model")
  # Match the retained positive-penalty metadata used by .policy_state().
  bad <- b; bad$penalty_inference <- "not_validated"
  bad$composition$penalty_weight <- 0.1
  expect_identical(convention_nested(a, bad, "WLSMV")$test$reason, "penalized")
  nonnested <- ordinal_nested_pair(list(paste(base, "visual ~~ 0*textual", sep = "\n"),
    paste(base, "visual =~ 0.8*x2", sep = "\n")), "theta", "WLSMV")
  expect_identical(convention_nested(nonnested$fits[[1]], nonnested$fits[[2]], "WLSMV")$test$reason,
                   "not_nested")
})
