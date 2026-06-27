# Two-level (multilevel) normal-theory ML parity, exercising the exported
# fit_twolevel() wrapper and the magmaan(cluster = ) one-call entry point against
# lavaan sem(..., cluster = ). Promoted from examples/twolevel_parity.R.

# Match free parameters by (lhs, op, rhs, block). For a two-level partable the
# block already encodes the level (and, for multi-group magmaan partables, the
# group), so this is a unique key. lavaan's single-group two-level parTable has
# no `group` column, hence keying on block rather than group.
.tl_key <- function(t) paste(t$lhs, t$op, t$rhs, t$block)

.tl_demo <- function() {
  skip_if_not_installed("lavaan")
  e <- new.env()
  utils::data("Demo.twolevel", package = "lavaan", envir = e)
  e$Demo.twolevel
}

test_that("single-group two-level ML matches lavaan (saturated CFA)", {
  skip_if_not_installed("lavaan")
  d <- .tl_demo()
  model <- "
    level: 1
      fw =~ y1 + y2 + y3
    level: 2
      fb =~ y1 + y2 + y3
  "
  fit_lav <- lavaan::sem(model, data = d, cluster = "cluster")
  lav_chisq <- unname(lavaan::fitMeasures(fit_lav, "chisq"))
  lav_df    <- as.integer(unname(lavaan::fitMeasures(fit_lav, "df")))
  pt_l      <- lavaan::parTable(fit_lav)

  fit <- fit_twolevel(model, d, cluster = "cluster")

  expect_s3_class(fit, "magmaan_fit")
  expect_true(fit$converged)
  expect_true(is.list(fit$audit))
  expect_true(is.logical(fit$audit$stationary))
  expect_true(is.list(fit$diagnostics))
  expect_true(isTRUE(fit$diagnostics$sigma_pd_all))
  expect_equal(fit$ngroups, 1L)
  expect_equal(fit$df, lav_df)
  expect_equal(fit$chisq, lav_chisq, tolerance = 1e-3)

  mfree <- fit$partable[fit$partable$free > 0, ]
  mfree$k  <- .tl_key(mfree)
  mfree$se <- fit$se[mfree$free]
  lfree <- pt_l[pt_l$free > 0, ]
  lfree$k <- .tl_key(lfree)
  cmp <- merge(mfree[, c("k", "est", "se")], lfree[, c("k", "est", "se")],
               by = "k", suffixes = c(".m", ".l"))

  expect_equal(nrow(cmp), sum(fit$partable$free > 0))
  expect_lt(max(abs(cmp$est.m - cmp$est.l)), 1e-3)
  expect_lt(max(abs(cmp$se.m  - cmp$se.l)),  5e-3)
})

test_that("magmaan(cluster = ) one-call entry reproduces fit_twolevel()", {
  skip_if_not_installed("lavaan")
  d <- .tl_demo()
  model <- "
    level: 1
      fw =~ y1 + y2 + y3
    level: 2
      fb =~ y1 + y2 + y3
  "
  fit_one <- magmaan(model, d, estimator = "ML", cluster = "cluster")
  fit_two <- fit_twolevel(model, d, cluster = "cluster")

  expect_s3_class(fit_one, "magmaan_fit")
  expect_equal(fit_one$theta, fit_two$theta, tolerance = 1e-8)
  expect_equal(fit_one$chisq, fit_two$chisq, tolerance = 1e-8)
  expect_equal(fit_one$df, fit_two$df)

  # cluster = requires ML; robust corrections stay post-fit.
  expect_error(magmaan(model, d, estimator = "MLR", cluster = "cluster"))
})

test_that("single-group two-level ML matches lavaan on a non-saturated model", {
  skip_if_not_installed("lavaan")
  d <- .tl_demo()
  model <- "
    level: 1
      fw =~ y1 + y2 + y3 + y4 + y5
    level: 2
      fb =~ y1 + y2 + y3 + y4 + y5
  "
  fit_lav <- lavaan::sem(model, data = d, cluster = "cluster")
  lav_chisq <- unname(lavaan::fitMeasures(fit_lav, "chisq"))
  lav_df    <- as.integer(unname(lavaan::fitMeasures(fit_lav, "df")))
  pt_l      <- lavaan::parTable(fit_lav)

  fit <- fit_twolevel(model, d, cluster = "cluster")

  expect_true(fit$converged)
  expect_gt(fit$df, 0L)              # genuinely over-identified
  expect_equal(fit$df, lav_df)
  expect_equal(fit$chisq, lav_chisq, tolerance = 1e-2)

  mfree <- fit$partable[fit$partable$free > 0, ]
  mfree$k  <- .tl_key(mfree)
  mfree$se <- fit$se[mfree$free]
  lfree <- pt_l[pt_l$free > 0, ]
  lfree$k <- .tl_key(lfree)
  cmp <- merge(mfree[, c("k", "est", "se")], lfree[, c("k", "est", "se")],
               by = "k", suffixes = c(".m", ".l"))
  expect_equal(nrow(cmp), sum(fit$partable$free > 0))
  expect_lt(max(abs(cmp$est.m - cmp$est.l)), 1e-2)
  expect_lt(max(abs(cmp$se.m  - cmp$se.l)),  5e-2)
})

test_that("two-level standard bounds stabilize a weak six-indicator between factor", {
  skip_if_not_installed("lavaan")
  d <- .tl_demo()
  model <- "
    level: 1
      fw =~ y1 + y2 + y3 + y4 + y5 + y6
    level: 2
      fb =~ y1 + y2 + y3 + y4 + y5 + y6
  "
  fit_lav <- lavaan::sem(model, data = d, cluster = "cluster")
  lav_chisq <- unname(lavaan::fitMeasures(fit_lav, "chisq"))
  lav_df    <- as.integer(unname(lavaan::fitMeasures(fit_lav, "df")))
  pt_l      <- lavaan::parTable(fit_lav)

  fit <- fit_twolevel(model, d, cluster = "cluster", bounds = "standard")

  expect_true(fit$converged)
  expect_true(isTRUE(fit$diagnostics$sigma_pd_all))
  expect_equal(fit$df, lav_df)
  expect_equal(fit$chisq, lav_chisq, tolerance = 1e-2)
  expect_gt(subset(fit$partable, lhs == "fb" & op == "~~" & rhs == "fb")$est,
            1e-2)

  mfree <- fit$partable[fit$partable$free > 0, ]
  mfree$k  <- .tl_key(mfree)
  mfree$se <- fit$se[mfree$free]
  lfree <- pt_l[pt_l$free > 0, ]
  lfree$k <- .tl_key(lfree)
  cmp <- merge(mfree[, c("k", "est", "se")], lfree[, c("k", "est", "se")],
               by = "k", suffixes = c(".m", ".l"))
  expect_equal(nrow(cmp), sum(fit$partable$free > 0))
  expect_lt(max(abs(cmp$est.m - cmp$est.l)), 1e-3)
  expect_lt(max(abs(cmp$se.m  - cmp$se.l)),  1e-3)

  fit_one <- magmaan(model, d, estimator = "ML", cluster = "cluster",
                     bounds = "standard")
  expect_equal(fit_one$theta, fit$theta, tolerance = 1e-8)
  expect_equal(fit_one$chisq, fit$chisq, tolerance = 1e-8)
})

test_that("multi-group two-level ML separates into independent single-group fits", {
  # Internal-validity check that needs no oracle: an unconstrained multi-group
  # model's joint likelihood separates over groups, so each group's MLE must
  # equal the single-group fit on that group's data. This validates the
  # multi-group two-level path (group ids -> per-group cluster stats -> per-group
  # level blocks) end to end.
  skip_if_not_installed("lavaan")
  d <- .tl_demo()
  model <- "
    level: 1
      fw =~ y1 + y2 + y3
    level: 2
      fb =~ y1 + y2 + y3
  "
  # Group the *clusters* (so clusters never split across groups), then label rows.
  ucl <- sort(unique(d$cluster))
  grp <- stats::setNames(ifelse(seq_along(ucl) %% 2L == 0L, "A", "B"), ucl)
  d$g <- grp[as.character(d$cluster)]
  labels <- unique(as.character(d$g))   # lavaan/magmaan group order = appearance

  fit_mg <- fit_twolevel(model, d, cluster = "cluster", group = "g")
  expect_equal(fit_mg$ngroups, length(labels))
  expect_true(fit_mg$converged)

  # Per-group estimates from the multi-group partable (block 2g-1 = within,
  # 2g = between), matched to a single-group fit on each group's data.
  group_est <- function(pt, g) {
    s <- pt[pt$group == g & pt$free > 0, ]
    s$lb <- s$block - 2L * (g - 1L)     # 1 = within, 2 = between
    stats::setNames(s$est, paste(s$lhs, s$op, s$rhs, s$lb))
  }
  single_est <- function(fit) {
    s <- fit$partable[fit$partable$free > 0, ]
    stats::setNames(s$est, paste(s$lhs, s$op, s$rhs, s$block))
  }
  max_d <- 0
  for (g in seq_along(labels)) {
    fit_g <- fit_twolevel(model, d[d$g == labels[g], ], cluster = "cluster")
    x <- group_est(fit_mg$partable, g)
    y <- single_est(fit_g)
    k <- intersect(names(x), names(y))
    expect_equal(length(k), length(x))
    max_d <- max(max_d, max(abs(x[k] - y[k])))
  }
  expect_lt(max_d, 1e-3)
  expect_equal(fit_mg$ntotal, nrow(d))
})

test_that("multi-group two-level parity vs lavaan (skips: no oracle yet)", {
  skip_if_not_installed("lavaan")
  d <- .tl_demo()
  ucl <- sort(unique(d$cluster))
  grp <- stats::setNames(ifelse(seq_along(ucl) %% 2L == 0L, "A", "B"), ucl)
  d$g <- grp[as.character(d$cluster)]
  model <- "
    level: 1
      fw =~ y1 + y2 + y3
    level: 2
      fb =~ y1 + y2 + y3
  "
  lav <- tryCatch(
    lavaan::sem(model, data = d, cluster = "cluster", group = "g"),
    error = function(e) e)
  if (inherits(lav, "error")) {
    skip(paste0("lavaan ", as.character(utils::packageVersion("lavaan")),
                " cannot fit a multi-group two-level model (",
                conditionMessage(lav), "); no multi-group oracle available"))
  }
  # If a future lavaan supports it, compare estimates.
  fit_mg <- fit_twolevel(model, d, cluster = "cluster", group = "g")
  lav_chisq <- unname(lavaan::fitMeasures(lav, "chisq"))
  expect_equal(fit_mg$chisq, lav_chisq, tolerance = 1e-2)
})
