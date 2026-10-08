test_that("ordinary advanced fitting choices use the shared engine", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  m <- "f =~ x1+x2+x3+x4"
  fit <- magmaan(m, d, inference = FALSE,
      options = list(preset = "lavaan-0.7.2"))
  lab <- as_lab_fit(fit)
  expect_equal(lab$fitting$effective$convergence, "lavaan-0.7.2")
  expect_true(lab$fitting$post_check$ok)
  expect_equal(lab$verdict$policy, "lavaan-0.7.2")
  hybrid <- magmaan(m, d, inference = FALSE,
      options = list(preset = "lavaan-0.7.2", convergence = "newton"))
  expect_true(as_lab_fit(hybrid)$fitting$modified_preset)
  expect_null(as_lab_fit(hybrid)$fitting$post_check)
  expect_equal(as_lab_fit(hybrid)$verdict$status, as_lab_fit(hybrid)$diagnostics$verdict$status)
  # An explicit start overrides the preset's.
  fabin <- magmaan(m, d, inference = FALSE,
      options = list(preset = "lavaan-0.7.2", start = "fabin3"))
  expect_identical(as_lab_fit(fabin)$fitting$effective$starts, "scaled-fabin")
  expect_true(as_lab_fit(fabin)$fitting$modified_preset)
  lavaan_start <- magmaan(m, d, inference = FALSE, options = list(start = "lavaan-0.7.2"))
  expect_identical(as_lab_fit(lavaan_start)$fitting$effective$starts, "lavaan-0.7.2")
  expect_identical(as_lab_fit(lavaan_start)$fitting$effective$convergence, "newton")
  expect_error(magmaan(m, d, inference = FALSE, covariance = "psd",
      options = list(preset = "lavaan-0.7.2")), "continuous ML or FIML")
  expect_error(magmaan(m, d, inference = FALSE, estimator = "ML2S",
      options = list(start = "lavaan-0.7.2")), "continuous ML or FIML")
})

test_that("the lavaan preset with a mean structure follows lavaan's search", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  m <- "f =~ x1+x2+x3+x4"
  fit <- magmaan(m, d, inference = FALSE, options = list(preset = "lavaan-0.7.2"))
  lav <- lavaan::cfa(m, d, meanstructure = TRUE)
  expect_identical(as_lab_fit(fit)$iterations, lavaan::lavInspect(lav, "iterations"))
  theirs <- lavaan::parTable(lav)
  theirs <- theirs[theirs$free > 0L, ]
  p <- coef(summary(fit))
  idx <- match(paste(theirs$lhs, theirs$op, theirs$rhs), paste(p$lhs, p$op, p$rhs))
  expect_equal(p$est[idx], theirs$est, tolerance = 1e-10)
})


test_that("ordinary FIML uses the pinned fitting engine", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  d$x2[seq(1,nrow(d),by=4)] <- NA_real_
  fit <- magmaan("f =~ x1+x2+x3+x4",d,estimator="FIML",inference=FALSE,
    options=list(preset="lavaan-0.7.2"))
  lab <- as_lab_fit(fit)
  lv <- lavaan::sem("f =~ x1+x2+x3+x4",d,missing="ml",fixed.x=FALSE,
    meanstructure=TRUE,se="none",test="none")
  expect_equal(lab$fmin,as.numeric(lv@optim$fx),tolerance=1e-9)
  expect_identical(lab$fitting$effective$optimizer,"lavaan-0.7.2")
  expect_equal(lab$converged,lavaan::lavInspect(lv,"converged"))
})

test_that("ordinary all-ordinal DWLS exposes the lavaan fitting preset", {
  skip_if_not_installed("lavaan")
  skip_if(as.character(utils::packageVersion("lavaan")) != "0.7.2")
  d <- lavaan::HolzingerSwineford1939
  for (name in paste0("x", 1:4)) d[[name]] <- as.integer(cut(d[[name]], 3))
  for (parameterization in c("delta", "theta")) for (grouped in c(FALSE, TRUE)) {
    group <- if (grouped) "school" else NULL
    equal <- if (!grouped) character() else if (parameterization == "theta")
        c("loadings", "thresholds") else "loadings"
    m <- magmaan_model("f =~ x1+x2+x3+x4", d, ordered = paste0("x", 1:4),
        group = group, group.equal = equal, parameterization = parameterization)
    fit <- magmaan(m, d, estimator = "DWLS", inference = FALSE,
        options = list(preset = "lavaan-0.7.2"))
    lv <- lavaan::cfa("f =~ x1+x2+x3+x4", d, ordered = paste0("x", 1:4),
        group = group, group.label = if (grouped) m$groups else NULL,
        group.equal = equal, parameterization = parameterization,
        estimator = "WLSMV", se = "none", test = "none")
    lab <- as_lab_fit(fit)
    pt <- lavaan::parTable(lv)
    key <- function(x) paste(x$lhs, x$op, x$rhs, x$group)
    mp <- lab$partable[lab$partable$free > 0L, ]
    mp <- mp[order(mp$free), ]
    expect_equal(as.numeric(lab$theta), pt$est[match(key(mp), key(pt))], tolerance = 1e-5)
    expect_equal(lab$converged, lavaan::lavInspect(lv, "converged"))
    expect_identical(lab$fitting$effective$optimizer, "lavaan-0.7.2")
    for (estimator in c("ULS", "WLS"))
      expect_error(magmaan(m, d, estimator = estimator, inference = FALSE,
          options = list(preset = "lavaan-0.7.2")), "continuous ML or FIML")
  }
  mixed <- magmaan_model("f =~ x1+x2+x3+x4", d, ordered = c("x1", "x2"))
  expect_true(as_lab_fit(magmaan(mixed, d, estimator = "DWLS", inference = FALSE,
      options = list(preset = "lavaan-0.7.2")))$converged)
})

test_that("preset reports supplied tables and previous fits as first-attempt starts", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  m <- "f =~ x1+x2+x3+x4"
  preset <- list(preset = "lavaan-0.7.2")
  anchor <- magmaan(m, d, inference = FALSE, options = preset)
  base <- as_lab_fit(anchor)
  expect_null(base$fitting$requested$starts)
  expect_identical(base$fitting$effective$starts, "lavaan-0.7.2")
  table <- coef(summary(anchor))
  # One supplied loading leaves all other parameters on the preset convention.
  table <- table[table$op == "=~" & table$rhs == "x2", ]
  table$est <- 0.8
  partial <- magmaan(m, d, inference = FALSE,
    options = c(preset, list(start = table)))
  warm <- magmaan(m, d, inference = FALSE,
    options = c(preset, list(start = anchor)))
  for (fit in list(partial, warm)) {
    report <- as_lab_fit(fit)$fitting
    expect_identical(report$requested$starts, "table")
    expect_identical(report$effective$starts, "lavaan-0.7.2+table")
    expect_identical(fit$fitting, report)
  }
  pt <- base$partable
  free <- pt[pt$free > 0L, ]
  supplied <- which(free$op == "=~" & free$rhs == "x2")
  expected <- base$fitting$attempts[[1]]$start
  expected[supplied] <- 0.8
  expect_equal(as_lab_fit(partial)$fitting$attempts[[1]]$start, expected)
  expect_equal(as_lab_fit(warm)$fitting$attempts[[1]]$start, base$theta)
})
