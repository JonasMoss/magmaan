expect_convention_measures <- function(fit, oracle, convention) {
  # Test the post-fit composer at the oracle point, separately from optimizer
  # endpoint accuracy. Match semantic rows rather than assuming free order.
  pt <- fit$partable
  op <- lavaan::parTable(oracle)
  key <- function(x) paste(x$lhs, x$op, x$rhs, x$group, sep="/")
  rows <- which(pt$free > 0)
  target <- op$est[match(key(pt)[rows], key(op))]
  expect_false(anyNA(target))
  expect_true(max(abs(fit$theta[pt$free[rows]] - target)) < 1e-4)
  fit$theta[pt$free[rows]] <- target
  actual <- convention_fit_measures(fit, convention)
  expected <- lavaan::fitMeasures(oracle)
  expect_identical(attr(actual, "lavaan_compat"), convention)
  expect_false(anyDuplicated(actual$index) > 0)
  values <- setNames(actual$estimate, actual$index)
  for (index in intersect(names(values), names(expected))) {
    x <- values[[index]]; y <- expected[[index]]
    if (is.na(y)) expect_true(is.na(x), info=index)
    else expect_true(abs(x-y) <= 1e-6 * max(1, abs(x), abs(y)), info=paste(convention,index,x,y))
  }
}

test_that("complete ML compatibility fit measures match lavaan families", {
  skip_if_not_installed("lavaan")
  syntax <- "visual =~ x1+x2+x3\ntextual =~ x4+x5+x6\nspeed =~ x7+x8+x9"
  d <- lavaan::HolzingerSwineford1939
  for (grouped in c(FALSE, TRUE)) {
    spec <- model_spec(syntax, meanstructure=TRUE,
      group=if (grouped) "school" else NULL,
      group_labels=if (grouped) unique(as.character(d$school)) else NULL)
    fit <- fit_model(spec, d)
    expect_true(fit$converged)
    for (convention in c("ML", "MLM", "MLR")) {
      oracle <- lavaan::cfa(syntax, d, estimator=convention, meanstructure=TRUE,
        group=if (grouped) "school" else NULL)
      expect_convention_measures(fit, oracle, convention)
    }
  }
  syntax <- 'ind60 =~ x1+x2+x3\ndem60 =~ y1+y2+y3+y4\ndem65 =~ y5+y6+y7+y8\ndem60 ~ ind60\ndem65 ~ ind60+dem60\ny1 ~~ y5\ny2 ~~ y4+y6\ny3 ~~ y7\ny4 ~~ y8\ny6 ~~ y8'
  fit <- fit_model(model_spec(syntax, meanstructure=TRUE), lavaan::PoliticalDemocracy)
  for (convention in c("ML", "MLM", "MLR"))
    expect_convention_measures(fit, lavaan::sem(syntax, lavaan::PoliticalDemocracy,
      estimator=convention, meanstructure=TRUE), convention)
})

test_that("FIML compatibility families remain explicitly unavailable pending validation", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  vars <- paste0("x",1:6)
  set.seed(109)
  d$x2[runif(nrow(d)) < plogis(d$x1-5)] <- NA
  d$x5[runif(nrow(d)) < .2] <- NA
  syntax <- "visual =~ x1+x2+x3\ntextual =~ x4+x5+x6"
  fit <- fit_model(model_spec(syntax, meanstructure=TRUE, fixed_x=FALSE), d, estimator="FIML")
  for (convention in c("ML", "MLR")) {
    actual <- convention_fit_measures(fit, convention)
    expect_identical(attr(actual, "lavaan_compat"), convention)
    expect_true(all(is.na(actual$estimate)))
    expect_true(all(actual$reason == "unsupported_model: not yet validated against lavaan"))
  }
})

test_that("ordinal convention measures use n minus groups and CATML robust families", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  vars <- paste0("x", 1:6)
  for (v in vars) d[[v]] <- ordered(cut(d[[v]],
    quantile(d[[v]], c(0,1/3,2/3,1)), include.lowest=TRUE, labels=FALSE))
  syntax <- "visual =~ x1+x2+x3\ntextual =~ x4+x5+x6"
  for (grouped in c(FALSE, TRUE)) for (p in c("delta", "theta"))
    for (convention in c("DWLS", "WLSMV", "ULS", "ULSMV", "WLS")) {
      estimator <- if (convention %in% c("DWLS", "WLSMV")) "DWLS" else if (convention == "WLS") "WLS" else "ULS"
      spec <- model_spec(syntax, ordered=vars, meanstructure=TRUE,
        parameterization=p, group=if (grouped) "school" else NULL,
        group_labels=if (grouped) unique(as.character(d$school)) else NULL)
      fit <- fit_model(spec, d, estimator=estimator)
      oracle <- lavaan::cfa(syntax, d, ordered=vars, estimator=convention,
        parameterization=p, group=if (grouped) "school" else NULL)
      expect_convention_measures(fit, oracle, convention)
    }
})

test_that("convention fit measures reject incompatible bundles and fit states", {
  skip_if_not_installed("lavaan")
  fit <- fit_model(model_spec("visual =~ x1+x2+x3"), lavaan::HolzingerSwineford1939)
  expect_error(convention_fit_measures(fit,"WLSMV"),"different fitted estimator")
  fit$converged <- FALSE
  expect_error(convention_fit_measures(fit,"ML"),"convergence verdict")
})
