#!/usr/bin/env Rscript
# Frozen component oracle for the explicitly supported lavaan fitting version.
# Run from the repository root: Rscript cpp/tests/tools/regen_fitting_options.R
stopifnot(as.character(utils::packageVersion("lavaan")) == "0.7.2")
stopifnot(gsub("-", ".", trimws(readLines("cpp/tests/fixtures/lavaan_version.txt"))) == "0.7.2")
lambda <- c(1, .8, .6, .9)
s <- tcrossprod(lambda) + diag(.7, 4)
dimnames(s) <- list(paste0("x", 1:4), paste0("x", 1:4))
# `rescale` multiplies x1 and divides x3 by k, so the pinned search accepts
# only a standardized retry (k = 100, 1000) or no attempt at all (k = 1e5).
# At k = 1000 lavaan accepts fmin = 0.276 on exact-fit moments. For k >= 1000
# the endpoints depend on floating-point paths (`endpoint_parity = FALSE`):
# starts, coordinates and verdicts stay comparable, estimates do not.
cases <- list(
  list(model = "f =~ x1+x2+x3+x4", std_lv = FALSE),
  list(model = "f =~ x1+x2+x3+x4", std_lv = TRUE),
  list(model = "x4 ~ x1+x2+x3", std_lv = FALSE),
  list(model = "f =~ x1+x2+x3+x4", std_lv = FALSE, invalid_start = TRUE),
  list(model = "f1 =~ x1\nx1 ~~ 0.3*x1\ng =~ f1 + x2 + x3 + x4", std_lv = FALSE),
  list(model = "f =~ x1+x2+x3+x4", std_lv = FALSE, rescale = 100),
  list(model = "f =~ x1+x2+x3+x4", std_lv = FALSE, rescale = 1000, endpoint_parity = FALSE),
  list(model = "f =~ x1+x2+x3+x4", std_lv = FALSE, rescale = 1e5, endpoint_parity = FALSE))
# Capture the actual optimizer calls, including retries. The public optim$x is
# expanded; return attributes retain the packed start and reduced gradient.
.fitting_attempts <- list()
trace("lav_model_est", where = asNamespace("lavaan"), print = FALSE,
    exit = quote({
      out <- returnValue()
      .GlobalEnv$.fitting_attempts[[length(.GlobalEnv$.fitting_attempts) + 1L]] <- list(
          simple = start == "simple", standardized = lavoptions$optim.parscale != "none",
          start = as.numeric(attr(out, "start")),
          parameter_scale = as.numeric(attr(out, "parscale")),
          port_scale = if (exists("scale_1", inherits = FALSE)) as.numeric(scale_1) else numeric(),
          gradient = as.numeric(attr(out, "dx")),
          theta = as.numeric(out), accepted = isTRUE(attr(out, "converged")),
          iterations = attr(out, "iterations"))
    }))
cases <- c(cases, list(
    list(model = "f =~ x1+a*x2+a*x3+x4", std_lv = FALSE, equality = TRUE),
    list(model = "f =~ x1+a*x2+a*x3+x4", std_lv = TRUE, equality = TRUE),
    list(model = "f =~ a*x1+b*x2+c*x3+d*x4\na == c\nb == d", std_lv = TRUE, equality = TRUE),
    list(model = "f =~ a*x1+b*x2+c*x3+d*x4\nb == d\na == c", std_lv = TRUE, equality = TRUE),
    list(model = "f =~ x1+a*x2+b*x3+c*x4\na == b\nb == c\na == c", std_lv = FALSE, equality = TRUE),
    list(model = "f =~ x1+a*x2+b*x3+x4\na == 2*b", std_lv = FALSE, equality = TRUE),
    list(model = "f =~ x1+a*x2+b*x3+x4\na + b == 1.5", std_lv = FALSE, equality = TRUE),
    list(model = "f =~ x1+a*x2+b*x3+c*x4\na == b\nb == c", std_lv = FALSE, equality = TRUE),
    list(model = "f =~ x1+x2+x3+x4", std_lv = FALSE, equality = TRUE, group_equal = c("loadings", "intercepts")),
    list(model = "f =~ x1+a*x2+a*x3+x4", std_lv = FALSE, equality = TRUE, rescale = 100)))
for (i in seq_along(cases)) {
  case <- cases[[i]]
  si <- s
  if (!is.null(case$rescale)) {
    d <- diag(c(case$rescale, 1, 1 / case$rescale, 1))
    si <- d %*% s %*% d
    dimnames(si) <- dimnames(s)
  }
  args <- list(model = case$model, sample.cov = si, sample.nobs = 200,
      sample.cov.rescale = FALSE, std.lv = case$std_lv, fixed.x = FALSE,
      meanstructure = FALSE, se = "none", test = "none")
  if (!is.null(case$group_equal)) {
    args$sample.cov <- list(si, si * 1.2)
    args$sample.nobs <- c(200, 180)
    args$sample.mean <- list(c(.1, .2, .3, .4), c(.3, .4, .5, .6))
    args$meanstructure <- TRUE
    args$group.equal <- case$group_equal
  }
  if (isTRUE(case$invalid_start)) {
    initial <- do.call(lavaan::sem, c(args, list(do.fit = FALSE)))
    args$start <- rep(0, max(lavaan::parTable(initial)$free))
  }
  .fitting_attempts <- list()
  lv <- suppressWarnings(do.call(lavaan::sem, args))
  pt <- lavaan::parTable(lv)
  cases[[i]] <- c(case, list(sample_cov = unname(lv@SampleStats@cov[[1]]), n = 200,
      parameters = pt[pt$free > 0L, c("lhs", "op", "rhs", "start", "est")],
      fmin = as.numeric(lv@optim$fx), optimizer_x = as.numeric(lv@optim$x),
      parscale = as.numeric(lv@optim$parscale),
      converged = lavaan::lavInspect(lv, "converged")))
  if (isTRUE(case$equality)) {
    cases[[i]]$parameters <- pt[pt$free > 0L, c("lhs", "op", "rhs", "group", "start", "est")]
    cases[[i]]$covariances <- lapply(lv@SampleStats@cov, unname)
    cases[[i]]$means <- lapply(lv@SampleStats@mean, unname)
    cases[[i]]$n_obs <- as.numeric(lv@SampleStats@nobs)
    cases[[i]]$jacobian <- unname(lv@Model@ceq.JAC)
    cases[[i]]$basis <- unname(lv@Model@eq.constraints.K)
    cases[[i]]$offset <- as.numeric(lv@Model@eq.constraints.k0)
    cases[[i]]$attempts <- .fitting_attempts
  }
}
untrace("lav_model_est", where = asNamespace("lavaan"))
dir.create("cpp/tests/fixtures/fitting", showWarnings = FALSE)
jsonlite::write_json(list(source = "regen_fitting_options.R; lavaan::sem, synthetic covariance",
    lavaan_version = "0.7.2", cases = cases), "cpp/tests/fixtures/fitting/lavaan_0_7_2.json",
    auto_unbox = TRUE, digits = 17, pretty = TRUE, na = "null")
