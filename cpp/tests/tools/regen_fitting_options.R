#!/usr/bin/env Rscript
# Frozen component oracle for the explicitly supported lavaan fitting version.
# Run from the repository root: Rscript cpp/tests/tools/regen_fitting_options.R
stopifnot(as.character(utils::packageVersion("lavaan")) == "0.7.2")
stopifnot(gsub("-", ".", trimws(readLines("cpp/tests/fixtures/lavaan_version.txt"))) == "0.7.2")
lambda <- c(1, .8, .6, .9)
s <- tcrossprod(lambda) + diag(.7, 4)
dimnames(s) <- list(paste0("x", 1:4), paste0("x", 1:4))
cases <- list(
  list(model = "f =~ x1+x2+x3+x4", std_lv = FALSE),
  list(model = "f =~ x1+x2+x3+x4", std_lv = TRUE),
  list(model = "x4 ~ x1+x2+x3", std_lv = FALSE),
  list(model = "f =~ x1+x2+x3+x4", std_lv = FALSE, invalid_start = TRUE))
for (i in seq_along(cases)) {
  case <- cases[[i]]
  args <- list(model = case$model, sample.cov = s, sample.nobs = 200,
      sample.cov.rescale = FALSE, std.lv = case$std_lv, fixed.x = FALSE,
      meanstructure = FALSE, se = "none", test = "none")
  if (isTRUE(case$invalid_start)) {
    initial <- do.call(lavaan::sem, c(args, list(do.fit = FALSE)))
    args$start <- rep(0, max(lavaan::parTable(initial)$free))
  }
  lv <- suppressWarnings(do.call(lavaan::sem, args))
  pt <- lavaan::parTable(lv)
  cases[[i]] <- c(case, list(sample_cov = unname(lv@SampleStats@cov[[1]]), n = 200,
      parameters = pt[pt$free > 0L, c("lhs", "op", "rhs", "start", "est")],
      fmin = as.numeric(lv@optim$fx), optimizer_x = as.numeric(lv@optim$x),
      converged = lavaan::lavInspect(lv, "converged")))
}
dir.create("cpp/tests/fixtures/fitting", showWarnings = FALSE)
jsonlite::write_json(list(source = "regen_fitting_options.R; lavaan::sem, synthetic covariance",
    lavaan_version = "0.7.2", cases = cases), "cpp/tests/fixtures/fitting/lavaan_0_7_2.json",
    auto_unbox = TRUE, digits = 17, pretty = TRUE, na = "null")
