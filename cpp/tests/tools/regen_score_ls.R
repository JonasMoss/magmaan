#!/usr/bin/env Rscript
# Regenerate only the convention-matched GLS and ADF/WLS ordinary MI fixtures.
# Run from any directory: Rscript cpp/tests/tools/regen_score_ls.R
suppressPackageStartupMessages({library(lavaan); library(jsonlite)})
script <- sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1L])
root <- normalizePath(file.path(dirname(script), "../../.."))
fixtures <- file.path(root, "cpp/tests/fixtures")
pinned <- trimws(readLines(file.path(fixtures, "lavaan_version.txt"))[1L])
installed <- as.character(packageVersion("lavaan"))
stopifnot(gsub("-", ".", pinned, fixed = TRUE) == installed)
syntax <- "f =~ x1+x2+x3+x4\nx1 ~~ 0*x2"
d <- HolzingerSwineford1939[paste0("x", 1:4)]
for (estimator in c("GLS", "WLS")) {
  fit <- cfa(syntax, d, estimator = estimator, meanstructure = FALSE,
             information = "expected")
  stopifnot(lavInspect(fit, "converged"))
  pt <- parTable(fit)
  free <- pt[pt$free > 0, ]
  free <- free[order(free$free), ]
  S <- lavInspect(fit, "sampstat")$cov
  W <- lavInspect(fit, "wls.v")
  N <- lavInspect(fit, "ntotal")
  stopifnot(identical(lavInspect(fit, "options")$sample.cov.rescale, FALSE))
  stopifnot(max(abs(S - cov(d))) < 1e-12)
  reference <- modindices(fit, information = "expected")
  targets <- reference[reference$op == "~~" &
    ((reference$lhs == "x1" & reference$rhs == "x2") |
     (reference$lhs == "x2" & reference$rhs == "x3")), ]
  stopifnot(nrow(targets) == 2L)

  # Independent one-factor covariance and analytic Jacobian. No
  # lavaan score/information worker enters the score/Schur reconstruction.
  moments <- function(theta) {
    pars <- pt$est
    pars[pt$free > 0] <- theta[pt$free[pt$free > 0]]
    lambda <- c(1, pars[pt$op == "=~" & pt$free > 0])
    phi <- pars[pt$op == "~~" & pt$lhs == "f" & pt$rhs == "f"]
    residual <- pars[pt$op == "~~" & pt$lhs == pt$rhs & pt$lhs != "f"]
    implied <- phi * tcrossprod(lambda) + diag(residual)
    implied[lower.tri(implied, diag = TRUE)]
  }
  theta <- free$est
  lambda <- c(1, free$est[free$op == "=~"])
  phi <- free$est[free$op == "~~" & free$lhs == "f"]
  J <- vapply(seq_len(nrow(free)), function(k) {
    row <- free[k, ]
    derivative <- matrix(0, 4, 4)
    if (row$op == "=~") {
      direction <- rep(0, 4)
      direction[match(row$rhs, names(d))] <- 1
      derivative <- phi * (tcrossprod(direction, lambda) + tcrossprod(lambda, direction))
    } else if (row$lhs == "f") {
      derivative <- tcrossprod(lambda)
    } else {
      j <- match(row$lhs, names(d))
      derivative[j, j] <- 1
    }
    derivative[lower.tri(derivative, diag = TRUE)]
  }, numeric(nrow(W)))
  residual <- S[lower.tri(S, diag = TRUE)] - moments(theta)
  rows <- lapply(seq_len(nrow(targets)), function(i) {
    t <- targets[i, ]
    E <- matrix(0, 4, 4)
    a <- match(t$lhs, names(d)); b <- match(t$rhs, names(d))
    E[a, b] <- E[b, a] <- 1
    direction <- E[lower.tri(E, diag = TRUE)]
    cross <- drop(crossprod(J, W %*% direction))
    efficient <- direction - J %*% solve(crossprod(J, W %*% J), cross)
    score <- drop(crossprod(efficient, W %*% residual))
    information <- drop(crossprod(efficient, W %*% efficient))
    reconstructed <- N * score^2 / information
    # lavaan continuous GLS/WLS scales its score by (N-1)/N but leaves expected information
    # unscaled. lav_model_grad uses (nobs-1)/ntotal for these estimators;
    # modindices uses N * candidate_score^2 / Schur information. Keep raw
    # oracle values and record this transport explicitly.
    score_convention <- (N - 1) / N
    cat(estimator, t$lhs, t$rhs, "reconstructed", reconstructed, "reference", t$mi, "ratio", reconstructed/t$mi, "N", N, "\n")
    # modindices assumes the fitted nuisance score is zero. Check that its
    # unprojected candidate-score form matches before checking our efficient
    # score, whose small extra correction retains the actual nuisance score.
    candidate_score <- drop(crossprod(direction, W %*% residual))
    stationary_mi <- N * (candidate_score * score_convention)^2 / information
    stopifnot(abs(stationary_mi - t$mi) < 1e-8)
    stopifnot(abs(candidate_score * score_convention / information - t$epc) < 1e-8)
    stopifnot(abs(reconstructed * score_convention^2 - t$mi) < 1e-5)
    stopifnot(abs(score / information * score_convention - t$epc) < 1e-5)
    list(lhs = t$lhs, op = t$op, rhs = t$rhs, group = 1L,
         mi = t$mi, epc = t$epc, sepc_lv = t$sepc.lv, sepc_all = t$sepc.all,
         score_convention = score_convention,
         stationary_mi = stationary_mi,
         nuisance_score_max = max(abs(crossprod(J, W %*% residual))),
         independent_mi = reconstructed,
         independent_epc = score / information)
  })
  id <- if (estimator == "GLS") "0013_gls_fixed_and_absent" else "0014_wls_fixed_and_absent"
  payload <- list(`_meta` = list(format_version = 1L, fixture_kind = "score",
    corpus_id = id, tool = "lavaan::modindices", lavaan_version = installed,
    data = "lavaan::HolzingerSwineford1939 x1..x4",
    moment_order = "column-major lower triangle including diagonal (x1..x4)",
    sample_cov_rescale = FALSE, sample_cov_divisor = "N-1",
    score_source = "lavaan 0.7-2 lav_model_grad: GLS/WLS group score factor (nobs-1)/ntotal; modindices: N times squared candidate gradient over unscaled expected-information Schur complement",
    note = "Expected information; frozen fitted W and sample covariance preserve estimator-specific divisors. Independent df=1 score/Schur reconstruction uses a one-factor covariance formula and analytic derivatives."),
    kind = "ls", estimator = estimator, input = syntax, meanstructure = FALSE,
    weight = list(list(block = 0L, matrix = unname(W))),
    fit = list(converged = TRUE, theta_hat = theta, n_obs = N,
      n_obs_per_block = as.integer(N), score_information = "expected",
      sample_cov = list(list(block = 0L, matrix = unname(S))), sample_mean = NULL,
      modindices = rows, score_tests = list(total = NULL, rows = list())))
  write_json(payload, file.path(fixtures, "score", paste0(id, ".score.json")),
             pretty = TRUE, auto_unbox = TRUE, digits = NA, null = "null")
  cat(estimator, "independent maximum MI error:",
      max(abs(vapply(rows, function(x) x$independent_mi * x$score_convention^2 - x$mi, numeric(1)))), "\n")
}
