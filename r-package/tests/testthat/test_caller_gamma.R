caller_gamma_data <- function(grouped = FALSE) {
  set.seed(4101)
  n <- 500L
  eta <- rt(n, 7)
  X <- outer(eta, c(1, 0.8, 0.7, 0.6)) + matrix(rnorm(n * 4L), n, 4L)
  colnames(X) <- paste0("x", 1:4)
  d <- as.data.frame(X)
  if (grouped) d$g <- rep(c("a", "b"), c(300L, 200L))
  d
}

caller_gamma_blocks <- function(fit, d) {
  blocks <- if (is.null(d$g)) list(as.matrix(d)) else {
    lapply(fit$group_labels, function(g) as.matrix(d[d$g == g, paste0("x", 1:4)]))
  }
  # Use the public C++ empirical covariance primitive, in the fitted OV order.
  ov <- magmaan_core$model_matrix_rep(fit$partable)$ov_names
  if (!is.list(ov)) ov <- list(ov)
  lapply(seq_along(blocks), function(b) {
    X <- blocks[[b]][, ov[[b]], drop = FALSE]
    if (fit$meanstructure) magmaan_core$robust_empirical_gamma_with_means(X) else {
      magmaan_core$robust_empirical_gamma(X)
    }
  })
}

test_that("caller Gamma reproduces continuous ML and LS MI and releases", {
  for (means in c(FALSE, TRUE)) for (grouped in c(FALSE, TRUE)) {
    d <- caller_gamma_data(grouped)
    for (estimator in c("ML", "GLS", "DWLS", "WLS", "ULS", "DLS", "supplied")) {
      fit <- fit_model("f =~ x1 + a*x2 + a*x3 + x4", d,
                       estimator = if (estimator == "supplied") "WLS" else estimator, meanstructure = means,
                       groups = if (grouped) "g" else NULL)
      if (estimator == "supplied") {
        fit <- fit_model("f =~ x1 + a*x2 + a*x3 + x4", d, estimator = "WLS",
                         W = fit$W, meanstructure = means,
                         groups = if (grouped) "g" else NULL)
      }
      expect_true(fit$converged, info = estimator)
      blocks <- caller_gamma_blocks(fit, d)
      gamma <- if (grouped) blocks else blocks[[1L]]
      for (worker in list(modification_indices_robust, score_tests_robust)) {
        raw <- worker(fit, data = d, bread = "expected", estimated_weight = FALSE)
        supplied <- worker(fit, gamma = gamma, bread = "expected", estimated_weight = FALSE)
        expect_gt(nrow(raw), 0L)
        expect_equal(supplied, raw, tolerance = 1e-10, info = estimator)
        expect_equal(worker(fit, gamma = blocks, bread = "expected", estimated_weight = FALSE),
                     raw, tolerance = 1e-10)
        # A changed meat must change scaling without changing the fitting W.
        doubled <- worker(fit, gamma = lapply(blocks, function(G) 2 * G),
                          bread = "expected", estimated_weight = FALSE)
        expect_equal(doubled$mi, raw$mi, tolerance = 1e-10)
        if (estimator == "ML") {
          # Exact likelihood meat includes group-score constants, which do not
          # scale when only the centered NACOV is multiplied by two.
          expect_true(all(doubled$scaling.factor <= 2 * raw$scaling.factor + 1e-10))
          expect_true(all(doubled$scaling.factor >= raw$scaling.factor - 1e-10))
          comparator <- worker(fit, gamma = gamma, bread = "expected",
                               moments = "structured", estimated_weight = FALSE)
          comparator_doubled <- worker(fit, gamma = lapply(blocks, function(G) 2 * G),
                               bread = "expected", moments = "structured", estimated_weight = FALSE)
          expect_equal(comparator_doubled$scaling.factor,
                       2 * comparator$scaling.factor, tolerance = 1e-10)
        } else {
          expect_equal(doubled$scaling.factor, 2 * raw$scaling.factor, tolerance = 1e-10)
        }
      }
      if (estimator == "ML") {
        raw <- score_tests_robust(fit, data = d, estimated_weight = FALSE)
        expect_equal(score_tests_robust(fit, gamma = gamma, estimated_weight = FALSE),
                     raw, tolerance = 1e-10)
      }
    }
  }
})

test_that("caller NACOV preserves ordinal fitting weights", {
  d <- caller_gamma_data(TRUE)
  for (j in paste0("x", 1:4)) d[[j]] <- as.integer(cut(d[[j]], c(-Inf, -0.5, 0.5, Inf)))
  for (grouped in c(FALSE, TRUE)) for (estimator in c("DWLS", "WLS")) {
    fit <- fit_model("f =~ x1 + a*x2 + a*x3 + x4", d,
                     estimator = estimator, ordered = paste0("x", 1:4),
                     groups = if (grouped) "g" else NULL)
    gamma <- fit$ordinal_stats$NACOV
    for (worker in list(modification_indices_robust, score_tests_robust)) {
      raw <- worker(fit, estimated_weight = FALSE)
      expect_gt(nrow(raw), 0L)
      expect_equal(worker(fit, gamma = gamma, estimated_weight = FALSE), raw, tolerance = 1e-10)
      if (!grouped) expect_equal(worker(fit, gamma = gamma[[1L]], estimated_weight = FALSE),
                                raw, tolerance = 1e-10)
      doubled <- worker(fit, gamma = lapply(gamma, function(G) 2 * G), estimated_weight = FALSE)
      expect_equal(doubled$mi, raw$mi, tolerance = 1e-10)
      expect_equal(doubled$scaling.factor, 2 * raw$scaling.factor, tolerance = 1e-10)
    }
  }
})

test_that("caller Gamma validates shape, symmetry, PSD and influence provenance", {
  d <- caller_gamma_data()
  fit <- fit_model("f =~ x1 + a*x2 + a*x3 + x4", d, estimator = "DWLS")
  G <- caller_gamma_blocks(fit, d)[[1L]]
  for (worker in list(modification_indices_robust, score_tests_robust)) {
    expect_error(worker(fit, gamma = G), "UnsupportedInference.*casewise.*estimated_weight = FALSE")
    call <- function(gamma, ...) worker(fit, gamma = gamma, bread = "expected",
                                        estimated_weight = FALSE, ...)
    expect_error(call(diag(2)), "dimension mismatch")
    expect_error(call(list(G, G)), "group count")
    asymmetric <- G; asymmetric[1, 2] <- asymmetric[1, 2] + 1
    expect_error(call(asymmetric), "symmetric")
    indefinite <- diag(nrow(G)); indefinite[1, 1] <- -1
    expect_error(call(indefinite), "positive semidefinite")
    nonfinite <- G; nonfinite[1, 1] <- NA_real_
    expect_error(call(nonfinite), "finite")
    expect_error(call("NT"), "numeric matrix")
    expect_error(call(G, cov = "model_implied"), "requires cov='empirical'")
    # Singular PSD covariance is accepted (the core may reject zero meat).
    singular <- G; singular[1, ] <- 0; singular[, 1] <- 0
    expect_true(all(is.finite(call(singular)$mi.scaled)))
  }
  grouped <- fit_model("f =~ x1+x2+x3+x4", caller_gamma_data(TRUE),
                       estimator = "DWLS", groups = "g")
  expect_error(modification_indices_robust(grouped, gamma = G, estimated_weight = FALSE),
               "list of matrices")
  fiml <- fit_model("f =~ x1+x2+x3+x4", d, estimator = "FIML")
  expect_error(modification_indices_robust(fiml, gamma = G, estimated_weight = FALSE),
               "UnsupportedInference.*FIML and ML2S")
  two_stage <- fit_model("f =~ x1+x2+x3+x4", d, estimator = "ML2S")
  expect_error(score_tests_robust(two_stage, gamma = G, estimated_weight = FALSE),
               "UnsupportedInference.*FIML and ML2S")
})

test_that("caller Gamma reaches continuous covariance and profile LRT adapters", {
  for (means in c(FALSE, TRUE)) for (grouped in c(FALSE, TRUE)) {
    d <- caller_gamma_data(grouped)
    X <- if (grouped) lapply(c("a", "b"), function(g) as.matrix(d[d$g == g, paste0("x", 1:4)])) else list(as.matrix(d))
    for (estimator in c("ML", "ULS", "GLS", "WLS")) {
      h1 <- fit_model("f =~ x1+x2+x3+x4", d, estimator = estimator,
                      meanstructure = means, groups = if (grouped) "g" else NULL)
      h0 <- fit_model("f =~ x1+a*x2+a*x3+x4", d, estimator = estimator,
                      meanstructure = means, groups = if (grouped) "g" else NULL)
      G <- caller_gamma_blocks(h1, d)
      if (estimator == "ML") {
        worker <- magmaan_core$infer_ml_profile_lrt
      } else {
        worker <- magmaan_core$infer_continuous_ls_profile_lrt
        raw <- magmaan_core$robust_continuous_ls(h1, X, fixed_weight = TRUE)
        supplied <- magmaan_core$robust_continuous_ls(h1, gamma = G, fixed_weight = TRUE)
        expect_equal(supplied, raw, tolerance = 1e-10)
        expect_error(magmaan_core$robust_continuous_ls(h1, gamma = G),
                     "UnsupportedInference.*fixed_weight = TRUE")
      }
      expect_equal(worker(h1, h0, gamma = G), worker(h1, h0, X), tolerance = 1e-10)
    }
  }
})


test_that("explicit Gamma_NT is available for complete ML releases", {
  d <- caller_gamma_data()
  fit <- fit_model("f =~ x1+a*x2+a*x3+x4", d, estimator = "ML", meanstructure = FALSE)
  Sigma <- magmaan_core$model_implied(fit)$sigma
  if (is.list(Sigma)) Sigma <- Sigma[[1L]]
  G <- magmaan_core$robust_gamma_nt(Sigma)
  ordinary <- score_tests(fit, cov = "model_implied", estimated_weight = FALSE,
                          bread = "expected")
  # Gamma_NT reduction is a structured metric convention, not the exact
  # likelihood recipe's affine mean/score correction.
  explicit <- score_tests_robust(fit, gamma = G, bread = "expected",
                                  moments = "structured", estimated_weight = FALSE)
  expect_equal(explicit$mi.scaled, ordinary$mi, tolerance = 1e-10)
  expect_equal(explicit$scaling.factor, rep(1, nrow(explicit)), tolerance = 1e-10)
  expect_error(score_tests(fit, gamma = G, cov = "model_implied", estimated_weight = FALSE),
               "requires cov='empirical'")
})

test_that("caller NACOV reaches supported mixed ordinal LS score routes", {
  d <- caller_gamma_data()
  for (j in c("x1", "x2")) d[[j]] <- as.integer(cut(d[[j]], c(-Inf, -0.5, 0.5, Inf)))
  for (estimator in c("DWLS", "WLS")) {
    fit <- fit_model("f =~ x1+a*x2+a*x3+x4", d, estimator = estimator,
                     ordered = c("x1", "x2"))
    G <- fit$mixed_ordinal_stats$NACOV
    for (worker in list(modification_indices_robust, score_tests_robust)) {
      raw <- worker(fit, estimated_weight = FALSE)
      expect_gt(nrow(raw), 0L)
      expect_equal(worker(fit, gamma = G, estimated_weight = FALSE), raw, tolerance = 1e-10)
      doubled <- worker(fit, gamma = lapply(G, function(g) 2 * g), estimated_weight = FALSE)
      expect_equal(doubled$mi, raw$mi, tolerance = 1e-10)
      expect_equal(doubled$scaling.factor, 2 * raw$scaling.factor, tolerance = 1e-10)
      expect_error(worker(fit, gamma = G), "UnsupportedInference.*casewise")
    }
  }
})
