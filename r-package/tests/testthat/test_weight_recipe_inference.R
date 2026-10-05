# Estimated-weight inference follows the weight recipe a fit records, never its
# computational label, and two-stage (ML2S) fits dispatch to the ML2S score
# tests. DLS with a = 0 and a = 1 rebuilds the NT and ADF weights, which gives
# exact cross-recipe reductions without an oracle.

recipe_data <- function(n = 400L, seed = 20261002L) {
  set.seed(seed)
  eta <- rt(n, df = 6) * sqrt(4 / 6)
  noise <- matrix(rt(n * 5L, df = 6), n, 5L) * 0.6
  X <- outer(eta, c(1, 0.8, 0.7, 0.6, 0.5)) + noise
  X[, 5L] <- X[, 5L] + 0.35 * X[, 4L]
  colnames(X) <- paste0("x", 1:5)
  as.data.frame(X)
}

recipe_fits <- function(spec, d) {
  list(
    nt = fit_model(spec, d, estimator = "GLS"),
    adf = fit_model(spec, d, estimator = "WLS"),
    dwls = fit_model(spec, d, estimator = "DWLS"),
    dls0 = fit_model(spec, d, estimator = "DLS", dls_a = 0),
    dls1 = fit_model(spec, d, estimator = "DLS", dls_a = 1),
    dls3 = fit_model(spec, d, estimator = "DLS", dls_a = 0.3)
  )
}

loading_parameter <- function(fit, rhs) {
  fit$partable$free[fit$partable$op == "=~" & fit$partable$rhs == rhs]
}

test_that("estimated-weight inference reads the recorded continuous recipe", {
  d <- recipe_data()
  fits <- recipe_fits(model_spec("f =~ x1 + x2 + x3 + x4 + x5",
                                 meanstructure = FALSE), d)
  for (nm in names(fits)) expect_false(is.null(fits[[nm]]$composition$weight), info = nm)

  mi <- lapply(fits, modification_indices_robust, data = d, bread = "expected",
               estimated_weight = TRUE)
  for (nm in names(mi)) {
    expect_true(all(is.finite(mi[[nm]]$mi.scaled)), info = nm)
  }
  # DLS(1) rebuilds the ADF weight exactly; DLS(0) and GLS are separate
  # optimizations of the same objective, so they agree to optimizer precision.
  expect_equal(mi$dls0$mi.scaled, mi$nt$mi.scaled, tolerance = 1e-5)
  expect_equal(mi$dls1$mi.scaled, mi$adf$mi.scaled, tolerance = 1e-10)
  expect_gt(max(abs(mi$dls3$mi.scaled - mi$dls0$mi.scaled)), 1e-3)
  expect_gt(max(abs(mi$dls3$mi.scaled - mi$dls1$mi.scaled)), 1e-3)
  # The weight influence is not zero for the estimated recipes.
  for (nm in c("dwls", "dls3")) {
    fixed <- modification_indices_robust(fits[[nm]], data = d, estimated_weight = FALSE, bread = "expected")
    expect_gt(max(abs(fixed$mi.scaled - mi[[nm]]$mi.scaled)), 1e-4)
  }

  res <- lapply(fits[c("nt", "adf", "dls0", "dls1", "dwls")], residuals,
                standardized = TRUE, estimated_weight = TRUE, data = d)
  expect_equal(res$dls0, res$nt, tolerance = 1e-5)
  expect_equal(res$dls1, res$adf, tolerance = 1e-10)

  influence <- lapply(fits[c("nt", "adf", "dls0", "dls1", "dls3")],
                      est_change_raw_approx, type = "estimated.weight")
  expect_equal(unclass(influence$dls0)[, ], unclass(influence$nt)[, ],
               tolerance = 1e-5)
  expect_equal(unclass(influence$dls1)[, ], unclass(influence$adf)[, ],
               tolerance = 1e-10)

  X <- as.matrix(d)
  profile <- lapply(fits[c("nt", "dls0", "dls3")], function(fit) {
    k <- loading_parameter(fit, "x3")
    magmaan_core$frontier_profile_lrt_parameter_gmm(
      fit, k, 0.95 * fit$theta[k], raw_data = X, robust = TRUE,
      estimated_weight = TRUE)
  })
  expect_equal(profile$dls0$scaling_factor, profile$nt$scaling_factor,
               tolerance = 1e-5)
  expect_true(is.finite(profile$dls3$scaling_factor))

  rbm <- lapply(fits[c("nt", "dls0")], magmaan_core$frontier_rbm, raw_data = X)
  expect_equal(rbm$dls0$theta, rbm$nt$theta, tolerance = 1e-5)
})

test_that("release score tests read the recorded recipe", {
  d <- recipe_data()
  spec <- model_spec("f =~ x1 + a*x2 + a*x3 + x4 + x5", meanstructure = FALSE)
  fits <- recipe_fits(spec, d)
  st <- lapply(fits, score_tests_robust, data = d, bread = "expected", estimated_weight = TRUE)
  for (nm in names(st)) expect_true(all(is.finite(st[[nm]]$mi.scaled)), info = nm)
  expect_equal(st$dls0$mi.scaled, st$nt$mi.scaled, tolerance = 1e-5)
  expect_equal(st$dls1$mi.scaled, st$adf$mi.scaled, tolerance = 1e-10)
  # The ordinary release tests use the recorded fitting weight.
  expect_true(all(is.finite(score_tests(fits$dwls, bread = "expected", estimated_weight = FALSE, cov = "model_implied")$mi)))
})

test_that("supplied weights are used as fitted but have no estimated-weight recipe", {
  d <- recipe_data()
  spec <- model_spec("f =~ x1 + x2 + x3 + x4 + x5", meanstructure = FALSE)
  adf <- fit_model(spec, d, estimator = "WLS")
  supplied <- fit_model(spec, d, estimator = "WLS", W = adf$W)
  expect_identical(supplied$composition$weight, "custom")

  rbm_fixed <- magmaan_core$frontier_rbm(
    supplied, raw_data = as.matrix(d), estimated_weight = FALSE)
  expect_equal(rbm_fixed$theta, magmaan_core$frontier_rbm(
    adf, raw_data = as.matrix(d), estimated_weight = FALSE)$theta,
    tolerance = 1e-6)
  expect_error(magmaan_core$frontier_rbm(
    supplied, raw_data = as.matrix(d)), "UnsupportedInference")
  expect_error(magmaan_core$frontier_rbm(
    supplied, raw_data = as.matrix(d), estimated_weight = TRUE),
    "UnsupportedInference")

  fixed <- modification_indices_robust(supplied, data = d, estimated_weight = FALSE, bread = "expected")
  expect_equal(fixed$mi.scaled, modification_indices_robust(adf, data = d, estimated_weight = FALSE, bread = "expected")$mi.scaled,
               tolerance = 1e-6)
  expect_equal(modification_indices(supplied, bread = "expected", estimated_weight = FALSE, cov = "model_implied")$mi, modification_indices(adf, bread = "expected", estimated_weight = FALSE, cov = "model_implied")$mi,
               tolerance = 1e-6)
  expect_equal(modification_indices_robust(adf, data = d, weight = adf$W, estimated_weight = FALSE, bread = "expected")$mi.scaled,
               fixed$mi.scaled, tolerance = 1e-6)
  expect_error(modification_indices_robust(adf, data = d, weight = diag(nrow(adf$W)), estimated_weight = FALSE, bread = "expected"),
               "fit\\$W")

  expect_error(modification_indices_robust(supplied, data = d, estimated_weight = TRUE, bread = "expected"),
               "UnsupportedInference")
  expect_error(residuals(supplied, standardized = TRUE, estimated_weight = TRUE, data = d),
               "UnsupportedInference")
  expect_error(est_change_raw_approx(supplied, type = "estimated.weight"),
               "UnsupportedInference")
  k <- loading_parameter(supplied, "x3")
  expect_error(magmaan_core$frontier_profile_lrt_parameter_gmm(
    supplied, k, 0.95 * supplied$theta[k], raw_data = as.matrix(d), robust = TRUE,
    estimated_weight = TRUE), "UnsupportedInference")
})

test_that("ordinal NT, DLS and supplied weights refuse the weight influence", {
  cont <- recipe_data(n = 600L)
  ord <- as.data.frame(lapply(cont[1:4], function(x)
    ordered(cut(x, c(-Inf, -0.5, 0.5, Inf)))))
  spec <- model_spec("f =~ x1 + x2 + x3 + x4", ordered = names(ord),
                     meanstructure = FALSE)
  dwls <- fit_model(spec, ord, estimator = "DWLS")
  expect_true(all(is.finite(
    modification_indices_robust(dwls, estimated_weight = TRUE, bread = "expected")$mi.scaled)))
  supplied <- fit_model(spec, ord, estimator = "DWLS",
                        W = diag(seq(0.7, 1.3, length.out = 14L)))
  fits <- list(nt = fit_model(spec, ord, estimator = "GLS"),
               dls = fit_model(spec, ord, estimator = "DLS", dls_a = 0.3),
               supplied = supplied)
  for (nm in names(fits)) {
    expect_true(all(is.finite(modification_indices_robust(fits[[nm]], estimated_weight = FALSE, bread = "expected")$mi.scaled)),
                info = nm)
    expect_error(modification_indices_robust(fits[[nm]], estimated_weight = TRUE, bread = "expected"),
                 "UnsupportedInference", info = nm)
  }
  # Association ML has no MI contract until 0.3.0.
  ml <- fit_model(spec, ord, estimator = "ML")
  expect_error(modification_indices(ml, bread = "expected", estimated_weight = FALSE, cov = "model_implied"), "association")
  expect_error(modification_indices_robust(ml, estimated_weight = FALSE, bread = "expected"), "association")
})

test_that("two-stage MI and releases reduce to complete-data moment comparators", {
  d <- recipe_data(n = 300L)
  spec <- model_spec("f =~ x1 + x2 + x3 + x4 + x5", meanstructure = TRUE)
  spec_eq <- model_spec("f =~ x1 + a*x2 + a*x3 + x4 + x5", meanstructure = TRUE)
  complete <- c(nt = "ML", uls = "ULS", dwls = "DWLS", adf = "WLS", dls = "DLS")
  for (w in names(complete)) {
    for (s in list(spec, spec_eq)) {
      two <- fit_model(s, d, estimator = "ML2S", stage2_weight = w, dls_a = 0.3)
      one <- fit_model(s, d, estimator = complete[[w]], dls_a = 0.3)
      expect_equal(two$theta, one$theta, tolerance = 1e-10, info = w)
      run <- if (identical(s, spec)) modification_indices_robust else score_tests_robust
      for (ew in c(FALSE, TRUE)) {
        if (ew && w %in% c("nt", "uls")) next
        a <- run(two, estimated_weight = ew, bread = "expected")
        # NT Stage 2 propagates centered moment influence; exact ML score
        # rows also carry candidate/group constants at the evaluation point.
        b <- run(one, data = d, estimated_weight = ew, bread = "expected",
                 moments = if (w == "nt") "structured" else "auto")
        expect_identical(attr(a, "mi_type"), "naive_stage2")
        expect_equal(a$mi, b$mi, tolerance = 1e-10, info = paste(w, ew))
        expect_equal(a$mi.scaled, b$mi.scaled, tolerance = 1e-10, info = paste(w, ew))
      }
      naive <- if (identical(s, spec)) modification_indices(two, bread = "expected", estimated_weight = FALSE, cov = "model_implied") else score_tests(two, bread = "expected", estimated_weight = FALSE, cov = "model_implied")
      expect_identical(attr(naive, "mi_type"), "naive_stage2")
      expect_equal(naive$mi, run(two, bread = "expected")$mi, tolerance = 1e-8, info = w)
    }
  }
})

test_that("two-stage MI under missing data uses the recorded Stage-2 weight", {
  d <- recipe_data(n = 300L)
  set.seed(7L)
  d$x2[sample.int(nrow(d), 40L)] <- NA_real_
  d$x4[sample.int(nrow(d), 40L)] <- NA_real_
  spec <- model_spec("f =~ x1 + x2 + x3 + x4 + x5", meanstructure = TRUE)
  nt <- fit_model(spec, d, estimator = "ML2S")
  dls <- fit_model(spec, d, estimator = "ML2S", stage2_weight = "dls", dls_a = 0.3)

  nt_mi <- modification_indices_robust(nt, estimated_weight = FALSE, bread = "expected")
  expect_true(all(is.finite(nt_mi$mi.scaled) & nt_mi$scaling.factor > 0))
  expect_equal(modification_indices_robust(nt, estimated_weight = TRUE, bread = "expected")$mi.scaled,
               nt_mi$mi.scaled, tolerance = 1e-10)
  fixed <- modification_indices_robust(dls, estimated_weight = FALSE, bread = "expected")
  estimated <- modification_indices_robust(dls, estimated_weight = TRUE, bread = "expected")
  expect_true(all(is.finite(estimated$mi.scaled)))
  expect_equal(estimated$mi, fixed$mi, tolerance = 1e-10)
  expect_gt(max(abs(estimated$mi.scaled - fixed$mi.scaled)), 1e-4)
  expect_equal(modification_indices(dls, bread = "expected", estimated_weight = FALSE, cov = "model_implied")$mi, fixed$mi, tolerance = 1e-8)

  expect_error(modification_indices_robust(nt, information = "observed", estimated_weight = FALSE, bread = "expected"),
               "UnsupportedInference")
  expect_error(modification_indices_robust(nt, data = d, estimated_weight = FALSE, bread = "expected"), "omit `data`")
  expect_error(modification_indices_robust(nt, bread = "observed", estimated_weight = FALSE), "bread")
  expect_error(modification_indices_robust(dls, weight = diag(20), estimated_weight = FALSE, bread = "expected"), "omit `weight`")

  # Refits and case influence use the recorded Stage-2 weight.
  expect_error(magmaan_core$frontier_rbm(dls, stage2_weight = "nt"), "recorded")
  expect_error(magmaan_core$frontier_rbm(dls, dls_a = 0.5), "recorded")
  expect_true(all(is.finite(est_change_raw_approx(dls, type = "estimated.weight"))))
})

test_that("two-stage unequal groups reduce to complete-data moment comparators", {
  d <- rbind(recipe_data(240L, 20261003L), recipe_data(160L, 20261004L))
  d$g <- rep(c("a", "b"), c(240L, 160L))
  estimators <- c(nt = "ML", uls = "ULS", dwls = "DWLS", adf = "WLS", dls = "DLS")
  for (syntax in c("f =~ x1+x2+x3+x4+x5", "f =~ x1+a*x2+a*x3+x4+x5")) {
    spec <- model_spec(syntax, meanstructure = TRUE, group_labels = c("a", "b"))
    for (w in names(estimators)) {
      two <- fit_model(spec, d, groups = "g", estimator = "ML2S",
                       stage2_weight = w, dls_a = 0.3)
      one <- fit_model(spec, d, groups = "g", estimator = estimators[[w]], dls_a = 0.3)
      expect_true(two$converged, info = w)
      expect_true(one$converged, info = w)
      expect_equal(two$theta, one$theta, tolerance = 1e-10, info = w)
      for (release in c(FALSE, TRUE)) {
        if (release && !grepl("a\\*", syntax)) next
        worker <- if (release) score_tests_robust else modification_indices_robust
        ordinary <- if (release) score_tests(two, bread = "expected", estimated_weight = FALSE, cov = "model_implied") else modification_indices(two, bread = "expected", estimated_weight = FALSE, cov = "model_implied")
        for (ew in c(FALSE, TRUE)) {
          a <- worker(two, estimated_weight = ew, bread = "expected")
          b <- worker(one, data = d,
                       estimated_weight = ew && !w %in% c("nt", "uls"), bread = "expected",
                       moments = if (w == "nt") "structured" else "auto")
          expect_true(nrow(a) > 0L, info = paste(w, release, ew))
          expect_identical(attr(a, "mi_type"), "naive_stage2")
          expect_equal(a$mi, ordinary$mi, tolerance = 1e-8)
          expect_equal(a$mi, b$mi, tolerance = 1e-10, info = paste(w, release, ew))
          expect_equal(a$mi.scaled, b$mi.scaled, tolerance = 1e-10,
                       info = paste(w, release, ew))
        }
      }
    }
  }
})

test_that("two-stage MAR MI and releases retain positive Stage-1 scaling", {
  d <- recipe_data(400L, 20261005L)
  # x1 stays observed: missingness depends on that observed covariate, hence MAR.
  set.seed(20261006L)
  d$x2[runif(nrow(d)) < plogis(-1 + 0.7 * d$x1)] <- NA_real_
  d$x4[runif(nrow(d)) < plogis(-1 - 0.6 * d$x1)] <- NA_real_
  spec <- model_spec("f =~ x1+a*x2+a*x3+x4+x5", meanstructure = TRUE)
  for (w in c("nt", "uls", "dwls", "adf", "dls")) {
    fit <- fit_model(spec, d, estimator = "ML2S", stage2_weight = w, dls_a = 0.3)
    expect_true(fit$converged, info = w)
    for (release in c(FALSE, TRUE)) {
      worker <- if (release) score_tests_robust else modification_indices_robust
      ordinary <- if (release) score_tests(fit, bread = "expected", estimated_weight = FALSE, cov = "model_implied") else modification_indices(fit, bread = "expected", estimated_weight = FALSE, cov = "model_implied")
      fixed <- worker(fit, estimated_weight = FALSE, bread = "expected")
      estimated <- worker(fit, estimated_weight = TRUE, bread = "expected")
      expect_gt(nrow(fixed), 0L)
      expect_identical(attr(ordinary, "mi_type"), "naive_stage2")
      for (result in list(fixed, estimated)) {
        expect_true(all(is.finite(result$mi.scaled)), info = paste(w, release))
        expect_true(all(is.finite(result$scaling.factor) & result$scaling.factor > 0))
        expect_equal(result$mi, ordinary$mi, tolerance = 1e-8)
        expect_equal(result$mi.scaled, result$mi / result$scaling.factor,
                     tolerance = 1e-12)
      }
      if (w %in% c("nt", "uls")) {
        expect_equal(estimated$mi.scaled, fixed$mi.scaled, tolerance = 1e-10)
      }
    }
  }
})

test_that("continuous LS MI and releases preserve means and equality constraints", {
  d <- recipe_data()
  d$x1 <- d$x1 + 0.8
  d$x3 <- d$x3 - 0.4
  for (syntax in c("f =~ x1+x2+x3+x4+x5", "f =~ x1+a*x2+a*x3+x4+x5")) {
    spec <- model_spec(syntax, meanstructure = TRUE)
    fits <- c(list(uls = fit_model(spec, d, estimator = "ULS")), recipe_fits(spec, d))
    for (release in c(FALSE, TRUE)) {
      if (release && !grepl("a\\*", syntax)) next
      worker <- if (release) score_tests_robust else modification_indices_robust
      ordinary_worker <- if (release) score_tests else modification_indices
      for (ew in c(FALSE, TRUE)) {
        results <- lapply(fits, worker, data = d, bread = "expected", estimated_weight = ew)
        for (nm in names(results)) {
          result <- results[[nm]]
          expect_true(fits[[nm]]$converged, info = nm)
          expect_gt(nrow(result), 0L)
          expect_equal(result$mi, ordinary_worker(fits[[nm]], bread = "expected", estimated_weight = FALSE, cov = "model_implied")$mi, tolerance = 1e-8)
          expect_true(all(is.finite(result$scaling.factor) & result$scaling.factor > 0))
          expect_equal(result$mi.scaled, result$mi / result$scaling.factor,
                       tolerance = 1e-12)
        }
        expect_equal(results$dls0$mi.scaled, results$nt$mi.scaled, tolerance = 1e-5)
        expect_equal(results$dls1$mi.scaled, results$adf$mi.scaled, tolerance = 1e-10)
        expect_equal(worker(fits$uls, data = d, estimated_weight = ew, bread = "expected")$mi.scaled,
                     worker(fits$uls, data = d, bread = "expected")$mi.scaled, tolerance = 1e-10)
      }
    }
  }
})

test_that("complete ML MI and equality releases share ordinary and robust coordinates", {
  d <- recipe_data()
  fit <- fit_model("f =~ x1+a*x2+a*x3+x4+x5\nx1 ~~ 0*x2", d,
                   estimator = "ML", meanstructure = TRUE)
  expect_true(fit$converged)
  for (release in c(FALSE, TRUE)) {
    worker <- if (release) score_tests_robust else modification_indices_robust
    ordinary <- if (release) score_tests(fit, bread = "expected", estimated_weight = FALSE, cov = "model_implied") else modification_indices(fit, bread = "expected", estimated_weight = FALSE, cov = "model_implied")
    robust <- worker(fit, data = d, estimated_weight = FALSE, bread = "expected")
    expect_gt(nrow(ordinary), 0L)
    expect_equal(robust$mi, ordinary$mi, tolerance = 1e-8)
    expect_equal(robust$epc, ordinary$epc, tolerance = 1e-8)
    expect_true(all(is.finite(robust$mi.scaled) & robust$scaling.factor > 0))
    if (release) {
      # The R release adapter has no sample-only ML/Gamma_NT path.
      expect_error(worker(fit, cov = "model_implied", estimated_weight = FALSE, bread = "expected"), "require.*fitting data")
    } else {
      normal <- worker(fit, cov = "model_implied", estimated_weight = FALSE, bread = "expected")
      expect_equal(normal$mi.scaled, ordinary$mi, tolerance = 1e-8)
    }
    expect_error(worker(fit, data = d, estimated_weight = TRUE, bread = "expected"), "estimated_weight")
  }
})

test_that("ordinal recipe matrix gates releases and estimated-weight refusals", {
  cont <- recipe_data(700L)
  d <- as.data.frame(lapply(cont[1:4], function(x)
    ordered(cut(x, c(-Inf, -0.5, 0.5, Inf)))))
  spec <- model_spec("f =~ x1+a*x2+a*x3+x4\nx1 ~~ 0*x3", ordered = names(d))
  for (estimator in c("ULS", "DWLS", "WLS", "GLS", "DLS", "supplied")) {
    fit <- if (estimator == "supplied") {
      fit_model(spec, d, estimator = "DWLS", W = diag(seq(0.7, 1.3, length.out = 14L)))
    } else fit_model(spec, d, estimator = estimator, dls_a = 0.3)
    expect_true(fit$converged, info = estimator)
    for (release in c(FALSE, TRUE)) {
      ordinary <- if (release) score_tests(fit, bread = "expected", estimated_weight = FALSE, cov = "model_implied") else modification_indices(fit, bread = "expected", estimated_weight = FALSE, cov = "model_implied")
      worker <- if (release) score_tests_robust else modification_indices_robust
      fixed <- worker(fit, estimated_weight = FALSE, bread = "expected")
      expect_gt(nrow(ordinary), 0L)
      expect_equal(fixed$mi, ordinary$mi, tolerance = 1e-8)
      expect_equal(fixed$epc, ordinary$epc, tolerance = 1e-8)
      expect_true(all(is.finite(fixed$mi.scaled) & fixed$scaling.factor > 0))
      if (estimator %in% c("GLS", "DLS", "supplied")) {
        expect_error(worker(fit, estimated_weight = TRUE, bread = "expected"), "UnsupportedInference")
      } else {
        estimated <- worker(fit, estimated_weight = TRUE, bread = "expected")
        expect_equal(estimated$mi, ordinary$mi, tolerance = 1e-8)
        expect_true(all(is.finite(estimated$mi.scaled) & estimated$scaling.factor > 0))
        if (estimator == "ULS") expect_equal(estimated, fixed, tolerance = 1e-10)
      }
    }
  }
  association <- fit_model(spec, d, estimator = "ML")
  for (worker in list(modification_indices, score_tests, modification_indices_robust,
                      score_tests_robust)) expect_error(worker(association, bread = "expected"), "association")
})

test_that("mixed ordinal MI matrix gates fixed weights and explicit refusals", {
  cont <- recipe_data(700L)
  d <- cont[1:4]
  d[1:2] <- lapply(d[1:2], function(x) ordered(cut(x, c(-Inf, -0.5, 0.5, Inf))))
  spec <- model_spec("f =~ x1+a*x2+a*x3+x4\nx3 ~~ 0*x4", ordered = names(d)[1:2])
  # Core mixed ULS ordinary scores exist, but the R fitting surface refuses it.
  expect_error(fit_model(spec, d, estimator = "ULS"), "ULS is not supported.*mixed")
  for (estimator in c("DWLS", "WLS")) {
    fit <- fit_model(spec, d, estimator = estimator)
    expect_true(fit$converged, info = estimator)
    for (release in c(FALSE, TRUE)) {
      ordinary <- if (release) score_tests(fit, bread = "expected", estimated_weight = FALSE, cov = "model_implied") else modification_indices(fit, bread = "expected", estimated_weight = FALSE, cov = "model_implied")
      worker <- if (release) score_tests_robust else modification_indices_robust
      expect_gt(nrow(ordinary), 0L)
      expect_true(all(is.finite(ordinary$mi)))
      fixed <- worker(fit, estimated_weight = FALSE, bread = "expected")
      expect_equal(fixed$mi, ordinary$mi, tolerance = 1e-8)
      expect_true(all(is.finite(fixed$mi.scaled) & fixed$scaling.factor > 0))
      estimated <- worker(fit, estimated_weight = TRUE, bread = "observed")
      expect_gt(nrow(estimated), 0L)
      expect_true(all(is.finite(estimated$mi.scaled)))
      expect_equal(worker(fit), estimated)
      supplied <- worker(fit, estimated_weight = FALSE, bread = "observed",
                         gamma = fit$mixed_ordinal_stats$NACOV)
      doubled <- worker(fit, estimated_weight = FALSE, bread = "observed",
                        gamma = lapply(fit$mixed_ordinal_stats$NACOV, function(G) 2*G))
      expect_equal(doubled$scaling.factor, 2*supplied$scaling.factor, tolerance = 1e-10)
      expect_equal(doubled$mi, supplied$mi, tolerance = 1e-10)
    }
  }
})


test_that("lab estimated-weight switches default to misspecification-robust weights", {
  ns <- asNamespace("magmaanlab")
  checked <- character()
  for (name in ls(ns, all.names = TRUE)) {
    worker <- get(name, envir = ns)
    if (!is.function(worker)) next
    args <- formals(worker)
    if (!"estimated_weight" %in% names(args)) next
    expect_identical(args$estimated_weight, TRUE, info = name)
    checked <- c(checked, name)
  }
  expect_true("frontier_rbm" %in% checked)
  expect_true("fit_measures_misspec" %in% checked)
})


test_that("mixed estimated-weight MI and releases reach lab defaults across groups and coordinates", {
  d <- recipe_data(500L)[1:4]
  d[1:2] <- lapply(d[1:2], function(x) ordered(cut(x, c(-Inf, -0.5, 0.5, Inf))))
  for (groups in c(1L, 2L)) {
    data <- d
    if (groups == 2L) data$group <- rep(c("A", "B"), each = nrow(d)/2)
    for (parameterization in c("delta", "theta")) {
      spec <- model_spec("f =~ x1+a*x2+a*x3+x4\nx3 ~~ 0*x4",
        ordered = names(d)[1:2], parameterization = parameterization,
        group = if (groups == 2L) "group" else NULL,
        group_labels = if (groups == 2L) c("A", "B") else NULL)
      for (estimator in c("DWLS", "WLS")) {
        fit <- fit_model(spec, data, estimator = estimator)
        for (worker in list(modification_indices, score_tests)) {
          result <- worker(fit)
          expect_gt(nrow(result), 0L)
          expect_true(all(is.finite(result$mi.scaled)))
          expect_equal(result, worker(fit, bread = "observed", estimated_weight = TRUE))
        }
      }
    }
  }
})
