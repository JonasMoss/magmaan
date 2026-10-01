covariance_composition_data <- function(n = 400L) {
  set.seed(261002L)
  f <- rnorm(n)
  z <- sapply(c(.8, .7, .6, .5), function(l) l*f + sqrt(1-l*l)*rnorm(n))
  d <- as.data.frame(z)
  names(d) <- paste0("x", 1:4)
  d
}

covariance_refit <- function(fit, spec, data) {
  getFromNamespace(".route_refit_fun", "magmaanlab")(fit)(spec, data)
}

test_that("non-mixed moment discrepancies compose both barriers and preserve the fitting metric", {
  d <- covariance_composition_data()
  spec <- model_spec("f =~ x1 + x2 + x3 + x4", meanstructure = FALSE)
  W <- diag(seq(.7, 1.3, length.out = 10L))
  for (method in c("ML", "ULS", "GLS", "WLS")) {
    args <- if (method == "WLS") list(W = W) else list()
    for (target in c("joint", "determinacy")) {
      fit <- do.call(fit_model, c(list(spec, d, estimator = method, covariance = "barrier",
        barrier = list(target = target, weight = .25)), args))
      expect_identical(fit$estimator, method)
      expect_identical(fit$covariance_policy, "barrier")
      expect_identical(fit$composition$model_penalty, target)
      expect_identical(fit$composition$inference, "not_validated")
      expect_true(is.finite(fit$penalty$penalized_fmin))
      expect_equal(fit$penalty$penalized_fmin,
        fit$fmin - fit$penalty$weight/fit$penalty$n_total*fit$penalty$value, tolerance = 1e-10)
      if (method %in% c("ULS", "WLS")) {
        implied <- magmaan_core$model_implied(fit)$sigma[[1L]]
        residual <- (implied - fit$S[[1L]])[lower.tri(implied, diag = TRUE)]
        metric <- if (method == "ULS") diag(length(residual)) else W
        expect_equal(fit$fmin, .5*as.numeric(crossprod(residual, metric %*% residual)), tolerance = 1e-9)
      }
      refit <- covariance_refit(fit, spec, d)
      expect_equal(refit$theta, fit$theta, tolerance = 1e-8)
      expect_error(vcov(fit), "sampling/inference contract")
      expect_error(fit_measures(fit), "sampling/inference contract")
    }
    psd <- do.call(fit_model, c(list(spec, d, estimator = method, covariance = "psd"), args))
    legacy <- do.call(fit_model, c(list(spec, d, estimator = method, psd = TRUE), args))
    expect_equal(psd$theta, legacy$theta, tolerance = 0)
    expect_identical(psd$composition$covariance_domain, "psd")
    expect_equal(covariance_refit(psd, spec, d)$theta, psd$theta, tolerance = 1e-8)
  }
})

test_that("prepared continuous and ordinal fits share covariance policies", {
  continuous <- covariance_composition_data()
  ordinal <- as.data.frame(lapply(continuous, function(x) ordered(cut(x, c(-Inf,-.5,.5,Inf)))))
  for (ordered in c(FALSE, TRUE)) {
    d <- if (ordered) ordinal else continuous
    spec <- model_spec("f =~ x1 + x2 + x3 + x4", ordered = if (ordered) names(d) else NULL,
                       meanstructure = FALSE)
    model <- prepare_model(spec, prototype = if (ordered) d else NULL)
    data <- prepare_data(model, d)
    methods <- if (ordered) c("ML", "ULS", "DWLS", "WLS") else c("ML", "ULS", "GLS")
    for (method in methods) for (policy in c("psd", "barrier")) {
      fit <- fit_model(spec, d, estimator = method, covariance = policy)
      staged <- estimate(model, data, estimator = method, covariance = policy)
      expect_equal(staged$fmin, fit$fmin, tolerance = 1e-6)
      expect_identical(staged$composition, fit$composition)
      expect_identical(staged$covariance_policy, policy)
      expect_equal(covariance_refit(fit, spec, d)$theta, fit$theta, tolerance = 1e-8)
      if (ordered) {
        expect_equal(fit$polychoric, fit$ordinal_stats$R, tolerance = 0)
        expect_equal(fit$thresholds, fit$ordinal_stats$thresholds, tolerance = 0)
        if (method == "DWLS") {
          lean_weight <- prepare_weight(data, "DWLS", full = FALSE)
          lean <- estimate(model, data, estimator = method, weight = lean_weight, covariance = policy)
          expect_equal(lean$fmin, staged$fmin, tolerance = 1e-6)
          expect_equal(nrow(lean$ordinal_stats$NACOV[[1L]]), 0L)
        }
      }
    }
  }
})

test_that("two-stage FIML preserves Stage 1 under every non-mixed covariance policy", {
  d <- covariance_composition_data()
  d$x2[seq(2L, nrow(d), by = 7L)] <- NA_real_
  spec <- model_spec("f =~ x1 + x2 + x3 + x4", meanstructure = TRUE)
  for (weight in c("nt", "uls", "dwls", "adf", "dls")) {
    ordinary <- fit_model(spec, d, estimator = "ML2S", stage2_weight = weight)
    for (policy in c("psd", "barrier")) {
      fit <- fit_model(spec, d, estimator = "ML2S", stage2_weight = weight, covariance = policy)
      expect_equal(fit$stage1, ordinary$stage1, tolerance = 0)
      expect_identical(fit$composition$moment_source, "saturated_fiml")
      expect_identical(fit$composition$covariance_domain, if (policy == "barrier") "barrier_interior" else "psd")
      expect_null(fit$ml2s)
      expect_equal(covariance_refit(fit, spec, d)$theta, fit$theta, tolerance = 1e-7)
    }
  }
  fiml <- fit_model(spec, d, estimator = "FIML", covariance = "barrier")
  model <- prepare_model(spec)
  staged <- estimate(model, prepare_data(model, d, kind = "raw"), estimator = "FIML", covariance = "barrier")
  expect_equal(staged$fmin, fiml$fmin, tolerance = 1e-6)
  expect_identical(fiml$composition$moment_source, "raw_observed")
  expect_error(vcov(fiml), "sampling/inference contract")
})

test_that("pairwise MCAR moments share fitting routes while retaining provenance", {
  d <- covariance_composition_data()
  d$x2[seq(2L, nrow(d), by = 7L)] <- NA_real_
  spec <- model_spec("f =~ x1 + x2 + x3 + x4", meanstructure = FALSE)
  for (policy in c("unrestricted", "psd", "barrier")) {
    fit <- fit_model(spec, d, estimator = "ML", missing = "pairwise", covariance = policy)
    expect_identical(fit$composition$moment_source, "pairwise_mcar")
    expect_true(!is.null(fit$pairwise_stats$n_pair))
    expect_equal(fit$S, fit$pairwise_stats$S, tolerance = 0)
    expect_equal(covariance_refit(fit, spec, d)$theta, fit$theta, tolerance = 1e-8)
    expect_error(vcov(fit), "sampling/inference contract")
  }
  complete <- covariance_composition_data()
  ordinary <- fit_model(spec, complete, estimator = "ML", covariance = "barrier")
  pairwise <- fit_model(spec, complete, estimator = "ML", covariance = "barrier", missing = "pairwise")
  # Pairwise moments use N divisors; df_to_data's default covariance scaling is
  # intentionally different, so verify the supplied pairwise criterion itself.
  explicit <- fit_model(spec, pairwise$pairwise_stats, estimator = "ML", covariance = "barrier")
  expect_equal(pairwise$fmin, explicit$fmin, tolerance = 1e-10)
  expect_identical(explicit$composition$moment_source, "pairwise_mcar")
  expect_true(is.finite(ordinary$fmin))
})

test_that("covariance options reject contradictory or unsupported requests", {
  d <- covariance_composition_data()
  spec <- model_spec("f =~ x1 + x2 + x3 + x4")
  expect_error(fit_model(spec, d, covariance = "barrier", psd = TRUE), "different policies")
  expect_error(fit_model(spec, d, barrier = list()), "require covariance")
  expect_error(fit_model(spec, d, covariance = "barrier", barrier = list(weight = Inf)), "finite")
  expect_error(fit_model(spec, d, covariance = "barrier", barrier = list(unknown = 1)), "target")
  mixed <- d
  mixed$x1 <- ordered(cut(mixed$x1, c(-Inf,-.5,.5,Inf)))
  mixed_spec <- model_spec("f =~ x1 + x2 + x3 + x4", ordered = "x1")
  expect_error(fit_model(mixed_spec, mixed, estimator = "DWLS", covariance = "barrier"), "deferred")
})

test_that("ML and direct FIML barriers preserve units and report the caller-unit objective", {
  d <- covariance_composition_data()
  units <- c(.01, 3, 100, .2)
  spec <- model_spec("f =~ x1 + x2 + x3 + x4", meanstructure = TRUE)
  for (method in c("ML", "FIML")) {
    input <- d
    if (method == "FIML") input$x2[seq(2L, nrow(input), by = 7L)] <- NA_real_
    scaled <- as.data.frame(sweep(as.matrix(input), 2L, units, `*`))
    for (target in c("joint", "determinacy")) {
      control <- list(ftol = 1e-14, gtol = 1e-12)
      fit <- fit_model(spec, input, estimator = method, covariance = "barrier", barrier = list(target = target),
        optimizer = "nlopt-lbfgs", control = control)
      changed <- fit_model(spec, scaled, estimator = method, covariance = "barrier", barrier = list(target = target),
        optimizer = "nlopt-lbfgs", control = control)
      implied <- magmaan_core$model_implied(fit)
      changed_implied <- magmaan_core$model_implied(changed)
      expect_equal(sweep(sweep(changed_implied$sigma[[1L]], 1L, units, `/`), 2L, units, `/`),
        implied$sigma[[1L]], tolerance = 2e-6)
      expect_equal(changed_implied$mu[[1L]]/units, implied$mu[[1L]], tolerance = 2e-6)
      expect_equal(changed$penalty$value, fit$penalty$value, tolerance = 2e-6)
      if (method == "ML") expect_equal(changed$fmin, fit$fmin, tolerance = 1e-8)
      else {
        # Half Gaussian negative log likelihood changes by the observed-column
        # Jacobian, including the different missingness-pattern frequencies.
        jacobian <- sum(colSums(!is.na(input))*log(units))/nrow(input)
        expect_equal(changed$fmin - fit$fmin, jacobian, tolerance = 1e-8)
        expect_equal(changed$penalty$penalized_fmin - fit$penalty$penalized_fmin,
          jacobian, tolerance = 1e-8)
      }
    }
  }
})
