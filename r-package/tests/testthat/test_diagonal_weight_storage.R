diagonal_storage_data <- function() {
  set.seed(114)
  f <- rnorm(400)
  X <- outer(f, c(1, .8, .7, .6)) + matrix(rnorm(1600), 400, 4)
  d <- as.data.frame(X)
  names(d) <- paste0("x", 1:4)
  d$g <- rep(c("a", "b"), each = 200)
  d
}

test_that("continuous diagonal weights own compact entries and round-trip", {
  for (means in c(FALSE, TRUE)) for (grouped in c(FALSE, TRUE)) {
    d <- diagonal_storage_data()
    if (!grouped) d$g <- NULL
    spec <- model_spec("f =~ x1 + x2 + x3 + x4", meanstructure = means,
      group = if (grouped) "g" else NULL,
      group_labels = if (grouped) c("a", "b") else NULL)
    model <- prepare_model(spec)
    data <- prepare_data(model, d)
    q <- 10L + if (means) 4L else 0L
    nb <- if (grouped) 2L else 1L
    entries <- lapply(seq_len(nb), function(b) seq(.7, 1.3, length.out = q) * b)
    matrices <- lapply(entries, diag)
    compact <- prepare_weight(data, "DWLS", W = entries)
    dense <- prepare_weight(data, "DWLS", W = matrices)
    expect_identical(compact$W, entries)
    expect_identical(dense$W, entries)
    # Both the native weight and exported R values are owning copies.
    entries[[1]][1] <- 999
    expect_equal(compact$W[[1]][1], .7)
    fit <- estimate(model, data, weight = compact)
    reference <- estimate(model, data, weight = dense)
    expect_equal(fit$theta, reference$theta, tolerance = 1e-8)
    expect_equal(fit$fmin, reference$fmin, tolerance = 1e-10)
    expect_error(vcov(fit), "caller-supplied weight has no recipe")
    expect_equal(vcov(fit, regime = "sandwich_expected"),
      vcov(reference, regime = "sandwich_expected"), tolerance = 1e-8)
    raw <- fit$raw_data$X
    vector_cov <- magmaan_core$infer_continuous_ls_robust(fit, raw,
      weight = compact$W, bread = "expected", fixed_weight = TRUE)$vcov
    matrix_cov <- magmaan_core$infer_continuous_ls_robust(fit, raw,
      weight = matrices, bread = "expected", fixed_weight = TRUE)$vcov
    expect_equal(vector_cov, matrix_cov, tolerance = 1e-8)
    saved <- unserialize(serialize(compact$W, NULL))
    restored <- prepare_weight(data, "DWLS", W = saved)
    expect_identical(restored$W, compact$W)
    expect_equal(estimate(model, data, weight = restored)$theta, fit$theta, tolerance = 1e-8)
    generated <- prepare_weight(data, "DWLS")
    expect_lt(as.numeric(object.size(generated$W)),
      as.numeric(object.size(lapply(generated$W, diag))))
    expect_true(all(vapply(generated$W, function(w) is.numeric(w) && is.null(dim(w)) && length(w) == q, logical(1))))
    supplied <- fit_model(spec, d, estimator = "DWLS", W = matrices)
    expect_identical(supplied$W, compact$W)
    direct <- fit_model(spec, d, estimator = "DWLS")
    expect_equal(estimate(model, data, weight = generated)$theta, direct$theta, tolerance = 1e-6)
    expect_equal(vcov(estimate(model, data, weight = generated)), vcov(direct), tolerance = 1e-6)
    expect_false(attr(direct$W, "cancels_measurement_units"))
    for (b in seq_len(nb)) expect_identical(direct$W[[b]], generated$W[[b]])
  }
})

test_that("full weights retain off-diagonals and compact weights validate", {
  d <- diagonal_storage_data()[, 1:4]
  model <- prepare_model(model_spec("f =~ x1 + x2 + x3 + x4", meanstructure = FALSE))
  data <- prepare_data(model, d)
  W <- diag(10) + matrix(.01, 10, 10)
  full <- prepare_weight(data, "WLS", W = W)
  expect_true(is.matrix(full$W[[1]]))
  expect_identical(full$W[[1]], W)
  tiny <- diag(10)
  tiny[1, 2] <- tiny[2, 1] <- 1e-20
  expect_true(is.matrix(prepare_weight(data, "WLS", W = tiny)$W[[1]]))
  bare <- prepare_weight(data, "DWLS", W = rep(1, 10))
  expect_identical(bare$W[[1]], rep(1, 10))
  expect_error(prepare_weight(data, "DWLS", W = W), "diagonal")
  expect_error(prepare_weight(data, "DWLS", W = rep(-1, 10)), "non-negative")
  expect_error(prepare_weight(data, "DWLS", W = rep(NA_real_, 10)), "finite")
  expect_error(prepare_weight(data, "DWLS", W = rep(0, 10)), "positive definite")
  expect_error(prepare_weight(data, "DWLS", W = rep(1, 9)), "dimensions")
})
