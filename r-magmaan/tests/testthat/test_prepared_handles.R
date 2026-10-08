expect_prepared_parity <- function(model, data, estimator = "ML", covariance = "unrestricted",
                                   options = NULL) {
  prepared <- magmaan(model, data, estimator, covariance, options = options)
  # Compare the public composition with its previous estimation boundary.
  reference <- prepared
  args <- list(model = model$spec, data = magmaan:::.fit_data(model, data), estimator = estimator)
  policy <- magmaan:::.check_covariance(covariance)
  effective <- if (policy$policy == "barrier" && policy$lambda == 0) "unrestricted" else policy$policy
  args$covariance <- effective
  if (effective == "barrier") args$barrier <- list(target = "joint", weight = policy$lambda)
  opts <- magmaan:::.check_options(options)
  engine <- opts[intersect(c("preset", "optimizer", "convergence"), names(opts))]
  start <- magmaan:::.start_inputs(opts$start, estimator, effective, model$ordered, length(engine) > 0L)
  engine$starts <- start$starts
  if (length(engine)) args$options <- engine
  if (!is.null(start$control)) args$control <- start$control
  reference$lab <- do.call(magmaanlab::fit_model, args)
  reference$inference <- NULL
  reference <- infer(reference)
  for (field in c("theta", "fmin", "converged", "verdict"))
    expect_equal(prepared$lab[[field]], reference$lab[[field]], tolerance = 1e-8, label = field)
  expect_equal(prepared$lab$partable, reference$lab$partable, tolerance = 1e-8)
  expect_equal(prepared$inference$status, reference$inference$status)
  expect_equal(coef(prepared), coef(reference), tolerance = 1e-8)
  if (any(prepared$inference$status$component == "covariance" & prepared$inference$status$available))
    expect_equal(vcov(prepared), vcov(reference), tolerance = 1e-8)
  expect_identical(prepared$lab$options$route$fitter, "fit_model")
  prepared
}

test_that("ordinary estimators and covariance policies preserve fresh-fit results", {
  d <- hs()
  incomplete <- d
  incomplete$x2[seq(1, nrow(d), 7)] <- NA_real_
  ord <- ordinal_hs()
  mixed <- d
  mixed[paste0("x", 1:3)] <- ord[paste0("x", 1:3)]
  cases <- list(list("ML", d, NULL), list("ML", incomplete, NULL),
    list("FIML", incomplete, NULL), list("ML2S", incomplete, NULL),
    list("GLS", d, NULL), list("ULS", d, NULL),
    list("DWLS", ord, paste0("x", 1:6)), list("ULS", ord, paste0("x", 1:6)),
    list("WLS", ord, paste0("x", 1:6)), list("DWLS", mixed, paste0("x", 1:3)))
  for (case in cases) {
    model <- magmaan_model(cfa, prototype = case[[2]], ordered = case[[3]])
    expect_prepared_parity(model, case[[2]], case[[1]])
  }
  for (group in list(NULL, "school")) {
    model <- magmaan_model(cfa, prototype = d, group = group)
    for (covariance in list("psd", barrier(0.25), barrier(0)))
      suppressMessages(expect_prepared_parity(model, d, covariance = covariance))
  }
  for (opts in list(list(preset = "lavaan-0.7.2"), list(start = "fabin3")))
    expect_prepared_parity(magmaan_model(cfa, prototype = d), d, options = opts)
})

test_that("repeated fits reuse structure and serialized models rebuild it", {
  count <- magmaanlab:::prepared_structure_count_impl
  for (data in list(hs(), ordinal_hs())) {
    ordered <- if (is.ordered(data$x1)) paste0("x", 1:6) else NULL
    estimator <- if (length(ordered)) "DWLS" else "ML"
    before <- count()
    model <- magmaan_model(cfa, prototype = data, ordered = ordered)
    expect_s3_class(model$prepared_cache$handle, "magmaan_prepared_model")
    expect_identical(count(), before + 1)
    n <- count()
    fit <- magmaan(model, data, estimator)
    expect_identical(count(), n)
    again <- magmaan(model, data, estimator)
    expect_identical(count(), n)
    expect_equal(again$lab$theta, fit$lab$theta, tolerance = 1e-8)
    path <- tempfile(fileext = ".rds")
    saveRDS(model, path)
    restored <- readRDS(path)
    unlink(path)
    rebuilt <- magmaan(restored, data, estimator)
    expect_equal(count(), n + 1)
    expect_equal(rebuilt$lab$theta, fit$lab$theta, tolerance = 1e-8)
    expect_equal(rebuilt$inference$status, fit$inference$status)
  }
})

test_that("serialized models rebuild prepared handles on PSOCK workers", {
  # Socket access can be disabled by a check sandbox; local rebuilding is
  # covered separately so that it still runs in those environments.
  socket <- tryCatch(suppressWarnings(serverSocket(0)), error = identity)
  if (inherits(socket, "error"))
    skip(paste("PSOCK sockets unavailable:", conditionMessage(socket)))
  close(socket)
  worker <- parallel::makePSOCKcluster(1)
  on.exit(parallel::stopCluster(worker), add = TRUE)
  for (data in list(hs(), ordinal_hs())) {
    ordered <- if (is.ordered(data$x1)) paste0("x", 1:6) else NULL
    estimator <- if (length(ordered)) "DWLS" else "ML"
    model <- magmaan_model(cfa, prototype = data, ordered = ordered)
    fit <- magmaan(model, data, estimator)
    result <- parallel::clusterCall(worker, function(model, data, estimator, libs) {
      .libPaths(libs)
      Sys.setenv(OPENBLAS_NUM_THREADS = 1, OMP_NUM_THREADS = 1, MKL_NUM_THREADS = 1)
      magmaan::magmaan(model, data, estimator)
    }, model, data, estimator, .libPaths())[[1]]
    expect_equal(result$lab$theta, fit$lab$theta, tolerance = 1e-8)
    expect_equal(result$inference$status, fit$inference$status)
    expect_equal(coef(result), coef(fit), tolerance = 1e-8)
    expect_equal(vcov(result), vcov(fit), tolerance = 1e-8)
  }
})

test_that("ordinal DWLS fitting options reuse prepared handles and ML2S retains its fallback", {
  count <- magmaanlab:::prepared_structure_count_impl
  d <- ordinal_hs()
  model <- magmaan_model(cfa, prototype = d, ordered = paste0("x", 1:6))
  n <- count()
  for (opts in list(list(preset = "lavaan-0.7.2"), list(start = "lavaan-0.7.2")))
    expect_prepared_parity(model, d, "DWLS", options = opts)
  expect_identical(count(), n)
  expect_s3_class(model$prepared_cache$handle, "magmaan_prepared_model")
  n <- count()
  model <- magmaan_model(cfa, prototype = hs())
  expect_identical(count(), n + 1)
  expect_prepared_parity(model, hs(), "ML2S")
  expect_identical(count(), n + 1)
  expect_s3_class(model$prepared_cache$handle, "magmaan_prepared_model")
})
