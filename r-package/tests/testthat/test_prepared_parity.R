# Exercise the ordinary call vocabulary without importing the sibling package.
.prepared_parity_data <- function(kind) {
  d <- lavaan::HolzingerSwineford1939
  if (kind %in% c("ordinal", "mixed")) {
    for (v in paste0("x", if (kind == "ordinal") 1:6 else 1:3))
      d[[v]] <- ordered(cut(d[[v]], stats::quantile(d[[v]], c(0, 1/3, 2/3, 1)),
                             include.lowest = TRUE, labels = FALSE), levels = 1:3)
  }
  if (kind == "raw") d$x1[seq(1, nrow(d), by = 7)] <- NA_real_
  d
}
.prepared_parity_pair <- function(spec, d, estimator, args = list()) {
  detail <- paste(estimator, args$covariance %||% "unrestricted",
                  length(spec$group_labels), length(spec$ordered))
  before <- prepared_structure_count_impl()
  m <- prepare_model(spec, prototype = d)
  expect_equal(prepared_structure_count_impl(), before + 1)
  data <- prepare_data(m, d, kind = if (estimator %in% c("FIML", "ML2S")) "raw" else NULL,
                       missing = "listwise")
  count <- prepared_structure_count_impl()
  fresh <- suppressWarnings(do.call(fit_model, c(list(model = spec, data = d, estimator = estimator), args)))
  staged <- suppressWarnings(do.call(estimate, c(list(model = m, data = data, estimator = estimator), args)))
  expect_equal(prepared_structure_count_impl(), count)
  columns <- intersect(c("lhs", "op", "rhs", "group", "free", "ustart", "label"), names(fresh$partable))
  expect_equal(staged$partable[columns], fresh$partable[columns], info = detail)
  fixed <- fresh$partable$free == 0L
  expect_equal(staged$partable$est[fixed], fresh$partable$est[fixed], tolerance = 1e-8, info = detail)
  expect_equal(staged$theta, fresh$theta, tolerance = 1e-8, info = detail)
  expect_equal(staged$fmin, fresh$fmin, tolerance = 1e-8, info = detail)
  expect_equal(staged$converged, fresh$converged)
  expect_equal(staged$verdict$status, fresh$verdict$status)
  expect_equal(staged$options$route, fresh$options$route)
  list(model = m, data = data, fresh = fresh, staged = staged)
}

test_that("prepared fits match every ordinary estimator and covariance policy", {
  skip_if_not_installed("lavaan")
  cases <- list(moments = c("ML", "GLS", "ULS"), raw = "FIML",
                ordinal = c("DWLS", "ULS", "WLS"), mixed = c("DWLS", "WLS"))
  for (kind in names(cases)) for (grouped in c(FALSE, TRUE)) {
    d <- .prepared_parity_data(kind)
    ordered <- switch(kind, ordinal = paste0("x", 1:6), mixed = paste0("x", 1:3), NULL)
    spec <- model_spec("visual =~ x1+x2+x3\ntextual =~ x4+x5+x6", meanstructure = TRUE,
                       ordered = ordered, fixed_x = FALSE,
                       group = if (grouped) "school" else "",
                       group_labels = if (grouped) unique(as.character(d$school)) else NULL)
    for (estimator in cases[[kind]]) for (policy in c("unrestricted", "psd", "barrier")) {
      args <- if (policy == "unrestricted") list() else list(covariance = policy)
      if (policy == "barrier") args$barrier <- list(target = "joint", weight = 0.25)
      if (kind == "mixed" && policy == "barrier") {
        m <- prepare_model(spec, prototype = d)
        data <- prepare_data(m, d)
        expect_error(do.call(fit_model, c(list(model = spec, data = d, estimator = estimator), args)), "mixed/polyserial barrier")
        expect_error(do.call(estimate, c(list(model = m, data = data, estimator = estimator), args)), "mixed/polyserial barrier")
      } else {
        pair <- .prepared_parity_pair(spec, d, estimator, args)
        table <- pair$fresh$partable[c("lhs", "op", "rhs", "group", "est")]
        .prepared_parity_pair(spec, d, estimator,
                             c(args, list(control = list(start = table))))
        replay <- suppressWarnings(.route_refit_fun(pair$staged)(pair$staged$model, d))
        expect_equal(replay$theta, pair$fresh$theta, tolerance = 1e-8)
      }
    }
    if (kind == "raw") {
      m <- prepare_model(spec)
      data <- prepare_data(m, d, kind = "raw")
      # The agreed gap: the prepared path has no reusable Stage-1 ML2S handle.
      for (policy in c("unrestricted", "psd", "barrier"))
        expect_error(estimate(m, data, estimator = "ML2S", covariance = policy),
                     class = "magmaan_unsupported_estimator")
    }
  }
})

test_that("prepared fitting preserves starts, presets and structural reuse", {
  skip_if_not_installed("lavaan")
  for (estimator in c("ML", "FIML")) for (grouped in c(FALSE, TRUE)) {
    d <- .prepared_parity_data(if (estimator == "FIML") "raw" else "moments")
    spec <- model_spec("visual =~ x1+x2+x3\ntextual =~ x4+x5+x6", meanstructure = TRUE,
                       fixed_x = FALSE, group = if (grouped) "school" else "",
                       group_labels = if (grouped) unique(as.character(d$school)) else NULL)
    for (options in list(list(preset = "lavaan-0.7.2"),
                         list(optimizer = "nlopt-lbfgs", convergence = "newton", starts = "default")))
      .prepared_parity_pair(spec, d, estimator, list(options = options))
    pair <- .prepared_parity_pair(spec, d, estimator,
        list(control = list(start = if (estimator == "ML") "scaled-fabin" else "fabin3")))
    table <- pair$fresh$partable[c("lhs", "op", "rhs", "group", "est")]
    .prepared_parity_pair(spec, d, estimator, list(control = list(start = table)))
    .prepared_parity_pair(spec, d, estimator, list(control = list(start = table),
        options = list(preset = "lavaan-0.7.2")))
    count <- prepared_structure_count_impl()
    for (i in 1:2) {
      fit <- suppressWarnings(estimate(pair$model, pair$data, estimator = estimator,
          control = list(start = if (estimator == "ML") "scaled-fabin" else "fabin3")))
      expect_equal(fit$theta, pair$staged$theta, tolerance = 1e-8)
    }
    expect_equal(prepared_structure_count_impl(), count)
  }
})


test_that("fit-local starts preserve labeled prepared models across datasets", {
  skip_if_not_installed("lavaan")
  d <- .prepared_parity_data("moments")
  spec <- model_spec("visual =~ x1+a*x2+x3\ntextual =~ x4+x5+x6", meanstructure = TRUE,
                     fixed_x = FALSE)
  pair <- .prepared_parity_pair(spec, d, "ML")
  original <- pair$model$spec$partable
  table <- pair$fresh$partable[c("lhs", "op", "rhs", "group", "est")]
  count <- prepared_structure_count_impl()
  fit <- estimate(pair$model, pair$data, control = list(start = table))
  expect_equal(fit$options$route$fitter, "fit_model")
  expect_equal(pair$model$spec$partable, original)
  d$x1 <- d$x1 + 0.25
  data <- prepare_data(pair$model, d, missing = "listwise")
  staged <- estimate(pair$model, data)
  fresh <- fit_model(spec, d)
  expect_equal(staged$theta, fresh$theta, tolerance = 1e-8)
  expect_equal(prepared_structure_count_impl(), count)
})
