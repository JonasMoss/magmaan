# Pin the behavioral contract rather than silently following installed versions.
.fitting_oracle <- function() {
  skip_if_not_installed("lavaan")
  skip_if(as.character(utils::packageVersion("lavaan")) != "0.7.2",
          "versioned fitting oracle requires lavaan 0.7.2")
}
.fitting_keys <- function(pt) paste(pt$lhs, pt$op, pt$rhs, pt$group, sep = "\r")
.fitting_match <- function(fit, oracle, column) {
  mp <- fit$partable
  lp <- lavaan::parTable(oracle)
  free <- mp$free > 0L
  lp[[column]][match(.fitting_keys(mp[free, ]), .fitting_keys(lp))]
}

test_that("versioned starts and complete ML fits agree with lavaan", {
  .fitting_oracle()
  d <- lavaan::HolzingerSwineford1939
  cases <- list(
    list(model = "f =~ x1+x2+x3+x4", std = FALSE, fixed = FALSE),
    list(model = "f =~ x1+x2+x3+x4", std = TRUE, fixed = FALSE),
    list(model = "x4 ~ x1+x2+x3", std = FALSE, fixed = FALSE),
    list(model = "x4 ~ x1+x2+x3", std = FALSE, fixed = TRUE),
    list(model = "f =~ x1+x2+x3\ng =~ x4+x5+x6\ng ~ f", std = FALSE, fixed = FALSE),
    list(model = "f =~ NA*x1+1*x2+x3+x4", std = FALSE, fixed = FALSE),
    list(model = "f =~ x1\nx2 ~ f", std = FALSE, fixed = FALSE),
    list(model = "f1 =~ x1+x2+x3\nf2 =~ x4+x5+x6\nf3 =~ x7+x8+x9\ng =~ f1+f2+f3", std = FALSE, fixed = FALSE),
    list(model = "x1 ~ x2\nx2 ~ x3", std = FALSE, fixed = FALSE))
  for (case in cases) {
    fit <- suppressWarnings(fit_model(case$model, d, meanstructure = TRUE,
        std_lv = case$std, fixed_x = case$fixed,
        options = list(preset = "lavaan-0.7.2")))
    lv <- suppressWarnings(lavaan::sem(case$model, d, meanstructure = TRUE,
        std.lv = case$std, fixed.x = case$fixed))
    expect_equal(as.numeric(fit$fitting$attempts[[1]]$start),
        as.numeric(.fitting_match(fit, lv, "start")), tolerance = 1e-9)
    expect_equal(fit$converged, lavaan::lavInspect(lv, "converged"))
    expect_equal(fit$verdict$status, if (fit$converged) "passed" else "failed")
    expect_equal(as.numeric(fit$theta), as.numeric(.fitting_match(fit, lv, "est")), tolerance = 1e-5)
    expect_equal(fit$fmin, as.numeric(lavaan::fitMeasures(lv, "fmin")), tolerance = 1e-9)
    expect_false(fit$sample_normalized)
    expect_true(isTRUE(fit$diagnostics$newton_accuracy$checked))
    attempt <- fit$fitting$attempts[[fit$fitting$selected_attempt]]
    expect_true(attempt$raw_status %in% 3:6)
    expect_equal(attempt$gradient_max, max(abs(lv@optim$dx)), tolerance = 1e-6)
  }
})

test_that("grouped fitting retains the same native starts and estimates", {
  .fitting_oracle()
  d <- lavaan::HolzingerSwineford1939
  m <- "f =~ NA*x1+1*x2+x3+x4"
  fit <- fit_model(m, d, groups = "school", meanstructure = TRUE,
      options = list(preset = "lavaan-0.7.2"))
  lv <- lavaan::sem(m, d, group = "school", meanstructure = TRUE)
  expect_equal(as.numeric(fit$fitting$attempts[[1]]$start), as.numeric(.fitting_match(fit, lv, "start")), tolerance = 1e-9)
  expect_equal(as.numeric(fit$theta), as.numeric(.fitting_match(fit, lv, "est")), tolerance = 1e-5)
  expect_equal(fit$converged, lavaan::lavInspect(lv, "converged"))
})

test_that("lavaan retries retain rejection of the original invalid covariance", {
  .fitting_oracle()
  d <- lavaan::HolzingerSwineford1939
  model <- "f =~ x1+x2+x3+x4"
  spec <- model_spec(model, meanstructure = TRUE)
  n <- max(spec$partable$free)
  fit <- suppressWarnings(fit_model(spec, d, control = list(start = rep(0, n)),
      options = list(preset = "lavaan-0.7.2")))
  lv <- suppressWarnings(lavaan::sem(model, d, meanstructure = TRUE, start = rep(0, n)))
  expect_equal(length(fit$fitting$attempts), 4L)
  expect_equal(fit$fitting$selected_attempt, 4L)
  expect_true(fit$fitting$attempts[[3]]$simple_start)
  expect_equal(fit$converged, lavaan::lavInspect(lv, "converged"))
  expect_equal(as.numeric(fit$theta), as.numeric(.fitting_match(fit, lv, "est")), tolerance = 1e-5)
  expect_equal(as.numeric(fit$fitting$attempts[[4]]$optimizer_start), as.numeric(lv@optim$x), tolerance = 1e-12)
  expect_true(is.na(fit$fmin))
  expect_true(all(vapply(fit$fitting$attempts, function(a) !a$accepted && a$iterations == 0L, logical(1))))
  expect_true(all(vapply(fit$fitting$attempts, function(a) is.na(a$fmin), logical(1))))
})

test_that("components resolve separately and refits preserve the setup", {
  .fitting_oracle()
  d <- lavaan::HolzingerSwineford1939
  options <- list(preset = "lavaan-0.7.2", convergence = "newton")
  fit <- fit_model("f =~ x1+x2+x3+x4", d, options = options)
  expect_true(fit$fitting$modified_preset)
  expect_equal(fit$fitting$effective$convergence, "newton")
  expect_equal(fit$verdict$status, fit$diagnostics$verdict$status)
  refit <- getFromNamespace(".route_refit_fun", "magmaanlab")(fit)(fit$model, d)
  expect_equal(refit$fitting$effective, fit$fitting$effective)
  expect_equal(refit$theta, fit$theta, tolerance = 1e-10)
  expect_equal(fit$fitting$attempts[[1]]$controls$port$max_eval, 20000L)
  expect_equal(fit$fitting$attempts[[1]]$controls$port$x_tol, 1.5e-8)
  native <- fit_model("f =~ x1+x2+x3+x4", d,
      options = list(starts = "lavaan-0.7.2", optimizer = "port", convergence = "newton"))
  expect_equal(native$fitting$effective$optimizer, "port")
  expect_true(native$sample_normalized)
})

test_that("fully fixed models retain the selected acceptance convention", {
  .fitting_oracle()
  d <- lavaan::HolzingerSwineford1939
  m <- "x1 ~~ 1*x1\nx2 ~~ 1*x2\nx1 ~~ 0*x2"
  fit <- fit_model(m, d, options = list(preset = "lavaan-0.7.2"))
  lv <- lavaan::sem(m, d, se = "none", test = "none")
  expect_equal(fit$converged, lavaan::lavInspect(lv, "converged"))
  expect_equal(fit$fmin, lv@optim$fx, tolerance = 1e-9)
  expect_length(fit$fitting$attempts, 0L)
  expect_true(is.na(fit$fitting$selected_attempt))
  native <- fit_model(m, d, options = list(preset = "lavaan-0.7.2", convergence = "newton"))
  expect_true(native$converged)
})

test_that("zero variance bounds use lavaan's exact bound-masked gradient", {
  .fitting_oracle()
  s <- matrix(c(1, .7, .7, .7, 1, .3, .7, .3, 1), 3)
  dimnames(s) <- list(paste0("x", 1:3), paste0("x", 1:3))
  m <- "f =~ x1+x2+x3"
  ss <- structure(list(S = list(s), nobs = 200L), class = c("magmaan_data", "list"))
  fit <- getFromNamespace("fit_ml", "magmaanlab")(model_spec(m), ss,
      bounds = "variance", options = list(preset = "lavaan-0.7.2"))
  lv <- lavaan::sem(m, sample.cov = s, sample.nobs = 200,
      sample.cov.rescale = FALSE, bounds = "pos.var")
  expect_equal(as.numeric(fit$theta), as.numeric(.fitting_match(fit, lv, "est")), tolerance = 1e-5)
  expect_equal(fit$fmin, lv@optim$fx, tolerance = 1e-9)
  expect_equal(fit$converged, lavaan::lavInspect(lv, "converged"))
  lp <- lavaan::parTable(lv)
  lp <- lp[lp$free > 0L, ]
  active <- lp$est == lp$lower | lp$est == lp$upper
  expect_true(any(active))
  expect_gt(max(abs(lv@optim$dx)), .001)
  expect_equal(fit$fitting$attempts[[fit$fitting$selected_attempt]]$gradient_max,
      max(abs(lv@optim$dx[!active])), tolerance = 1e-6)
})

test_that("fitting options reject unknown versions, conflicts and unsupported sources", {
  d <- data.frame(x1 = 1:10, x2 = c(2,4,1,5,3,7,9,6,10,8), x3 = c(7,1,3,2,6,4,8,10,9,5))
  m <- "f =~ x1+x2+x3"
  expect_error(fit_model(m, d, options = list(preset = "lavaan")), "supported fitting preset")
  expect_error(fit_model(m, d, options = list(preset = "lavaan-0.7.3")), "supported fitting preset")
  expect_error(fit_model(m, d, options = list(convergence = "strict")), "convergence must be")
  expect_equal(suppressWarnings(fit_model(m, d, options = list(convergence = "default")))$fitting$effective$convergence, "newton")
  expect_error(fit_model(m, d, options = list(convergence = "lavaan-0.7.2")), "requires the PORT")
  expect_error(fit_model(m, d, options = list(wut = "newton")), "unknown fitting option")
  expect_error(fit_model(m, d, options = list(preset = "lavaan-0.7.2"), optimizer = "port"), "conflicts")
  expect_error(fit_model(m, d, options = list(preset = "lavaan-0.7.2"), control = list(start = "fabin3")), "constructor conflicts")
  expect_error(fit_model(m, d, options = list(preset = "lavaan-0.7.2"), psd = TRUE), "ordinary continuous")
  expect_error(fit_model(m, d, options = list(preset = "lavaan-0.7.2"), estimator = "ML2S"), "ordinary continuous")
  expect_error(fit_model(m, d, options = list(preset = "lavaan-0.7.2"), missing = "pairwise"), "ordinary continuous")
  expect_error(fit_model("f =~ x1 + a*x2 + b*x3\na == b*b", d, options = list(preset = "lavaan-0.7.2")), "constraints")
  expect_error(fit_model("f =~ x1 + a*x2 + b*x3\na > b", d,
      options = list(optimizer = "port", convergence = "lavaan-0.7.2")), "constraints")
})

# Records every fit_model() call that a refit makes inside the package.
.record_refits <- function(env = parent.frame()) {
  calls <- new.env()
  calls$args <- list()
  real <- magmaanlab::fit_model
  testthat::local_mocked_bindings(fit_model = function(model, data, ...) {
    calls$args[[length(calls$args) + 1L]] <- list(...)
    real(model, data, ...)
  }, .package = "magmaanlab", .env = env)
  calls
}

# A refit must replay every recorded argument except the ones it overrides
# and the model-structure arguments it rebuilds from the model.
.expect_replayed <- function(refit_args, anchor, overridden = character()) {
  recorded <- anchor$options$route$args
  skip_names <- c(getFromNamespace(".model_structure_args", "magmaanlab"), overridden)
  for (name in setdiff(names(recorded), skip_names))
    expect_identical(refit_args[[name]], recorded[[name]], label = name)
}

test_that("every fit_model() fit records the arguments it was given as its route", {
  d <- lavaan::HolzingerSwineford1939
  fit <- fit_model("f =~ x1+x2+x3+x4", d, optimizer = "port", std_lv = TRUE)
  expect_identical(fit$options$route$fitter, "fit_model")
  expect_setequal(names(fit$options$route$args), "optimizer")
  expect_identical(fit$options$route$args$optimizer, "port")
  # Every formal argument is recordable; omitted ones stay omitted.
  args <- setdiff(names(formals(fit_model)), c("model", "data", "..."))
  route_args <- getFromNamespace(".route_args", "magmaanlab")
  env <- list2env(stats::setNames(as.list(seq_along(args)), args))
  expect_setequal(names(route_args(env, fit_model, supplied = args)), args)
})

test_that("modification-index refits replay every fitting argument", {
  .fitting_oracle()
  d <- lavaan::HolzingerSwineford1939
  anchor <- fit_model("f =~ x1+x2+x3+x4", d, options = list(preset = "lavaan-0.7.2"))
  calls <- .record_refits()
  suppressWarnings(modification_indices_lrt(anchor, d, candidates = "covariances"))
  expect_gt(length(calls$args), 0L)
  for (args in calls$args) {
    .expect_replayed(args, anchor, c("groups", "group_equal", "group_partial", "W"))
    expect_identical(args$options, list(preset = "lavaan-0.7.2"))
  }
  # A fit without options keeps its other choices too, e.g. its optimizer.
  plain <- fit_model("f =~ x1+x2+x3+x4", d, optimizer = "port")
  calls <- .record_refits()
  suppressWarnings(modification_indices_lrt(plain, d, candidates = "covariances"))
  expect_gt(length(calls$args), 0L)
  for (args in calls$args) expect_identical(args$optimizer, "port")
})

test_that("equality-release refits replay every fitting argument", {
  d <- lavaan::HolzingerSwineford1939
  options <- list(starts = "fabin3", optimizer = "port", convergence = "newton")
  anchor <- fit_model("f =~ x1+x2+x3+x4", d, groups = "school",
      group_equal = "loadings", options = options)
  calls <- .record_refits()
  score_tests_lrt(anchor, d, candidates = "loadings")
  expect_gt(length(calls$args), 0L)
  for (args in calls$args) {
    .expect_replayed(args, anchor, c("groups", "group_equal", "group_partial", "W"))
    expect_identical(args$options, options)
  }
})

test_that("likelihood-ratio refits keep the anchor's group order", {
  d <- lavaan::HolzingerSwineford1939
  m <- "visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6"
  declared <- rev(unique(as.character(d$school)))
  anchor <- fit_model(model_spec(m, group = "school", group_labels = declared), d)
  expect_identical(anchor$group_labels, declared)
  # The same groups in the same order, from data whose appearance order is the
  # declared one: every group-specific candidate must give the same refit.
  reordered <- d[order(match(as.character(d$school), declared)), ]
  reference <- fit_model(m, reordered, groups = "school")
  expect_identical(reference$group_labels, declared)
  a <- modification_indices_lrt(anchor, d, candidates = "loadings")
  b <- modification_indices_lrt(reference, reordered, candidates = "loadings")
  key <- function(x) paste(x$lhs, x$op, x$rhs, x$group)
  expect_setequal(key(a), key(b))
  expect_equal(a$lrt[order(key(a))], b$lrt[order(key(b))], tolerance = 1e-6)
})

test_that("case reruns keep the anchor's fitting setup", {
  .fitting_oracle()
  d <- lavaan::HolzingerSwineford1939
  anchor <- fit_model("f =~ x1+x2+x3+x4", d, options = list(preset = "lavaan-0.7.2"))
  reruns <- case_rerun(anchor, d, to_rerun = 1:2)
  expect_true(all(reruns$converged))
  for (refit in reruns$rerun) {
    expect_identical(refit$fitting$requested, anchor$fitting$requested)
    expect_identical(refit$fitting$effective, anchor$fitting$effective)
    .expect_replayed(refit$options$route$args, anchor)
  }
})

test_that("explicit numeric starts are reported as explicit", {
  d <- lavaan::HolzingerSwineford1939
  base <- fit_model("f =~ x1+x2+x3+x4", d)
  fit <- fit_model("f =~ x1+x2+x3+x4", d, control = list(start = base$theta),
      options = list(optimizer = "port"))
  expect_identical(fit$start$method, "explicit")
})

test_that("fitting options reject every unsupported route, including pairwise moments", {
  d <- lavaan::HolzingerSwineford1939
  m <- "f =~ x1+x2+x3+x4"
  lavaan_options <- list(preset = "lavaan-0.7.2")
  expect_error(fit_model(m, d, cluster = "school", options = lavaan_options), "ordinary continuous")
  expect_error(fit_model(m, d, ordered = c("x1", "x2", "x3", "x4"), estimator = "DWLS",
      options = lavaan_options), "ordinary continuous")
  expect_error(fit_model(m, d, estimator = "ML2S", options = lavaan_options), "ordinary continuous")
  expect_error(fit_model(m, d, covariance = "barrier", options = lavaan_options), "ordinary continuous")
  incomplete <- d
  incomplete$x1[1:5] <- NA
  pairwise <- fit_model(m, incomplete, missing = "pairwise")$pairwise_stats
  expect_false(is.null(pairwise))
  expect_error(fit_model(m, pairwise, options = lavaan_options), "ordinary continuous")
  expect_error(fit_model("f =~ x1 + a*x2 + b*x3 + x4\na == b*b", d,
      options = list(starts = "lavaan-0.7.2")), "constraints")
  expect_error(fit_model("f =~ x1+a*x2+b*x3+x4\na > b", d, groups = "school", group_equal = "loadings",
      options = list(starts = "lavaan-0.7.2")), "constraints")
})


.fitting_equality_gradient_at <- function(fit, oracle, attempt) {
  lp <- lavaan::parTable(oracle)
  mp <- fit$partable
  index <- match(.fitting_keys(lp[lp$free > 0L, ]), .fitting_keys(mp[mp$free > 0L, ]))
  model <- lavaan:::lav_model_set_parameters(oracle@Model, as.numeric(fit$theta)[index])
  gradient <- lavaan:::lav_model_grad(lavmodel = model, lavsamplestats = oracle@SampleStats,
      lavdata = oracle@Data, lavcache = oracle@Cache)
  as.numeric(crossprod(model@eq.constraints.K,
      gradient / as.numeric(attempt$parameter_scale)[index]))
}

test_that("linear equality presets match installed lavaan coordinates and verdicts", {
  .fitting_oracle()
  d <- lavaan::HolzingerSwineford1939
  cases <- list(
    list(model = "f =~ x1+a*x2+a*x3+x4"),
    list(model = "f =~ x1+a*x2+b*x3+x4\na == 2*b"),
    list(model = "f =~ x1+a*x2+b*x3+x4\na+b == 1.5"),
    list(model = "f =~ x1+a*x2+b*x3+c*x4+d*x5+e*x6\na == c\nb == e"),
    list(model = "f =~ x1+a*x2+b*x3+c*x4+d*x5+e*x6\nb == e\na == c"),
    list(model = "f =~ x1+x2+x3+x4", equal = "loadings"),
    list(model = "f =~ x1+x2+x3+x4", equal = c("loadings", "intercepts")))
  for (case in cases) {
    ma <- list(model = case$model, data = d, meanstructure = TRUE, fixed_x = FALSE,
               options = list(preset = "lavaan-0.7.2"))
    la <- list(model = case$model, data = d, meanstructure = TRUE, fixed.x = FALSE)
    if (!is.null(case$equal)) {
      ma$groups <- la$group <- "school"
      ma$group_equal <- la$group.equal <- case$equal
    }
    fit <- suppressWarnings(do.call(fit_model, ma))
    lv <- suppressWarnings(do.call(lavaan::sem, la))
    expect_equal(as.numeric(fit$fitting$attempts[[1]]$start),
        as.numeric(.fitting_match(fit, lv, "start")), tolerance = 1e-9)
    k <- lv@Model@eq.constraints.K
    k0 <- lv@Model@eq.constraints.k0
    expected_start <- as.numeric(crossprod(k, lavaan::parTable(lv)$start[lavaan::parTable(lv)$free > 0] - k0))
    expect_equal(as.numeric(fit$fitting$attempts[[1]]$optimizer_start), expected_start, tolerance = 1e-9)
    expect_equal(as.numeric(fit$theta), as.numeric(.fitting_match(fit, lv, "est")), tolerance = 1e-5)
    expect_equal(fit$fmin, as.numeric(lavaan::fitMeasures(lv, "fmin")), tolerance = 1e-9)
    expect_equal(fit$converged, lavaan::lavInspect(lv, "converged"))
    attempt <- fit$fitting$attempts[[fit$fitting$selected_attempt]]
    gradient <- .fitting_equality_gradient_at(fit, lv, attempt)
    expect_lt(abs(attempt$gradient_max - max(abs(gradient))), 1e-9)
    expect_equal(attempt$accepted, attempt$raw_status %in% 3:6 &&
        all(is.finite(gradient)) && max(abs(gradient)) <= .001)
  }
})


test_that("constrained invalid initial covariance is an explicit error", {
  .fitting_oracle()
  d <- lavaan::HolzingerSwineford1939
  model <- "f =~ x1+a*x2+a*x3+x4"
  n <- max(model_spec(model, meanstructure = FALSE)$partable$free)
  expect_error(suppressWarnings(fit_model(model, d, meanstructure = FALSE,
      control = list(start = rep(0, n)), options = list(preset = "lavaan-0.7.2"))),
      "constrained initial model-implied covariance")
  expect_error(suppressWarnings(lavaan::sem(model, d, meanstructure = FALSE,
      start = rep(0, n), se = "none", test = "none")))
})


test_that("unsafe affine standardized retries error while shared-label retries remain valid", {
  .fitting_oracle()
  covariance <- tcrossprod(c(1, .8, .6, .9)) + diag(.7, 4)
  scaled <- function(k) {
    units <- diag(c(k, 1, 1/k, 1))
    s <- units %*% covariance %*% units
    dimnames(s) <- list(paste0("x", 1:4), paste0("x", 1:4))
    structure(list(S = list(s), nobs = 200L), class = c("magmaan_data", "list"))
  }
  fit <- getFromNamespace("fit_ml", "magmaanlab")
  preset <- list(preset = "lavaan-0.7.2")
  affine <- model_spec("f =~ x1+a*x2+b*x3+x4\na-b == 0.000001")
  expect_error(fit(affine, scaled(1000), options = preset),
      "standardized retry is unsupported for nonzero affine constraint RHS")
  ratio <- model_spec("f =~ x1+a*x2+b*x3+x4\na == 2*b")
  expect_error(fit(ratio, scaled(1000), options = preset),
      "parameter scaling changes the equality constraint surface")
  # This fixed unit scaling triggers a rejected original attempt and a valid
  # standardized shared-label retry in the R interface.
  ss <- scaled(200)
  model <- "f =~ x1+a*x2+a*x3+x4"
  actual <- suppressWarnings(fit(model_spec(model, fixed_x = FALSE, meanstructure = FALSE), ss, options = preset))
  oracle <- suppressWarnings(lavaan::sem(model, sample.cov = ss$S[[1]], sample.nobs = 200,
      sample.cov.rescale = FALSE, meanstructure = FALSE, fixed.x = FALSE, se = "none", test = "none"))
  expect_gte(length(actual$fitting$attempts), 2L)
  expect_false(actual$fitting$attempts[[1]]$accepted)
  expect_true(any(vapply(actual$fitting$attempts, function(a) a$standardized, logical(1))))
  attempt <- actual$fitting$attempts[[actual$fitting$selected_attempt]]
  gradient <- .fitting_equality_gradient_at(actual, oracle, attempt)
  expect_lt(abs(attempt$gradient_max - max(abs(gradient))), 1e-9)
  expect_equal(attempt$accepted, attempt$raw_status %in% 3:6 &&
      all(is.finite(gradient)) && max(abs(gradient)) <= .001)
  expect_equal(actual$converged, oracle@optim$converged)
  expect_equal(actual$fmin, as.numeric(oracle@optim$fx), tolerance = 1e-9)
  expect_equal(as.numeric(actual$theta), as.numeric(.fitting_match(actual, oracle, "est")), tolerance = 1e-5)
})

test_that("FIML preset matches live lavaan for MCAR MAR and grouped equalities", {
  .fitting_oracle()
  set.seed(10072)
  d <- lavaan::HolzingerSwineford1939
  for (mar in c(FALSE, TRUE)) for (grouped in c(FALSE, TRUE)) {
    x <- d
    for (j in 2:4) {
      probability <- if (mar) plogis(-1 + .2*(x$x1-mean(x$x1))) else rep(.25,nrow(x))
      x[runif(nrow(x))<probability,paste0("x",j)] <- NA_real_
    }
    model <- "f =~ x1+x2+x3+x4"
    group <- if (grouped) "school" else NULL
    equal <- if (grouped) c("loadings","intercepts") else NULL
    fit <- fit_model(model,x,estimator="FIML",groups=group,group_equal=equal,
      meanstructure=TRUE,fixed_x=FALSE,options=list(preset="lavaan-0.7.2"))
    lv <- lavaan::sem(model,x,missing="ml",group=group,group.equal=equal,
      meanstructure=TRUE,fixed.x=FALSE,se="none",test="none")
    expect_equal(as.numeric(fit$fitting$attempts[[1]]$start),
      as.numeric(.fitting_match(fit,lv,"start")),tolerance=1e-9)
    expect_equal(as.numeric(fit$theta),as.numeric(.fitting_match(fit,lv,"est")),tolerance=1e-5)
    expect_equal(fit$fmin,as.numeric(lv@optim$fx),tolerance=1e-9)
    expect_equal(fit$converged,lavaan::lavInspect(lv,"converged"))
    attempt <- fit$fitting$attempts[[fit$fitting$selected_attempt]]
    expect_equal(attempt$gradient_max,max(abs(lv@optim$dx)),tolerance=1e-6)
    expect_true(fit$diagnostics$newton_accuracy$checked)
    refit <- getFromNamespace(".route_refit_fun","magmaanlab")(fit)(fit$model,x)
    expect_equal(refit$fitting$effective,fit$fitting$effective)
    expect_equal(refit$theta,fit$theta,tolerance=1e-10)
  }
})

test_that("FIML preset keeps unsupported routes explicit", {
  d <- lavaan::HolzingerSwineford1939
  d$x2[seq(1,nrow(d),by=4)] <- NA_real_
  options <- list(preset="lavaan-0.7.2")
  expect_error(fit_model("f =~ x1+a*x2+b*x3+x4\na + b == 1.5",d,
    estimator="FIML",fixed_x=FALSE,options=options),"nonzero affine")
  expect_error(fit_model("f =~ x1+a*x2+b*x3+x4\na == b*b",d,
    estimator="FIML",fixed_x=FALSE,options=options),"nonlinear")
  expect_error(fit_model("f =~ x1+x2+x3+x4",d,estimator="FIML",
    options=options,covariance="psd"),"ordinary continuous")
})
