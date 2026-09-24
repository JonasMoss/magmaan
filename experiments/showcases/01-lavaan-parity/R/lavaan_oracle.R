# lavaan oracle for one corpus case: fit with lavaan and return free-parameter
# estimates, the optimizer objective, and the convergence flag. Used only by the
# --collect-lavaan / --run-refit paths; the default and --from-oracle audit
# paths replay a cached oracle and never fit lavaan. Experiment-local copy so
# this experiment does not source a paper.

# Build an unfitted lavaan object for one case and return a fit_once() closure
# that re-optimizes from pre-built slots, so a timed comparison is optimization
# work, not setup. Sample moments come from the magmaan problem, so lavaan and
# magmaan fit exactly the same covariance, mean, and sample size.
.lavaan_build <- function(case, optim_method = "nlminb",
                          check_gradient = FALSE) {
  if (!requireNamespace("lavaan", quietly = TRUE)) {
    stop("The lavaan package is required.", call. = FALSE)
  }

  spec_args <- case$model_spec_args %||% list()
  meanstructure <- isTRUE(spec_args$meanstructure)
  # magmaan calls the ADF weight "ADF"; lavaan spells the same thing "WLS".
  lav_estimator <- if (identical(toupper(case$estimator), "ADF")) "WLS"
                   else case$estimator
  # fixed.x defaults to TRUE to match lavaan's sem() wrapper; the case override
  # (when set) wins.
  fixed_x <- if (!is.null(spec_args$fixed_x)) isTRUE(spec_args$fixed_x) else TRUE
  model_type <- spec_args$model_type %||% "sem"

  problem <- make_problem(case)
  dat <- problem$dat
  multigroup <- length(dat$S) > 1L

  if (multigroup) {
    sample_cov <- dat$S
    nobs <- as.integer(unlist(dat$nobs))
    sample_mean <- if (meanstructure && !is.null(dat$mean)) dat$mean else NULL
    group_label <- spec_args$group_labels
  } else {
    sample_cov <- dat$S[[1L]]
    nobs <- dat$nobs
    if (is.list(nobs) || length(nobs) > 1L) nobs <- nobs[[1L]]
    sample_mean <- if (meanstructure && !is.null(dat$mean)) dat$mean[[1L]] else NULL
    group_label <- NULL
  }
  # lavaan's WLS path needs raw data to compute the Browne-1984 NACOV; it
  # refuses sample.cov alone. corpus_case() already filters ADF cells without
  # raw data, so every WLS call here has case[["data"]] populated.
  use_raw_data <- identical(lav_estimator, "WLS") && !multigroup &&
                  identical(case$data_kind, "raw") &&
                  is.data.frame(case[["data"]])

  partable <- lavaan::lavaanify(
    case$model,
    meanstructure = meanstructure,
    auto = TRUE,
    model.type = model_type,
    fixed.x = fixed_x,
    ngroups = if (multigroup) length(sample_cov) else 1L
  )
  build_args <- list(
    partable,
    estimator = lav_estimator,
    meanstructure = meanstructure,
    model.type = model_type,
    se = "none",
    test = "none",
    baseline = FALSE,
    h1 = FALSE,
    implied = FALSE,
    loglik = FALSE,
    check.gradient = isTRUE(check_gradient),
    check.start = FALSE,
    check.post = FALSE,
    check.vcov = FALSE,
    store.vcov = FALSE,
    optim.method = optim_method,
    do.fit = FALSE
  )
  if (use_raw_data) {
    build_args$data <- case[["data"]]
  } else {
    build_args$sample.cov  <- sample_cov
    build_args$sample.nobs <- nobs
    build_args$sample.mean <- sample_mean
  }
  if (!is.null(group_label)) build_args$group.label <- group_label
  # lavaan::sem() forces model.type = "sem"; growth-model cases must go through
  # lavaan::growth() to keep auto-defaults consistent with the growth parTable.
  builder <- if (identical(model_type, "growth")) lavaan::growth else lavaan::sem
  unfitted <- do.call(builder, build_args)
  slot_options <- unfitted@Options
  slot_options$do.fit <- TRUE

  function() {
    lavaan::lavaan(
      slotOptions     = slot_options,
      slotParTable    = unfitted@ParTable,
      slotSampleStats = unfitted@SampleStats,
      slotData        = unfitted@Data,
      slotModel       = unfitted@Model,
      slotCache       = unfitted@Cache
    )
  }
}

# Fit one case with lavaan; return free-parameter estimates, the optimizer
# objective value, and the convergence flag (no timing).
lavaan_estimates <- function(case, optim_method = "nlminb",
                             check_gradient = FALSE) {
  fit_once <- .lavaan_build(case, optim_method, check_gradient = check_gradient)
  fit <- fit_once()

  pt <- lavaan::parameterTable(fit)
  pt <- pt[pt$free > 0L, , drop = FALSE]
  group <- if ("group" %in% names(pt)) pt$group else 1L
  estimates <- data.frame(
    lhs = pt$lhs, op = pt$op, rhs = pt$rhs, group = group, est = pt$est,
    stringsAsFactors = FALSE
  )

  # The slot-reuse refit path leaves lavoptions$optimizer empty, so lavaan
  # silently SKIPS its own gradient post-check; lavInspect(fit, "converged")
  # then reflects only the optimizer's step-size stopping rule. When the caller
  # asks for the gradient check, apply it ourselves on lavaan's own returned
  # gradient using lavaan's own threshold (optim.dx.tol, default 1e-3) — the
  # same KKT projected-gradient check magmaan's audit_terminal_iterate applies.
  lav_converged <- isTRUE(lavaan::lavInspect(fit, "converged"))
  if (isTRUE(check_gradient) && lav_converged) {
    dx <- tryCatch(lavaan::lavInspect(fit, "gradient"),
                   error = function(e) NULL)
    tol <- fit@Options$optim.dx.tol %||% 1e-3
    if (!is.null(dx) && any(abs(dx) > tol)) {
      par_lower <- fit@ParTable$lower
      par_upper <- fit@ParTable$upper
      x_vec     <- fit@ParTable$est[fit@ParTable$free > 0L]
      bound_at_lo <- if (!is.null(par_lower)) {
        which(par_lower[fit@ParTable$free > 0L] == x_vec)
      } else integer()
      bound_at_up <- if (!is.null(par_upper)) {
        which(par_upper[fit@ParTable$free > 0L] == x_vec)
      } else integer()
      bound_idx <- unique(c(bound_at_lo, bound_at_up))
      non_zero <- setdiff(which(abs(dx) > tol), bound_idx)
      if (length(non_zero) > 0L) lav_converged <- FALSE
    }
  }

  list(
    estimates = estimates,
    objective = lavaan::lavInspect(fit, "optim")$fx,
    converged = lav_converged
  )
}
