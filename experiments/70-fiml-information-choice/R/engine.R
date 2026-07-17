fiml_generate_complete <- function(n, sigma, distribution) {
  z <- matrix(rnorm(n * ncol(sigma)), nrow = n)
  if (distribution == "t5") {
    z <- z * sqrt((5 - 2) / rchisq(n, df = 5))
  } else if (distribution != "normal") {
    stop("unknown distribution: ", distribution, call. = FALSE)
  }
  x <- z %*% chol(sigma)
  colnames(x) <- colnames(sigma)
  as.data.frame(x, check.names = FALSE)
}

fiml_logistic_intercept <- function(driver, target) {
  objective <- function(intercept)
    mean(plogis(intercept + 0.8 * driver)) - target
  uniroot(objective, interval = c(-20, 20), tol = 1e-12)$root
}

fiml_apply_missingness <- function(data, mechanism) {
  if (mechanism == "complete") return(data)
  n <- nrow(data)
  if (mechanism == "mcar30") {
    for (name in c("y2", "y3", "y5", "y6")) {
      data[[name]][runif(n) < 0.30] <- NA_real_
    }
  } else if (mechanism == "mar30") {
    drivers <- list(y2 = data$y1, y3 = data$y1,
                    y5 = data$y4, y6 = data$y4)
    for (name in names(drivers)) {
      driver <- as.numeric(scale(drivers[[name]]))
      intercept <- fiml_logistic_intercept(driver, 0.30)
      probability <- plogis(intercept + 0.8 * driver)
      data[[name]][runif(n) < probability] <- NA_real_
    }
  } else {
    stop("unknown missingness mechanism: ", mechanism, call. = FALSE)
  }
  data
}

fiml_generate_cell <- function(cell, seed) {
  set.seed(seed)
  sigma <- fiml_population_covariance(cell$crossloading[[1L]])
  complete <- fiml_generate_complete(
    cell$n[[1L]], sigma, cell$distribution[[1L]])
  fiml_apply_missingness(complete, cell$missingness[[1L]])
}

fiml_parameter_rows <- function(fit) {
  pt <- fit$partable
  rows <- lapply(seq_len(nrow(fiml_information_parameters)), function(i) {
    spec <- fiml_information_parameters[i, ]
    if (spec$op == "~~") {
      found <- which(
        pt$op == "~~" &
          ((pt$lhs == spec$lhs & pt$rhs == spec$rhs) |
           (pt$lhs == spec$rhs & pt$rhs == spec$lhs)))
    } else {
      found <- which(
        pt$lhs == spec$lhs & pt$op == spec$op & pt$rhs == spec$rhs)
    }
    if (length(found) != 1L || pt$free[found] < 1L) {
      stop("could not find one free slot for ", spec$parameter,
           call. = FALSE)
    }
    data.frame(
      parameter = spec$parameter,
      target = spec$target,
      free = as.integer(pt$free[found]),
      estimate = fit$theta[pt$free[found]],
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

fiml_min_eigenvalue <- function(information) {
  if (is.null(information) || !is.matrix(information) ||
      any(!is.finite(information))) return(NA_real_)
  min(eigen(
    0.5 * (information + t(information)),
    symmetric = TRUE, only.values = TRUE)$values)
}

fiml_empty_replication <- function(cell, rep, seed, stage, error, warning,
                                   fit_seconds, inference_seconds = NA_real_) {
  data.frame(
    cell_id = cell$cell_id,
    rep = rep,
    seed = seed,
    parameter = fiml_information_parameters$parameter,
    target = fiml_information_parameters$target,
    estimate = NA_real_,
    fit_ok = FALSE,
    comparison_ok = FALSE,
    stage = stage,
    error = error,
    warning = warning,
    fit_seconds = fit_seconds,
    inference_seconds = inference_seconds,
    expected_ok = FALSE,
    observed_h1_ok = FALSE,
    observed_hessian_ok = FALSE,
    min_eig_expected = NA_real_,
    min_eig_observed_h1 = NA_real_,
    min_eig_observed_hessian = NA_real_,
    se_expected_model = NA_real_,
    se_expected_sandwich = NA_real_,
    se_observed_h1_model = NA_real_,
    se_observed_h1_sandwich = NA_real_,
    se_observed_hessian_model = NA_real_,
    se_observed_hessian_sandwich = NA_real_,
    stringsAsFactors = FALSE
  )
}

fiml_one_replication <- function(cell, rep, seed_base) {
  seed <- as.integer(seed_base + cell$cell_id[[1L]] * 100000 + rep)
  data <- fiml_generate_cell(cell, seed)
  warnings <- character()
  started <- proc.time()[["elapsed"]]
  fit <- tryCatch(
    withCallingHandlers(
      magmaan::magmaan(
        fiml_information_model, data = data, estimator = "FIML",
        control = list(max_iter = 2000L, ftol = 1e-10, gtol = 1e-7)
      ),
      warning = function(w) {
        warnings <<- c(warnings, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) e
  )
  fit_seconds <- proc.time()[["elapsed"]] - started
  warning_text <- paste(unique(warnings), collapse = " | ")
  if (inherits(fit, "error")) {
    return(fiml_empty_replication(
      cell, rep, seed, "fit", conditionMessage(fit), warning_text,
      fit_seconds))
  }
  if (!isTRUE(fit$converged)) {
    return(fiml_empty_replication(
      cell, rep, seed, "fit", "fit did not report convergence", warning_text,
      fit_seconds))
  }

  parameters <- tryCatch(fiml_parameter_rows(fit), error = function(e) e)
  if (inherits(parameters, "error")) {
    return(fiml_empty_replication(
      cell, rep, seed, "parameter_map", conditionMessage(parameters),
      warning_text, fit_seconds))
  }

  started <- proc.time()[["elapsed"]]
  comparison <- tryCatch(
    magmaan::magmaan_core$inference_fiml_information_vcov(fit),
    error = function(e) e
  )
  inference_seconds <- proc.time()[["elapsed"]] - started
  if (inherits(comparison, "error")) {
    out <- fiml_empty_replication(
      cell, rep, seed, "information", conditionMessage(comparison),
      warning_text, fit_seconds, inference_seconds)
    out$fit_ok <- TRUE
    out$estimate <- parameters$estimate
    return(out)
  }

  keys <- c("expected", "observed_h1", "observed_hessian")
  ok <- setNames(vapply(keys, function(key)
    isTRUE(comparison[[key]]$ok), logical(1)), keys)
  extract_se <- function(key, kind, free) {
    if (!ok[[key]]) return(rep(NA_real_, length(free)))
    comparison[[key]][[paste0("se_", kind)]][free]
  }
  errors <- vapply(keys, function(key) {
    if (ok[[key]]) "" else as.character(comparison[[key]]$error %||% "failed")
  }, character(1))
  failed <- which(!ok)
  method_error <- if (length(failed)) {
    paste(paste0(keys[failed], ": ", errors[failed]), collapse = " | ")
  } else {
    ""
  }
  free <- parameters$free

  data.frame(
    cell_id = cell$cell_id,
    rep = rep,
    seed = seed,
    parameter = parameters$parameter,
    target = parameters$target,
    estimate = parameters$estimate,
    fit_ok = TRUE,
    comparison_ok = all(ok),
    stage = if (all(ok)) "complete" else "information_inversion",
    error = method_error,
    warning = warning_text,
    fit_seconds = fit_seconds,
    inference_seconds = inference_seconds,
    expected_ok = ok[["expected"]],
    observed_h1_ok = ok[["observed_h1"]],
    observed_hessian_ok = ok[["observed_hessian"]],
    min_eig_expected =
      fiml_min_eigenvalue(comparison$expected$information),
    min_eig_observed_h1 =
      fiml_min_eigenvalue(comparison$observed_h1$information),
    min_eig_observed_hessian =
      fiml_min_eigenvalue(comparison$observed_hessian$information),
    se_expected_model = extract_se("expected", "model", free),
    se_expected_sandwich = extract_se("expected", "sandwich", free),
    se_observed_h1_model = extract_se("observed_h1", "model", free),
    se_observed_h1_sandwich =
      extract_se("observed_h1", "sandwich", free),
    se_observed_hessian_model =
      extract_se("observed_hessian", "model", free),
    se_observed_hessian_sandwich =
      extract_se("observed_hessian", "sandwich", free),
    stringsAsFactors = FALSE
  )
}

fiml_run_chunk <- function(task, grid, seed_base) {
  cell <- grid[grid$cell_id == task$cell_id, , drop = FALSE]
  rows <- lapply(seq.int(task$rep_start, task$rep_end), function(rep) {
    tryCatch(
      fiml_one_replication(cell, rep, seed_base),
      error = function(e) fiml_empty_replication(
        cell, rep, seed_base + cell$cell_id * 100000 + rep,
        "unexpected", conditionMessage(e), "", NA_real_)
    )
  })
  do.call(rbind, rows)
}

fiml_lavaan_parity <- function(grid, seed_base, max_cells = 3L,
                               tolerance = 2e-3) {
  positions <- unique(as.integer(round(seq(
    1L, nrow(grid), length.out = min(max_cells, nrow(grid))))))
  selected <- grid[positions, , drop = FALSE]
  rows <- list()
  out_i <- 0L

  for (i in seq_len(nrow(selected))) {
    cell <- selected[i, , drop = FALSE]
    seed <- as.integer(seed_base + cell$cell_id * 100000 + 1L)
    data <- fiml_generate_cell(cell, seed)
    mag_fit <- magmaan::magmaan(
      fiml_information_model, data = data, estimator = "FIML",
      control = list(max_iter = 4000L, ftol = 1e-12, gtol = 1e-8))
    mag <- magmaan::magmaan_core$inference_fiml_information_vcov(mag_fit)
    mag_parameters <- fiml_parameter_rows(mag_fit)

    for (m in seq_len(nrow(fiml_information_methods))) {
      method <- fiml_information_methods[m, ]
      lav <- lavaan::cfa(
        fiml_information_model, data = data, missing = "fiml",
        meanstructure = TRUE,
        se = method$lavaan_se,
        information = method$lavaan_information,
        observed.information = method$lavaan_observed_information
      )
      lav_pe <- lavaan::parameterEstimates(lav)
      mag_info <- mag[[method$information]]
      mag_se_name <- paste0("se_", method$covariance)
      for (p in seq_len(nrow(fiml_information_parameters))) {
        spec <- fiml_information_parameters[p, ]
        if (spec$op == "~~") {
          found <- which(
            lav_pe$op == "~~" &
              ((lav_pe$lhs == spec$lhs & lav_pe$rhs == spec$rhs) |
               (lav_pe$lhs == spec$rhs & lav_pe$rhs == spec$lhs)))
        } else {
          found <- which(
            lav_pe$lhs == spec$lhs & lav_pe$op == spec$op &
              lav_pe$rhs == spec$rhs)
        }
        stopifnot(length(found) == 1L)
        mag_free <- mag_parameters$free[
          mag_parameters$parameter == spec$parameter]
        mag_value <- if (isTRUE(mag_info$ok))
          mag_info[[mag_se_name]][mag_free] else NA_real_
        lavaan_value <- lav_pe$se[found]
        difference <- mag_value - lavaan_value
        out_i <- out_i + 1L
        rows[[out_i]] <- data.frame(
          cell_id = cell$cell_id,
          distribution = cell$distribution,
          specification = cell$specification,
          missingness = cell$missingness,
          n = cell$n,
          method = method$method,
          parameter = spec$parameter,
          magmaan_se = mag_value,
          lavaan_se = lavaan_value,
          difference = difference,
          abs_difference = abs(difference),
          tolerance = tolerance,
          pass = is.finite(difference) && abs(difference) <= tolerance,
          stringsAsFactors = FALSE
        )
      }
    }
  }
  do.call(rbind, rows)
}
