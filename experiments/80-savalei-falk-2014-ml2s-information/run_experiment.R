#!/usr/bin/env Rscript

.support_helpers <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  script <- if (length(file_arg)) {
    normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE)
  } else normalizePath("run_experiment.R", mustWork = FALSE)
  file.path(dirname(dirname(script)), "_support", "R", "helpers.R")
}
source(.support_helpers())
rm(.support_helpers)
set_single_threaded_math()
require_pkg("magmaan")
suppressPackageStartupMessages(library(magmaan))

usage <- function() cat(
  "Usage: Rscript run_experiment.R [--smoke|--pilot|--replication|--fingerprint] [options]\n\n",
  "Identify the matrix-estimation conventions in the Savalei-Falk (2014)\n",
  "robust FIML and two-stage tests. One fit of each method per generated\n",
  "sample is reused for all canonical FIML residual/sandwich choices and\n",
  "all ML2S Stage-1 bread/meat by Stage-2 H choices. The named Mplus-style\n",
  "Yuan-Bentler MLR trace-difference is evaluated separately.\n\n",
  "Profiles:\n",
  "  --smoke        4 design cells x 4 reps (default)\n",
  "  --pilot        4 design cells x 200 reps\n",
  "  --replication  4 design cells x 1,000 reps\n\n",
  "  --fingerprint  6 MCAR cells x 1,000 reps (N=200/400/600, k=7/15)\n\n",
  "Options:\n",
  "  --reps N --cores N --seed-base N --results-dir PATH --resume\n",
  sep = "")

parse_args <- function(args) {
  out <- list(profile = "smoke", reps = NULL,
              cores = max(1L, min(8L, parallel::detectCores() - 1L)),
              seed_base = 20140280, results_dir = NULL, resume = FALSE)
  i <- 1L
  take <- function() {
    i <<- i + 1L
    if (i > length(args)) stop("missing option value", call. = FALSE)
    args[[i]]
  }
  while (i <= length(args)) {
    a <- args[[i]]
    if (a %in% c("-h", "--help")) {
      usage()
      quit(save = "no", status = 0L)
    } else if (a == "--smoke") out$profile <- "smoke"
    else if (a == "--pilot") out$profile <- "pilot"
    else if (a == "--replication") out$profile <- "replication"
    else if (a == "--fingerprint") out$profile <- "fingerprint"
    else if (a == "--reps") out$reps <- as.integer(take())
    else if (a == "--cores") out$cores <- as.integer(take())
    else if (a == "--seed-base") out$seed_base <- as.numeric(take())
    else if (a == "--results-dir") out$results_dir <- take()
    else if (a == "--resume") out$resume <- TRUE
    else stop("unknown argument: ", a, call. = FALSE)
    i <- i + 1L
  }
  if (is.null(out$reps)) {
    out$reps <- switch(out$profile, smoke = 4L, pilot = 200L,
                       replication = 1000L, fingerprint = 1000L)
  }
  if (is.na(out$reps) || out$reps < 1L ||
      is.na(out$cores) || out$cores < 1L ||
      !is.finite(out$seed_base) || out$seed_base < 1) {
    stop("reps, cores, and seed-base must be positive", call. = FALSE)
  }
  out
}

model_2_population <- function() {
  lambda <- c(
    0.743, 0.252, 0.604, 0.540, 0.201, 0.578, 0.729,
    0.536, 0.528, 0.692, 0.694, 0.509, 0.698, 0.562,
    0.426, 0.530, 0.637, 0.755, 0.564, 0.647, 0.825)
  phi <- matrix(c(
    1, 0.446, -0.124,
    0.446, 1, -0.085,
    -0.124, -0.085, 1), 3L, 3L, byrow = TRUE)
  Lambda <- matrix(0, 21L, 3L)
  Lambda[cbind(seq_len(21L), rep(seq_len(3L), each = 7L))] <- lambda
  Sigma <- Lambda %*% phi %*% t(Lambda) + diag(1 - lambda^2)
  dimnames(Sigma) <- list(paste0("V", seq_len(21L)),
                          paste0("V", seq_len(21L)))
  list(
    Sigma = Sigma,
    syntax = paste(
      "F1 =~ V1 + V2 + V3 + V4 + V5 + V6 + V7",
      "F2 =~ V8 + V9 + V10 + V11 + V12 + V13 + V14",
      "F3 =~ V15 + V16 + V17 + V18 + V19 + V20 + V21",
      sep = "\n"))
}

mp_missing_sets <- function() {
  list(
    list(condition = 1L, delete = c(2L, 8L, 9L)),
    list(condition = 3L, delete = 4L),
    list(condition = 5L, delete = c(6L, 7L)),
    list(condition = 15L, delete = c(13L, 14L, 16L)),
    list(condition = 17L, delete = 18L),
    list(condition = 19L, delete = c(20L, 21L)))
}

apply_mp_mar_l <- function(X, proportion, seed) {
  sets <- mp_missing_sets()
  target <- as.integer(round(nrow(X) * proportion))
  set.seed(seed)
  eligible_counts <- integer(length(sets))
  for (j in seq_along(sets)) {
    eligible <- which(X[, sets[[j]]$condition] > 0)
    eligible_counts[[j]] <- length(eligible)
    if (length(eligible) < target) {
      stop("fewer MAR-L eligible rows than the deletion target", call. = FALSE)
    }
    selected <- sample(eligible, target, replace = FALSE)
    X[selected, sets[[j]]$delete] <- NA_real_
  }
  list(
    data = X,
    eligible_min = min(eligible_counts),
    eligible_max = max(eligible_counts),
    patterns = nrow(unique(is.na(X))),
    missing_rate = mean(is.na(X)))
}

apply_mp_mcar <- function(X, proportion, seed) {
  sets <- mp_missing_sets()
  target <- as.integer(round(nrow(X) * proportion))
  set.seed(seed)
  for (set in sets) {
    selected <- sample.int(nrow(X), target, replace = FALSE)
    X[selected, set$delete] <- NA_real_
  }
  list(
    data = X,
    eligible_min = NA_integer_,
    eligible_max = NA_integer_,
    patterns = nrow(unique(is.na(X))),
    missing_rate = mean(is.na(X)))
}

mp_mar_l_is_feasible <- function(X, proportion = 0.30) {
  conditioning <- c(1L, 3L, 5L, 15L, 17L, 19L)
  target <- as.integer(round(nrow(X) * proportion))
  all(colSums(X[, conditioning, drop = FALSE] > 0) >= target)
}

fleishman_equations <- function(x, skewness, excess_kurtosis) {
  b <- x[[1L]]
  c0 <- x[[2L]]
  d <- x[[3L]]
  c(
    b^2 + 6 * b * d + 2 * c0^2 + 15 * d^2 - 1,
    2 * c0 * (b^2 + 24 * b * d + 105 * d^2 + 2) - skewness,
    24 * (b * d + c0^2 * (1 + b^2 + 28 * b * d) +
            d^2 * (12 + 48 * b * d + 141 * c0^2 + 225 * d^2)) -
      excess_kurtosis)
}

fleishman_jacobian <- function(x) {
  b <- x[[1L]]
  c0 <- x[[2L]]
  d <- x[[3L]]
  g <- b^2 + 24 * b * d + 105 * d^2 + 2
  g2 <- 12 + 48 * b * d + 141 * c0^2 + 225 * d^2
  rbind(
    c(2 * b + 6 * d, 4 * c0, 6 * b + 30 * d),
    c(2 * c0 * (2 * b + 24 * d), 2 * g,
      2 * c0 * (24 * b + 210 * d)),
    24 * c(
      d + c0^2 * (2 * b + 28 * d) + 48 * d^3,
      2 * c0 * (1 + b^2 + 28 * b * d) + 282 * c0 * d^2,
      b + 28 * b * c0^2 + 2 * d * g2 + d^2 * (48 * b + 450 * d)))
}

solve_fleishman_root <- function(start, skewness, excess_kurtosis) {
  x <- start
  for (iter in seq_len(100L)) {
    residual <- fleishman_equations(x, skewness, excess_kurtosis)
    if (max(abs(residual)) < 1e-10) return(x)
    step <- tryCatch(solve(fleishman_jacobian(x), residual), error = identity)
    if (inherits(step, "error") || any(!is.finite(step))) return(NULL)
    old <- sum(residual^2)
    scale <- 1
    accepted <- FALSE
    for (line_search in seq_len(30L)) {
      candidate <- x - scale * step
      candidate_error <- sum(fleishman_equations(
        candidate, skewness, excess_kurtosis)^2)
      if (is.finite(candidate_error) && candidate_error < old) {
        x <- candidate
        accepted <- TRUE
        break
      }
      scale <- scale / 2
    }
    if (!accepted) return(NULL)
  }
  if (max(abs(fleishman_equations(x, skewness, excess_kurtosis))) < 1e-8)
    x else NULL
}

vm_root_covariance_audit <- function(target, b, c0, d) {
  linear <- (b + 3 * d)^2
  quadratic <- 2 * c0^2
  cubic <- 6 * d^2
  latent <- diag(nrow(target))
  multiple <- 0L
  for (i in seq_len(nrow(target))) {
    if (i == 1L) next
    for (j in seq_len(i - 1L)) {
      roots <- polyroot(c(-target[i, j], linear, quadratic, cubic))
      valid <- Re(roots)[abs(Im(roots)) < 1e-8 & abs(Re(roots)) < 0.999]
      if (!length(valid)) {
        return(list(solvable = FALSE, multiple = multiple,
                    min_eigenvalue = NA_real_))
      }
      if (length(valid) > 1L) multiple <- multiple + 1L
      same_sign <- valid[valid * target[i, j] >= 0]
      rho <- (same_sign %||% valid)[which.min(abs(same_sign %||% valid))]
      latent[i, j] <- latent[j, i] <- rho
    }
  }
  list(solvable = TRUE, multiple = multiple,
       min_eigenvalue = min(eigen(latent, symmetric = TRUE,
                                  only.values = TRUE)$values))
}

enumerate_vm_roots <- function(target, skewness, excess_kurtosis,
                               selected) {
  starts <- as.matrix(expand.grid(
    b = seq(-2, 2, length.out = 9L),
    c = seq(-1, 1, length.out = 9L),
    d = seq(-0.4, 0.4, length.out = 9L)))
  roots <- list()
  for (i in seq_len(nrow(starts))) {
    root <- solve_fleishman_root(starts[i, ], skewness, excess_kurtosis)
    if (!is.null(root) && (!length(roots) || all(vapply(
      roots, function(old) max(abs(root - old)) > 1e-6, logical(1))))) {
      roots[[length(roots) + 1L]] <- root
    }
  }
  if (length(roots) != 4L) {
    stop("expected four Fleishman roots, found ", length(roots),
         " for skewness=", skewness, ", kurtosis=", excess_kurtosis,
         call. = FALSE)
  }
  coef <- do.call(rbind, roots)
  colnames(coef) <- c("b", "c", "d")
  derivative_extremum <- coef[, "b"] -
    coef[, "c"]^2 / (3 * coef[, "d"])
  covariance_key <- sprintf("%.8f/%.8f/%.8f",
                            (coef[, "b"] + 3 * coef[, "d"])^2,
                            2 * coef[, "c"]^2, 6 * coef[, "d"]^2)
  selected_row <- which.min(rowSums((coef - matrix(
    selected[c("b", "c", "d")], nrow(coef), 3L, byrow = TRUE))^2))
  covariance_audit <- lapply(seq_len(nrow(coef)), function(i)
    vm_root_covariance_audit(target, coef[i, "b"], coef[i, "c"],
                             coef[i, "d"]))
  out <- data.frame(
    skewness = skewness,
    excess_kurtosis = excess_kurtosis,
    a = -coef[, "c"], b = coef[, "b"], c = coef[, "c"], d = coef[, "d"],
    monotone_increasing = coef[, "d"] > 0 & derivative_extremum >= -1e-8,
    monotone_decreasing = coef[, "d"] < 0 & derivative_extremum <= 1e-8,
    selected = seq_len(nrow(coef)) == selected_row,
    same_joint_law_as_selected = covariance_key == covariance_key[selected_row],
    target_correlations_solvable = vapply(
      covariance_audit, `[[`, logical(1), "solvable"),
    intermediate_min_eigenvalue = vapply(
      covariance_audit, `[[`, numeric(1), "min_eigenvalue"),
    stringsAsFactors = FALSE)
  out[order(!out$selected, !out$same_joint_law_as_selected,
            !out$monotone_increasing, out$b), ]
}

add_statistic_conventions <- function(rows, n) {
  rows$sample_size <- n
  rows$chisq_scaled_n_minus_1 <- ifelse(
    rows$method == "ml2s", rows$chisq_scaled * (n - 1) / n, NA_real_)
  rows$p_value_n_minus_1 <- stats::pchisq(
    rows$chisq_scaled_n_minus_1, rows$df, lower.tail = FALSE)
  rows$reject_n_minus_1 <- rows$p_value_n_minus_1 < 0.05
  rows$matrix_audit_ok <- ifelse(
    rows$method == "ml2s",
    is.finite(rows$saturated_identity_error) &
      rows$saturated_identity_error < 1e-7 &
      is.finite(rows$default_scale_error) & rows$default_scale_error < 1e-7,
    !is.na(rows$information))
  rows
}

fit_one <- function(X, model, rep_id, cell_id, mechanism, skewness, kurtosis,
                    n, seed_base) {
  failure <- function(method, message, missing_rate = NA_real_,
                      patterns = NA_integer_, eligible_min = NA_integer_,
                      eligible_max = NA_integer_, converged = FALSE) data.frame(
    rep = rep_id, cell_id = cell_id, mechanism = mechanism,
    skewness = skewness, kurtosis = kurtosis, method = method,
    information = NA_character_, residual_information = NA_character_,
    omega_bread_point = NA_character_, omega_bread_kind = NA_character_,
    omega_meat_point = NA_character_, stage2_information = NA_character_,
    chisq = NA_real_, chisq_scaled = NA_real_, df = NA_integer_,
    scaling_factor = NA_real_, trace_ugamma_h1 = NA_real_,
    trace_ugamma_h0 = NA_real_, p_value = NA_real_, reject = NA,
    sample_size = n, chisq_scaled_n_minus_1 = NA_real_,
    p_value_n_minus_1 = NA_real_, reject_n_minus_1 = NA,
    matrix_audit_ok = FALSE,
    min_omega_bread_eigenvalue = NA_real_,
    omega_bread_negative_eigenvalues = NA_integer_,
    min_stage2_information_eigenvalue = NA_real_,
    stage2_information_negative_eigenvalues = NA_integer_,
    min_projection_eigenvalue = NA_real_,
    projection_negative_eigenvalues = NA_integer_,
    projection_rank = NA_integer_, delta_rank = NA_integer_,
    saturated_identity_error = NA_real_, stage1_bread_difference = NA_real_,
    default_scale_error = NA_real_, missing_rate = missing_rate,
    patterns = patterns, eligible_min = eligible_min,
    eligible_max = eligible_max, converged = converged,
    admissible = NA, error = message, stringsAsFactors = FALSE)

  missing_seed <- seed_base + kurtosis * 100000 + rep_id +
    (n - 200) * 10000 +
    if (mechanism == "MCAR") 70000000 else 0
  miss <- tryCatch(switch(
    mechanism,
    `MAR-L` = apply_mp_mar_l(X, 0.30, missing_seed),
    MCAR = apply_mp_mcar(X, 0.30, missing_seed),
    stop("unsupported missingness mechanism: ", mechanism, call. = FALSE)),
    error = identity)
  if (inherits(miss, "error")) {
    return(do.call(rbind, lapply(c("ml2s", "fiml", "fiml_mlr"), failure,
                                message = conditionMessage(miss))))
  }
  dat <- as.data.frame(miss$data)
  colnames(dat) <- paste0("V", seq_len(ncol(dat)))
  fits <- lapply(c(ml2s = "ML2S", fiml = "FIML"), function(estimator)
    tryCatch(magmaan::magmaan(
      model, dat, estimator = estimator, se = "none", test = "none"),
      error = identity))
  if (any(vapply(fits, inherits, logical(1), what = "error"))) {
    return(do.call(rbind, lapply(c("ml2s", "fiml", "fiml_mlr"), function(method) {
      fit <- fits[[if (method == "fiml_mlr") "fiml" else method]]
      failure(method,
              if (inherits(fit, "error")) conditionMessage(fit)
              else "the companion estimator failed",
              miss$missing_rate, miss$patterns, miss$eligible_min,
              miss$eligible_max, !inherits(fit, "error") &&
                isTRUE(fit$converged))
    })))
  }
  audits <- list(
    ml2s = tryCatch(
      magmaan::magmaan_core$frontier_ml2s_information_choices(fits$ml2s),
      error = identity),
    fiml = tryCatch(
      magmaan::magmaan_core$frontier_fiml_information_choices(fits$fiml),
      error = identity),
    fiml_mlr = tryCatch(
      magmaan::magmaan_core$estimate_fiml_robust_mlr(fits$fiml),
      error = identity))
  if (any(vapply(audits, inherits, logical(1), what = "error"))) {
    return(do.call(rbind, lapply(names(audits), function(method)
      failure(if (method == "fiml_mlr") "fiml_mlr" else method,
              if (inherits(audits[[method]], "error"))
                conditionMessage(audits[[method]])
              else "the companion matrix audit failed",
              miss$missing_rate, miss$patterns, miss$eligible_min,
              miss$eligible_max,
              isTRUE(fits[[if (method == "fiml_mlr") "fiml" else method]]
                     $converged)))))
  }

  ml2s <- audits$ml2s$choices
  ml2s$method <- "ml2s"
  ml2s$residual_information <- ml2s$stage2_information
  ml2s$omega_bread_point <- ml2s$stage1_bread_point
  ml2s$omega_bread_kind <- ml2s$stage1_bread_kind
  ml2s$omega_meat_point <- ml2s$stage1_meat_point
  ml2s$min_omega_bread_eigenvalue <-
    ml2s$min_stage1_information_eigenvalue
  ml2s$omega_bread_negative_eigenvalues <-
    ml2s$stage1_information_negative_eigenvalues
  ml2s$min_stage2_information_eigenvalue <-
    ml2s$min_information_eigenvalue
  ml2s$stage2_information_negative_eigenvalues <-
    ml2s$information_negative_eigenvalues
  ml2s$min_projection_eigenvalue <- ml2s$min_projector_eigenvalue
  ml2s$projection_negative_eigenvalues <-
    ml2s$projector_negative_eigenvalues
  ml2s$projection_rank <- ml2s$projector_rank
  ml2s$saturated_identity_error <-
    audits$ml2s$saturated_expected_observed_max_abs
  ml2s$stage1_bread_difference <-
    audits$ml2s$stage1_expected_observed_max_abs
  ml2s$trace_ugamma_h1 <- NA_real_
  ml2s$trace_ugamma_h0 <- NA_real_
  default_row <- ml2s$omega_bread_point == "saturated" &
    ml2s$omega_bread_kind == "observed" &
    ml2s$omega_meat_point == "saturated" &
    ml2s$stage2_information == "saturated_expected"
  ml2s$default_scale_error <- if (sum(default_row) == 1L) {
    abs(ml2s$scaling_factor[default_row] - fits$ml2s$scaling_factor)
  } else NA_real_

  fiml <- audits$fiml$choices
  fiml$method <- "fiml"
  fiml$stage2_information <- NA_character_
  fiml$min_projection_eigenvalue <-
    fiml$min_residual_information_eigenvalue
  fiml$projection_negative_eigenvalues <-
    fiml$residual_information_negative_eigenvalues
  fiml$projection_rank <- fiml$residual_rank
  fiml$min_stage2_information_eigenvalue <- NA_real_
  fiml$stage2_information_negative_eigenvalues <- NA_integer_
  fiml$saturated_identity_error <- NA_real_
  fiml$stage1_bread_difference <- NA_real_
  fiml$default_scale_error <- NA_real_
  fiml$trace_ugamma_h1 <- NA_real_
  fiml$trace_ugamma_h0 <- NA_real_

  fiml_mlr <- data.frame(
    method = "fiml_mlr",
    information = "yuan_bentler_mplus",
    residual_information = "trace_difference_h1_minus_h0",
    omega_bread_point = NA_character_,
    omega_bread_kind = NA_character_,
    omega_meat_point = NA_character_,
    stage2_information = NA_character_,
    trace_ugamma = audits$fiml_mlr$trace_ugamma,
    trace_ugamma_h1 = audits$fiml_mlr$trace_ugamma_h1,
    trace_ugamma_h0 = audits$fiml_mlr$trace_ugamma_h0,
    scaling_factor = audits$fiml_mlr$scaling_factor,
    chisq_scaled = audits$fiml_mlr$chisq_scaled,
    min_omega_bread_eigenvalue = NA_real_,
    omega_bread_negative_eigenvalues = NA_integer_,
    min_stage2_information_eigenvalue = NA_real_,
    stage2_information_negative_eigenvalues = NA_integer_,
    min_projection_eigenvalue = NA_real_,
    projection_negative_eigenvalues = NA_integer_,
    projection_rank = NA_integer_,
    saturated_identity_error = NA_real_,
    stage1_bread_difference = NA_real_,
    default_scale_error = NA_real_,
    stringsAsFactors = FALSE)

  matrix_cols <- c(
    "method", "information", "residual_information",
    "omega_bread_point", "omega_bread_kind", "omega_meat_point",
    "stage2_information", "trace_ugamma", "scaling_factor", "chisq_scaled",
    "trace_ugamma_h1", "trace_ugamma_h0",
    "min_omega_bread_eigenvalue", "omega_bread_negative_eigenvalues",
    "min_stage2_information_eigenvalue",
    "stage2_information_negative_eigenvalues", "min_projection_eigenvalue",
    "projection_negative_eigenvalues", "projection_rank",
    "saturated_identity_error", "stage1_bread_difference",
    "default_scale_error")
  rows <- rbind(ml2s[, matrix_cols], fiml[, matrix_cols],
                fiml_mlr[, matrix_cols])
  method_audit <- list(
    ml2s = audits$ml2s,
    fiml = audits$fiml,
    fiml_mlr = list(chisq = audits$fiml_mlr$chisq,
                    df = audits$fiml_mlr$df,
                    delta_rank = audits$fiml$delta_rank))
  rows$rep <- rep_id
  rows$cell_id <- cell_id
  rows$mechanism <- mechanism
  rows$skewness <- skewness
  rows$kurtosis <- kurtosis
  rows$chisq <- vapply(rows$method, function(m) method_audit[[m]]$chisq,
                       numeric(1))
  rows$df <- vapply(rows$method,
                    function(m) as.integer(method_audit[[m]]$df), integer(1))
  rows$p_value <- stats::pchisq(rows$chisq_scaled, rows$df,
                                lower.tail = FALSE)
  rows$reject <- rows$p_value < 0.05
  rows$delta_rank <- vapply(rows$method,
                            function(m) as.integer(method_audit[[m]]$delta_rank),
                            integer(1))
  rows$missing_rate <- miss$missing_rate
  rows$patterns <- miss$patterns
  rows$eligible_min <- miss$eligible_min
  rows$eligible_max <- miss$eligible_max
  rows$converged <- vapply(rows$method, function(m)
    isTRUE(fits[[if (m == "fiml_mlr") "fiml" else m]]$converged), logical(1))
  rows$admissible <- vapply(rows$method, function(m)
    fits[[if (m == "fiml_mlr") "fiml" else m]]$diagnostics$admissibility
      $admissible %||% NA, logical(1))
  rows$error <- ""
  rows <- add_statistic_conventions(rows, n)
  rows[, names(failure("ml2s", ""))]
}

summarize_results <- function(raw) {
  ok <- !is.na(raw$information) & is.finite(raw$p_value) &
    raw$matrix_audit_ok
  x <- raw[ok, , drop = FALSE]
  if (!nrow(x)) return(data.frame())
  keys <- interaction(x$cell_id, x$method, x$information, drop = TRUE)
  parts <- split(x, keys)
  do.call(rbind, lapply(parts, function(z) data.frame(
    cell_id = z$cell_id[[1L]],
    mechanism = z$mechanism[[1L]],
    skewness = z$skewness[[1L]],
    kurtosis = z$kurtosis[[1L]],
    method = z$method[[1L]],
    information = z$information[[1L]],
    residual_information = z$residual_information[[1L]],
    omega_bread_point = z$omega_bread_point[[1L]],
    omega_bread_kind = z$omega_bread_kind[[1L]],
    omega_meat_point = z$omega_meat_point[[1L]],
    stage2_information = z$stage2_information[[1L]],
    replications = nrow(z),
    rejection_rate = mean(z$reject),
    rejection_se = sqrt(mean(z$reject) * (1 - mean(z$reject)) / nrow(z)),
    rejection_rate_n_minus_1 = if (z$method[[1L]] == "ml2s")
      mean(z$reject_n_minus_1) else NA_real_,
    rejection_se_n_minus_1 = if (z$method[[1L]] == "ml2s")
      sqrt(mean(z$reject_n_minus_1) * (1 - mean(z$reject_n_minus_1)) /
             nrow(z)) else NA_real_,
    mean_scaling_factor = mean(z$scaling_factor),
    median_scaling_factor = stats::median(z$scaling_factor),
    mean_chisq_scaled = mean(z$chisq_scaled),
    omega_bread_indefinite_rate =
      mean(z$omega_bread_negative_eigenvalues > 0),
    stage2_information_indefinite_rate = if (all(is.na(
      z$stage2_information_negative_eigenvalues))) NA_real_ else mean(
        z$stage2_information_negative_eigenvalues > 0, na.rm = TRUE),
    projection_indefinite_rate =
      mean(z$projection_negative_eigenvalues > 0),
    median_min_omega_bread_eigenvalue =
      stats::median(z$min_omega_bread_eigenvalue),
    median_min_stage2_information_eigenvalue = if (all(is.na(
      z$min_stage2_information_eigenvalue))) NA_real_ else stats::median(
        z$min_stage2_information_eigenvalue, na.rm = TRUE),
    median_min_projection_eigenvalue =
      stats::median(z$min_projection_eigenvalue),
    stringsAsFactors = FALSE)))
}

make_design <- function(profile) {
  if (profile == "fingerprint") {
    return(data.frame(
      cell_id = sprintf("fingerprint-mcar-n%d-k%d",
                        rep(c(200L, 400L, 600L), each = 2L),
                        rep(c(7L, 15L), 3L)),
      kurtosis = rep(c(7L, 15L), 3L), skewness = 2,
      n = rep(c(200L, 400L, 600L), each = 2L),
      missing_proportion = 0.30, patterns = "MP", mechanism = "MCAR",
      model = "Savalei-Falk Model 2",
      paper_ts_rejection = c(0.12, 0.18, 0.067, 0.10, 0.067, 0.070),
      paper_fiml_rejection = c(0.45, 0.51, 0.195, 0.22, 0.122, 0.145),
      paper_target_source = paste0(
        "Figure ", rep(8:10, each = 2L),
        ", graph-digitized approximate"),
      stringsAsFactors = FALSE))
  }
  data.frame(
    cell_id = c("normal-mar-l-k0", "paper-mar-l-k7", "paper-mar-l-k15",
                "diagnostic-mcar-k7"),
    kurtosis = c(0L, 7L, 15L, 7L), skewness = c(0, 2, 2, 2), n = 200L,
    missing_proportion = 0.30, patterns = "MP",
    mechanism = c("MAR-L", "MAR-L", "MAR-L", "MCAR"),
    model = "Savalei-Falk Model 2",
    paper_ts_rejection = c(NA_real_, 0.100, NA_real_, 0.12),
    paper_fiml_rejection = c(NA_real_, 0.633, NA_real_, 0.45),
    paper_target_source = c(NA_character_, "text, exact", NA_character_,
                            "Figure 8, graph-digitized approximate"),
    stringsAsFactors = FALSE)
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
out_dir <- args$results_dir %||% experiment_path("results", args$profile)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
pop <- model_2_population()
model <- magmaan::model_spec(pop$syntax, meanstructure = TRUE)
design <- make_design(args$profile)
design <- transform(design,
  fleishman_a = NA_real_, fleishman_b = NA_real_,
  fleishman_c = NA_real_, fleishman_d = NA_real_,
  transform_derivative_minimum = NA_real_,
  theoretical_eligible_proportion = NA_real_,
  intermediate_min_eigenvalue = NA_real_,
  candidate_draws_generated = NA_integer_,
  candidate_draws_consumed = NA_integer_,
  eligibility_rejections = NA_integer_,
  mean_realized_patterns = NA_real_,
  mean_realized_missing_rate = NA_real_)

all_raw <- list()
reused_cells <- 0L
wall_start <- proc.time()[["elapsed"]]
for (cell in seq_len(nrow(design))) {
  cell_id <- design$cell_id[[cell]]
  mechanism <- design$mechanism[[cell]]
  k <- design$kurtosis[[cell]]
  s <- design$skewness[[cell]]
  n <- design$n[[cell]]
  cat(sprintf("[%s] calibrating VM skewness %g, excess kurtosis %d...\n",
              cell_id, s, k))
  cal <- magmaan::magmaan_core$sim_vm_calibrate(
    pop$Sigma, rep(s, 21L), rep(k, 21L))
  coef <- unname(cal$coefficients[1L, ])
  design[cell, c("fleishman_a", "fleishman_b", "fleishman_c",
                 "fleishman_d")] <- coef
  design$transform_derivative_minimum[[cell]] <- if (coef[[4L]] > 0) {
    coef[[2L]] - coef[[3L]]^2 / (3 * coef[[4L]])
  } else coef[[2L]]
  zero <- stats::uniroot(
    function(z) coef[[1L]] + coef[[2L]] * z +
      coef[[3L]] * z^2 + coef[[4L]] * z^3,
    c(-10, 10))$root
  design$theoretical_eligible_proportion[[cell]] <- if (mechanism == "MAR-L") {
    stats::pnorm(zero, lower.tail = FALSE)
  } else NA_real_
  design$intermediate_min_eigenvalue[[cell]] <-
    min(eigen(cal$intermediate_corr, symmetric = TRUE,
              only.values = TRUE)$values)
  candidate_n <- if (mechanism == "MAR-L") {
    as.integer(ceiling(1.15 * args$reps) + 12L)
  } else args$reps
  candidate_draws <- magmaan::magmaan_core$sim_vm_draw(
    cal, n = n, reps = candidate_n,
    seed_base = args$seed_base + k * 1000000 + (n - 200) * 10000)$draws
  feasible <- if (mechanism == "MAR-L") {
    vapply(candidate_draws, mp_mar_l_is_feasible, logical(1))
  } else rep(TRUE, length(candidate_draws))
  chosen <- which(feasible)
  if (length(chosen) < args$reps) {
    stop("candidate batch did not contain enough feasible samples",
         call. = FALSE)
  }
  chosen <- chosen[seq_len(args$reps)]
  draws <- candidate_draws[chosen]
  design$candidate_draws_generated[[cell]] <- candidate_n
  design$candidate_draws_consumed[[cell]] <- max(chosen)
  design$eligibility_rejections[[cell]] <- max(chosen) - args$reps
  legacy_checkpoint_tag <- if (mechanism == "MAR-L") {
    sprintf("k%d", k)
  } else sprintf("%s-k%d", tolower(mechanism), k)
  checkpoint <- file.path(out_dir, sprintf(
    "replications-%s-N%d-R%d-seed%.0f-checkpoint.csv",
    cell_id, n, args$reps, args$seed_base))
  legacy_checkpoint <- experiment_path(
    "results", "replication", sprintf(
      "replications-%s-n%d-seed%.0f-checkpoint.csv",
      legacy_checkpoint_tag, args$reps, args$seed_base))
  checkpoint_candidates <- unique(c(
    checkpoint, if (n == 200L) legacy_checkpoint else character()))
  reusable <- checkpoint_candidates[file.exists(checkpoint_candidates)]
  if (args$resume && length(reusable)) {
    cached <- read.csv(reusable[[1L]], stringsAsFactors = FALSE,
                       check.names = FALSE)
    cached$cell_id <- cell_id
    cached$mechanism <- mechanism
    cached$skewness <- s
    cached <- add_statistic_conventions(cached, n)
    cached$error[is.na(cached$error)] <- ""
    cached_counts <- table(cached$rep)
    cached_methods <- with(cached, table(rep, method))
    if (nrow(cached) == 81L * args$reps &&
        length(cached_counts) == args$reps && all(cached_counts == 81L) &&
        all(cached_methods[, "fiml"] == 48L) &&
        all(cached_methods[, "ml2s"] == 32L) &&
        all(cached_methods[, "fiml_mlr"] == 1L)) {
      all_raw[[cell]] <- cached
      reused_cells <- reused_cells + 1L
      cat(sprintf("[%s] reused %s (%d/%d)\n",
                  cell_id, reusable[[1L]], args$reps, args$reps))
      next
    }
  }
  chunk_size <- max(args$cores, ceiling(args$reps / 20L))
  chunks <- split(seq_len(args$reps),
                  ceiling(seq_len(args$reps) / chunk_size))
  cell_rows <- list()
  for (chunk_id in seq_along(chunks)) {
    ids <- chunks[[chunk_id]]
    worker <- function(i) fit_one(
      draws[[i]], model, i, cell_id, mechanism, s, k, n, args$seed_base)
    rows <- if (args$cores > 1L && length(ids) > 1L) {
      parallel::mclapply(ids, worker, mc.cores = min(args$cores, length(ids)),
                         mc.preschedule = TRUE, mc.set.seed = FALSE)
    } else lapply(ids, worker)
    cell_rows[[chunk_id]] <- do.call(rbind, rows)
    cat(sprintf(
      "[%s] %d/%d replications complete (%.1f%%)\n",
      cell_id, max(ids), args$reps, 100 * max(ids) / args$reps))
  }
  all_raw[[cell]] <- do.call(rbind, cell_rows)
  write_csv(all_raw[[cell]], checkpoint)
}
raw <- do.call(rbind, all_raw)
for (cell in seq_len(nrow(design))) {
  cell_rows <- raw[raw$cell_id == design$cell_id[[cell]], , drop = FALSE]
  cell_rows <- cell_rows[!duplicated(cell_rows$rep), , drop = FALSE]
  design$mean_realized_patterns[[cell]] <- mean(cell_rows$patterns, na.rm = TRUE)
  design$mean_realized_missing_rate[[cell]] <-
    mean(cell_rows$missing_rate, na.rm = TRUE)
}
summary <- summarize_results(raw)
dgp_root_cells <- which(design$kurtosis > 0 & !duplicated(
  design[, c("skewness", "kurtosis")]))
dgp_roots <- do.call(rbind, lapply(dgp_root_cells, function(cell)
  enumerate_vm_roots(
    pop$Sigma, design$skewness[[cell]], design$kurtosis[[cell]],
    c(b = design$fleishman_b[[cell]], c = design$fleishman_c[[cell]],
      d = design$fleishman_d[[cell]]))))
matrix_audit_failures <- raw$method == "ml2s" & !raw$matrix_audit_ok &
  !is.na(raw$information)
raw$error[matrix_audit_failures & !nzchar(raw$error)] <-
  "ML2S saturated-identity/default-scale matrix audit failed"
failure_rows <- (!is.na(raw[["error"]]) &
                   nzchar(as.character(raw[["error"]]))) |
  matrix_audit_failures
failure_columns <- c("rep", "cell_id", "mechanism", "kurtosis", "method",
                     "converged", "error")
failures <- raw[failure_rows, failure_columns, drop = FALSE]
failures <- failures[!duplicated(failures), , drop = FALSE]
valid <- raw[!is.na(raw$information), , drop = FALSE]
audited_ml2s <- valid[valid$method == "ml2s" & valid$matrix_audit_ok, ]
bad_ml2s_keys <- unique(paste(
  valid$cell_id[valid$method == "ml2s" & !valid$matrix_audit_ok],
  valid$rep[valid$method == "ml2s" & !valid$matrix_audit_ok]))
replication_key <- interaction(valid$cell_id, valid$rep, drop = TRUE)
method_counts <- with(valid, table(cell_id, rep, method))
audit <- data.frame(
  key = c("saturated_identity_max_abs", "stage1_bread_difference_max_abs",
          "default_scale_max_abs",
          "ml2s_matrix_audit_failed_replications",
          "delta_rank_min", "delta_rank_max", "df_min", "df_max"),
  value = c(max(audited_ml2s$saturated_identity_error, na.rm = TRUE),
            max(audited_ml2s$stage1_bread_difference, na.rm = TRUE),
            max(audited_ml2s$default_scale_error, na.rm = TRUE),
            length(bad_ml2s_keys),
            min(valid$delta_rank), max(valid$delta_rank),
            min(valid$df), max(valid$df)),
  stringsAsFactors = FALSE)
write_csv(raw, file.path(out_dir, "replications.csv"))
write_csv(summary, file.path(out_dir, "summary.csv"))
write_csv(failures, file.path(out_dir, "failures.csv"))
write_csv(design, file.path(out_dir, "design.csv"))
write_csv(dgp_roots, file.path(out_dir, "dgp_roots.csv"))
write_csv(audit, file.path(out_dir, "audit.csv"))

stopifnot(
  nrow(valid) > 0,
  all(table(replication_key) == 81L),
  all(method_counts[, , "fiml"] == 48L),
  all(method_counts[, , "fiml_mlr"] == 1L),
  all(method_counts[, , "ml2s"] == 32L),
  max(audited_ml2s$saturated_identity_error, na.rm = TRUE) < 1e-7,
  max(audited_ml2s$default_scale_error, na.rm = TRUE) < 1e-7,
  length(bad_ml2s_keys) <= max(1L, ceiling(0.01 * nrow(design) * args$reps)),
  all(valid$delta_rank == 66L),
  all(valid$df == 186L))

write_metadata(
  file.path(out_dir, "metadata.csv"),
  values = list(
    experiment = "80-savalei-falk-2014-ml2s-information",
    profile = args$profile,
    reps_per_cell = args$reps,
    cells = nrow(design),
    seed_base = args$seed_base,
    cores = args$cores,
    resume = args$resume,
    checkpoint_cells_reused = reused_cells,
    elapsed_seconds = proc.time()[["elapsed"]] - wall_start,
    paper = "Savalei and Falk (2014), doi:10.1080/10705511.2014.882692",
    paper_target = if (args$profile == "fingerprint") paste(
      "Model 2 MCAR/MP/30% Figures 8-10 fingerprint",
      "at N=200/400/600 and s2k7/s2k15") else paste(
      "Model 2 matrix-choice diagnostics at N=200:",
      "two nonnormal MAR-L cells, a normal control, and an MCAR control"),
    statistic_conventions = paste(
      "direct FIML uses the raw likelihood-ratio T_ML;",
      "ML2S records both magmaan n*F and Savalei-Falk (n-1)*F"),
    kurtosis_convention = "Fleishman excess kurtosis, matching the EQS/SEM convention",
    git_head = git_scalar(c("rev-parse", "HEAD")),
    git_dirty = git_dirty()),
  packages = "magmaan")
cat("Wrote results to ", out_dir, "\n", sep = "")
