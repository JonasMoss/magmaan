# Self-contained score-shape and SEM-GOF machinery for experiment 71.
# This experiment is a leaf: it uses only magmaan and experiments/_support.

pss_population <- function() {
  lambda <- c(.8, .9, 1.1, 1.4, .7)
  theta <- diag(length(lambda))
  Sigma <- tcrossprod(lambda) + theta
  list(lambda = lambda, theta = theta, Sigma = Sigma, p = length(lambda),
       df = as.integer(length(lambda) * (length(lambda) - 3L) / 2L))
}

pss_model_spec <- function(p = 5L) {
  magmaan::model_spec(
    paste0("f =~ ", paste0("x", seq_len(p), collapse = " + ")),
    std_lv = FALSE, meanstructure = FALSE)
}

pss_calibrate_samplers <- function(pop, skew = 2, exkurt = 7) {
  list(
    normal = list(kind = "normal", L = chol(pop$Sigma), sds = sqrt(diag(pop$Sigma))),
    vm = list(
      kind = "vm",
      calibration = magmaan:::sim_vm_calibrate_impl(
        stats::cov2cor(pop$Sigma),
        rep(skew, pop$p), rep(exkurt, pop$p)),
      sds = sqrt(diag(pop$Sigma))))
}

pss_draw <- function(sampler, n, seed) {
  if (sampler$kind == "normal") {
    set.seed(as.integer(seed))
    X <- matrix(stats::rnorm(n * ncol(sampler$L)), nrow = n) %*% sampler$L
  } else {
    batch <- magmaan:::sim_vm_draw_impl(
      sampler$calibration, n = as.integer(n), reps = 1L,
      seed_base = as.integer(seed))
    X <- sweep(batch$draws[[1L]], 2L, sampler$sds, "*")
  }
  colnames(X) <- paste0("x", seq_len(ncol(X)))
  storage.mode(X) <- "double"
  X
}

pss_fit <- function(spec, X) {
  magmaan::magmaan(
    spec, as.data.frame(X), estimator = "ML",
    optimizer = "nlopt-lbfgs-slsqp-fallback", se = "none", test = "none")
}

# Model-centred saturated covariance contributions in the expected-information
# residual basis. Their unstudentized quadratic is magmaan's structured-weight
# RLS GOF statistic; their centered covariance supplies the robust score meat.
pss_projected_scores <- function(fit, X) {
  core <- magmaan::magmaan_core
  uf <- core$infer_build_u_factor(
    fit, bread = "expected", moments = "structured")
  if (!identical(uf$kind, "ProjectionExpected") ||
      length(uf$blocks) != 1L || isTRUE(uf$has_means)) {
    stop("experiment 71 expects one covariance-only ML block", call. = FALSE)
  }
  centred <- core$infer_casewise_contributions(fit$partable, X)
  residual_matrix <- uf$blocks[[1L]]$S - uf$blocks[[1L]]$Sigma_hat
  residual <- residual_matrix[lower.tri(residual_matrix, diag = TRUE)]
  if (ncol(centred) != length(residual) ||
      nrow(uf$B) != length(residual)) {
    stop("covariance contribution layout mismatch", call. = FALSE)
  }
  model_centred <- sweep(centred, 2L, residual, "+")
  psi <- model_centred %*% uf$B
  if (ncol(psi) < 1L || any(!is.finite(psi))) {
    stop("projected score matrix is empty or non-finite", call. = FALSE)
  }
  psi
}

.pss_inverse_quadratic <- function(u, M, label) {
  eig <- eigen(0.5 * (M + t(M)), symmetric = TRUE)
  tol <- 1e-10 * max(1, max(eig$values))
  if (any(!is.finite(eig$values)) || min(eig$values) <= tol) {
    return(list(
      statistic = NA_real_, available = FALSE,
      min_eigenvalue = min(eig$values), condition = Inf,
      error = paste0(label, " is not positive definite")))
  }
  coordinates <- crossprod(eig$vectors, u)
  list(
    statistic = sum(coordinates^2 / eig$values),
    available = TRUE,
    min_eigenvalue = min(eig$values),
    condition = max(eig$values) / min(eig$values),
    error = "")
}

# For z = B^(-1/2)(psi-E psi), define H = zz' - I. The score-only projection
# H-perp = H - Cov(H,z)z removes the first-order covariance between numerator
# and studentizer. Matching E||H-perp||_F^2 to a q-variate Wishart meat gives
#
#   shape = q(q+1) / E||H-perp||_F^2,
#   eta_n = (n-1) shape.
#
# The unprojected version leaves H unchanged. Full vec(H) is intentional:
# off-diagonal entries count twice, exactly as in the Frobenius norm.
pss_score_shape <- function(psi) {
  n <- nrow(psi)
  q <- ncol(psi)
  if (n <= q + 1L) stop("score-shape estimate needs n > q + 1", call. = FALSE)
  xc <- sweep(psi, 2L, colMeans(psi), "-")
  meat <- crossprod(xc) / (n - 1)
  eig <- eigen(0.5 * (meat + t(meat)), symmetric = TRUE)
  tol <- 1e-10 * max(1, max(eig$values))
  if (min(eig$values) <= tol || any(!is.finite(eig$values))) {
    stop("score-shape meat is not positive definite", call. = FALSE)
  }
  whitening <- sweep(eig$vectors, 2L, sqrt(eig$values), "/")
  z <- xc %*% whitening
  identity <- diag(q)
  H <- t(vapply(seq_len(n), function(i) {
    as.vector(tcrossprod(z[i, ]) - identity)
  }, numeric(q * q)))
  H <- sweep(H, 2L, colMeans(H), "-")
  covariance_hz <- crossprod(z, H) / (n - 1)
  H_perp <- H - z %*% covariance_hz
  omega_unprojected <- sum(H^2) / (n - 1)
  omega_projected <- sum(H_perp^2) / (n - 1)
  numerator <- q * (q + 1)
  list(
    q = q,
    omega_unprojected = omega_unprojected,
    omega_projected = omega_projected,
    shape_unprojected = numerator / omega_unprojected,
    shape_projected = numerator / omega_projected,
    projection_fraction = 1 - omega_projected / omega_unprojected,
    meat_min_eigenvalue = min(eig$values),
    meat_condition = max(eig$values) / min(eig$values))
}

pss_hotelling_p <- function(statistic, q, eta) {
  if (!is.finite(statistic) || !is.finite(eta) || eta <= q - 1) {
    return(NA_real_)
  }
  df2 <- eta - q + 1
  transformed <- ((eta - q + 1) / (eta * q)) * statistic
  stats::pf(transformed, df1 = q, df2 = df2, lower.tail = FALSE)
}

pss_reference_shape <- function(spec, sampler, distribution, n_reference,
                                reference_rep, seed) {
  X <- pss_draw(sampler, n_reference, seed)
  fit <- pss_fit(spec, X)
  if (!isTRUE(fit$converged)) {
    stop("reference fit did not converge for ", distribution, call. = FALSE)
  }
  shape <- pss_score_shape(pss_projected_scores(fit, X))
  data.frame(
    distribution = distribution,
    n_reference = n_reference,
    reference_rep = reference_rep,
    q = shape$q,
    omega_unprojected = shape$omega_unprojected,
    omega_projected = shape$omega_projected,
    shape_unprojected = shape$shape_unprojected,
    shape_projected = shape$shape_projected,
    projection_fraction = shape$projection_fraction,
    meat_min_eigenvalue = shape$meat_min_eigenvalue,
    meat_condition = shape$meat_condition,
    stringsAsFactors = FALSE)
}

pss_summarize_reference <- function(reference_shapes) {
  pieces <- split(reference_shapes, reference_shapes$distribution)
  out <- do.call(rbind, lapply(pieces, function(x) {
    qtile <- function(column, probability) {
      unname(stats::quantile(x[[column]], probability, type = 8))
    }
    data.frame(
      distribution = x$distribution[[1L]],
      n_reference = x$n_reference[[1L]],
      reference_reps = nrow(x),
      q = x$q[[1L]],
      omega_unprojected = stats::median(x$omega_unprojected),
      omega_projected = stats::median(x$omega_projected),
      shape_unprojected = stats::median(x$shape_unprojected),
      shape_projected = stats::median(x$shape_projected),
      shape_projected_p10 = qtile("shape_projected", .1),
      shape_projected_p90 = qtile("shape_projected", .9),
      projection_fraction = stats::median(x$projection_fraction),
      meat_min_eigenvalue = stats::median(x$meat_min_eigenvalue),
      meat_condition = stats::median(x$meat_condition),
      stringsAsFactors = FALSE)
  }))
  row.names(out) <- NULL
  out
}

pss_one_rep <- function(cell, rep_id, spec, sampler, reference_shapes,
                        seed_base, run_fmg = TRUE) {
  seed <- as.integer(seed_base + cell$cell_id * 100000L + rep_id)
  base <- data.frame(
    cell_id = cell$cell_id, distribution = cell$distribution, n = cell$n,
    q = cell$q, rep = rep_id, seed = seed,
    fit_ok = FALSE, score_ok = FALSE,
    fmg_ok = if (run_fmg) FALSE else NA,
    fit_error = "", score_error = "", fmg_error = "",
    fit_seconds = NA_real_, score_seconds = NA_real_, fmg_seconds = NA_real_,
    q_centered = NA_real_, q_raw = NA_real_, q_raw_centered_ratio = NA_real_,
    rls_statistic = NA_real_, raw_opg_identity_error = NA_real_,
    centered_meat_min_eigenvalue = NA_real_,
    centered_meat_condition = NA_real_,
    shape_empirical_unprojected = NA_real_,
    shape_empirical_projected = NA_real_,
    projection_fraction_empirical = NA_real_,
    eta_empirical_unprojected = NA_real_,
    eta_empirical_projected = NA_real_,
    eta_empirical_unprojected_available = NA,
    eta_empirical_projected_available = NA,
    stringsAsFactors = FALSE)
  p_names <- c(
    "p_score_chisq_centered", "p_score_chisq_raw",
    "p_hotelling_gaussian",
    "p_satterthwaite_normal_unprojected",
    "p_satterthwaite_normal_projected",
    "p_satterthwaite_oracle_projected",
    "p_satterthwaite_empirical_unprojected",
    "p_satterthwaite_empirical_projected",
    "p_fmg_sb_rls", "p_fmg_mv_rls", "p_fmg_peba4_rls")
  p <- stats::setNames(rep(NA_real_, length(p_names)), p_names)

  X <- tryCatch(pss_draw(sampler, cell$n, seed), error = function(e) e)
  if (inherits(X, "error")) {
    base$fit_error <- paste0("draw: ", conditionMessage(X))
    return(cbind(base, as.data.frame(as.list(p))))
  }
  fit_begin <- proc.time()[["elapsed"]]
  fit <- tryCatch(pss_fit(spec, X), error = function(e) e)
  base$fit_seconds <- proc.time()[["elapsed"]] - fit_begin
  if (inherits(fit, "error")) {
    base$fit_error <- conditionMessage(fit)
    return(cbind(base, as.data.frame(as.list(p))))
  }
  if (!isTRUE(fit$converged)) {
    base$fit_error <- "ML fit did not converge"
    return(cbind(base, as.data.frame(as.list(p))))
  }
  base$fit_ok <- TRUE

  score_begin <- proc.time()[["elapsed"]]
  score <- tryCatch({
    psi <- pss_projected_scores(fit, X)
    n <- nrow(psi)
    q <- ncol(psi)
    u <- sqrt(n) * colMeans(psi)
    xc <- sweep(psi, 2L, colMeans(psi), "-")
    meat_centered <- crossprod(xc) / (n - 1)
    meat_raw <- crossprod(psi) / n
    qc <- .pss_inverse_quadratic(u, meat_centered, "centered meat")
    qr <- .pss_inverse_quadratic(u, meat_raw, "raw meat")
    shape <- pss_score_shape(psi)
    if (!qc$available) stop(qc$error, call. = FALSE)
    raw_opg_identity_error <- if (qr$available) {
      abs(qr$statistic - n * qc$statistic / (n - 1 + qc$statistic))
    } else {
      NA_real_
    }
    if (is.finite(raw_opg_identity_error) &&
        raw_opg_identity_error > 1e-8 * max(1, abs(qr$statistic))) {
      stop("raw-OPG rank-one identity failed", call. = FALSE)
    }

    normal <- reference_shapes[
      reference_shapes$distribution == "normal", , drop = FALSE]
    oracle <- reference_shapes[
      reference_shapes$distribution == cell$distribution, , drop = FALSE]
    if (nrow(normal) != 1L || nrow(oracle) != 1L) {
      stop("reference score shapes are incomplete", call. = FALSE)
    }
    eta_gaussian <- n - 1
    eta_normal_unprojected <- (n - 1) * normal$shape_unprojected
    eta_normal_projected <- (n - 1) * normal$shape_projected
    eta_oracle_projected <- (n - 1) * oracle$shape_projected
    eta_empirical_unprojected <- (n - 1) * shape$shape_unprojected
    eta_empirical_projected <- (n - 1) * shape$shape_projected

    list(
      q = q, qc = qc, qr = qr, shape = shape,
      rls = sum(colSums(psi)^2) / n,
      raw_opg_identity_error = raw_opg_identity_error,
      eta_gaussian = eta_gaussian,
      eta_normal_unprojected = eta_normal_unprojected,
      eta_normal_projected = eta_normal_projected,
      eta_oracle_projected = eta_oracle_projected,
      eta_empirical_unprojected = eta_empirical_unprojected,
      eta_empirical_projected = eta_empirical_projected)
  }, error = function(e) e)
  base$score_seconds <- proc.time()[["elapsed"]] - score_begin
  if (inherits(score, "error")) {
    base$score_error <- conditionMessage(score)
  } else {
    base$score_ok <- TRUE
    base$q_centered <- score$qc$statistic
    base$q_raw <- score$qr$statistic
    base$q_raw_centered_ratio <- score$qr$statistic / score$qc$statistic
    base$rls_statistic <- score$rls
    base$raw_opg_identity_error <- score$raw_opg_identity_error
    base$centered_meat_min_eigenvalue <- score$qc$min_eigenvalue
    base$centered_meat_condition <- score$qc$condition
    base$shape_empirical_unprojected <- score$shape$shape_unprojected
    base$shape_empirical_projected <- score$shape$shape_projected
    base$projection_fraction_empirical <- score$shape$projection_fraction
    base$eta_empirical_unprojected <- score$eta_empirical_unprojected
    base$eta_empirical_projected <- score$eta_empirical_projected
    base$eta_empirical_unprojected_available <-
      score$eta_empirical_unprojected > score$q - 1
    base$eta_empirical_projected_available <-
      score$eta_empirical_projected > score$q - 1
    p[["p_score_chisq_centered"]] <- stats::pchisq(
      score$qc$statistic, score$q, lower.tail = FALSE)
    p[["p_score_chisq_raw"]] <- stats::pchisq(
      score$qr$statistic, score$q, lower.tail = FALSE)
    p[["p_hotelling_gaussian"]] <- pss_hotelling_p(
      score$qc$statistic, score$q, score$eta_gaussian)
    p[["p_satterthwaite_normal_unprojected"]] <- pss_hotelling_p(
      score$qc$statistic, score$q, score$eta_normal_unprojected)
    p[["p_satterthwaite_normal_projected"]] <- pss_hotelling_p(
      score$qc$statistic, score$q, score$eta_normal_projected)
    p[["p_satterthwaite_oracle_projected"]] <- pss_hotelling_p(
      score$qc$statistic, score$q, score$eta_oracle_projected)
    p[["p_satterthwaite_empirical_unprojected"]] <- pss_hotelling_p(
      score$qc$statistic, score$q, score$eta_empirical_unprojected)
    p[["p_satterthwaite_empirical_projected"]] <- pss_hotelling_p(
      score$qc$statistic, score$q, score$eta_empirical_projected)
  }

  if (run_fmg) {
    fmg_begin <- proc.time()[["elapsed"]]
    fmg <- tryCatch(
      magmaan::fmg_tests(
        fit, tests = c("sb_rls", "mv_rls", "peba4_rls"), data = X),
      error = function(e) e)
    base$fmg_seconds <- proc.time()[["elapsed"]] - fmg_begin
    if (inherits(fmg, "error")) {
      base$fmg_error <- conditionMessage(fmg)
    } else {
      base$fmg_ok <- TRUE
      for (label in c("sb_rls", "mv_rls", "peba4_rls")) {
        row <- which(fmg$label == label)
        if (length(row) == 1L) {
          p[[paste0("p_fmg_", label)]] <- fmg$p_value[[row]]
        }
      }
    }
  }
  cbind(base, as.data.frame(as.list(p)))
}

pss_attach_independent_shape_pairs <- function(raw) {
  raw$pair_role <- "unused"
  raw$independent_shape_donor_rep <- NA_integer_
  raw$shape_independent_projected <- NA_real_
  raw$eta_independent_projected <- NA_real_
  raw$eta_independent_projected_available <- NA

  matched <- c(
    "score_chisq_centered",
    "hotelling_gaussian",
    "satterthwaite_oracle_projected",
    "satterthwaite_empirical_projected")
  for (method in matched) {
    raw[[paste0("p_", method, "_matched")]] <- NA_real_
  }
  raw$p_satterthwaite_independent_projected <- NA_real_

  cell_rows <- split(seq_len(nrow(raw)), raw$cell_id)
  for (indices in cell_rows) {
    indices <- indices[order(raw$rep[indices])]
    n_pairs <- length(indices) %/% 2L
    if (!n_pairs) next
    target <- indices[seq_len(n_pairs)]
    donor <- indices[n_pairs + seq_len(n_pairs)]
    raw$pair_role[target] <- "target"
    raw$pair_role[donor] <- "shape_donor"
    raw$independent_shape_donor_rep[target] <- raw$rep[donor]
    raw$shape_independent_projected[target] <-
      raw$shape_empirical_projected[donor]
    raw$eta_independent_projected[target] <-
      (raw$n[target] - 1) * raw$shape_independent_projected[target]
    raw$eta_independent_projected_available[target] <-
      is.finite(raw$eta_independent_projected[target]) &
      raw$eta_independent_projected[target] > raw$q[target] - 1

    for (method in matched) {
      source <- paste0("p_", method)
      destination <- paste0(source, "_matched")
      raw[[destination]][target] <- raw[[source]][target]
    }
    raw$p_satterthwaite_independent_projected[target] <- mapply(
      pss_hotelling_p,
      statistic = raw$q_centered[target],
      q = raw$q[target],
      eta = raw$eta_independent_projected[target])
  }
  raw
}

pss_wilson <- function(rejected, n, z = 1.95996398454005) {
  if (!is.finite(n) || n <= 0L) return(c(lower = NA_real_, upper = NA_real_))
  phat <- rejected / n
  den <- 1 + z^2 / n
  centre <- (phat + z^2 / (2 * n)) / den
  half <- z * sqrt(phat * (1 - phat) / n + z^2 / (4 * n^2)) / den
  c(lower = max(0, centre - half), upper = min(1, centre + half))
}

pss_summarize_methods <- function(raw) {
  p_columns <- grep("^p_", names(raw), value = TRUE)
  long <- do.call(rbind, lapply(p_columns, function(column) {
    data.frame(
      raw[c("cell_id", "distribution", "n", "q", "rep")],
      method = sub("^p_", "", column),
      p_value = raw[[column]], stringsAsFactors = FALSE)
  }))
  pieces <- split(long, interaction(long$cell_id, long$method, drop = TRUE))
  out <- do.call(rbind, lapply(pieces, function(x) {
    valid <- is.finite(x$p_value)
    n_valid <- sum(valid)
    rejected <- sum(x$p_value[valid] <= .05)
    ci <- pss_wilson(rejected, n_valid)
    data.frame(
      x[1L, c("cell_id", "distribution", "n", "q", "method")],
      n_valid = n_valid, rejected = rejected,
      rejection_rate = if (n_valid) rejected / n_valid else NA_real_,
      ci_lower = ci[[1L]], ci_upper = ci[[2L]],
      mean_p = if (n_valid) mean(x$p_value[valid]) else NA_real_,
      stringsAsFactors = FALSE)
  }))
  out[order(out$cell_id, out$method), , drop = FALSE]
}

pss_summarize_independent_pairs <- function(raw) {
  required <- c(
    "p_satterthwaite_empirical_projected_matched",
    "p_satterthwaite_independent_projected",
    "eta_empirical_projected", "eta_independent_projected")
  if (!all(required %in% names(raw))) {
    stop("independent shape pairs have not been attached", call. = FALSE)
  }
  pieces <- split(raw[raw$pair_role == "target", , drop = FALSE],
                  raw$cell_id[raw$pair_role == "target"])
  out <- do.call(rbind, lapply(pieces, function(x) {
    same_p <- x$p_satterthwaite_empirical_projected_matched
    independent_p <- x$p_satterthwaite_independent_projected
    valid <- is.finite(same_p) & is.finite(independent_p)
    same_reject <- same_p[valid] <= .05
    independent_reject <- independent_p[valid] <= .05
    difference <- as.numeric(same_reject) - as.numeric(independent_reject)
    n_valid <- sum(valid)
    paired_se <- if (n_valid > 1L) {
      stats::sd(difference) / sqrt(n_valid)
    } else {
      NA_real_
    }
    safe_cor <- function(a, b) {
      keep <- is.finite(a) & is.finite(b)
      if (sum(keep) > 2L &&
          stats::sd(a[keep]) > 0 && stats::sd(b[keep]) > 0) {
        stats::cor(a[keep], b[keep])
      } else {
        NA_real_
      }
    }
    data.frame(
      x[1L, c("cell_id", "distribution", "n", "q")],
      n_pairs = n_valid,
      rejection_same_sample =
        if (n_valid) mean(same_reject) else NA_real_,
      rejection_independent =
        if (n_valid) mean(independent_reject) else NA_real_,
      rejection_difference =
        if (n_valid) mean(difference) else NA_real_,
      difference_se = paired_se,
      same_only = if (n_valid) {
        sum(same_reject & !independent_reject)
      } else {
        0L
      },
      independent_only = if (n_valid) {
        sum(!same_reject & independent_reject)
      } else {
        0L
      },
      median_shape_same = stats::median(
        x$shape_empirical_projected, na.rm = TRUE),
      median_shape_independent = stats::median(
        x$shape_independent_projected, na.rm = TRUE),
      median_eta_same = stats::median(
        x$eta_empirical_projected, na.rm = TRUE),
      median_eta_independent = stats::median(
        x$eta_independent_projected, na.rm = TRUE),
      cor_q_eta_same = safe_cor(
        x$q_centered, x$eta_empirical_projected),
      cor_q_eta_independent = safe_cor(
        x$q_centered, x$eta_independent_projected),
      stringsAsFactors = FALSE)
  }))
  row.names(out) <- NULL
  out
}

pss_summarize_diagnostics <- function(raw) {
  columns <- c(
    "q_centered", "q_raw", "q_raw_centered_ratio", "rls_statistic",
    "raw_opg_identity_error",
    "centered_meat_min_eigenvalue", "centered_meat_condition",
    "shape_empirical_unprojected", "shape_empirical_projected",
    "projection_fraction_empirical", "eta_empirical_unprojected",
    "eta_empirical_projected", "fit_seconds", "score_seconds", "fmg_seconds")
  optional <- c("shape_independent_projected", "eta_independent_projected")
  columns <- c(columns, optional[optional %in% names(raw)])
  pieces <- split(raw, raw$cell_id)
  out <- do.call(rbind, lapply(pieces, function(x) {
    base <- x[1L, c("cell_id", "distribution", "n", "q")]
    for (column in columns) {
      values <- x[[column]]
      values <- values[is.finite(values)]
      base[[paste0("median_", column)]] <-
        if (length(values)) stats::median(values) else NA_real_
      base[[paste0("p90_", column)]] <-
        if (length(values)) unname(stats::quantile(values, .9, type = 8)) else NA_real_
      base[[paste0("p10_", column)]] <-
        if (length(values)) unname(stats::quantile(values, .1, type = 8)) else NA_real_
    }
    base$fit_success <- mean(x$fit_ok)
    base$score_success <- mean(x$score_ok)
    base$fmg_success <- mean(x$fmg_ok)
    base$eta_empirical_unprojected_availability <- mean(
      x$eta_empirical_unprojected_available, na.rm = TRUE)
    base$eta_empirical_projected_availability <- mean(
      x$eta_empirical_projected_available, na.rm = TRUE)
    if ("eta_independent_projected_available" %in% names(x)) {
      base$eta_independent_projected_availability <- mean(
        x$eta_independent_projected_available, na.rm = TRUE)
    }
    base
  }))
  row.names(out) <- NULL
  out
}
