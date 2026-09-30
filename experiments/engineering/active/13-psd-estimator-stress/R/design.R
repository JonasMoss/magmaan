model_syntax <- function(structure = "one_factor") {
  switch(
    structure,
    one_factor = "f =~ x1 + x2 + x3 + x4",
    two_factor = paste(
      "f1 =~ x1 + x2 + x3",
      "f2 =~ x4 + x5 + x6",
      "f1 ~~ f2",
      "x2 ~~ x3",
      sep = "\n"
    ),
    two_factor_misspecified = paste(
      "f1 =~ x1 + x2 + x3",
      "f2 =~ x4 + x5 + x6",
      "f1 ~~ f2",
      sep = "\n"
    ),
    two_group_one_factor = paste(
      "f =~ x1 + c(l2_g1,l2_g2)*x2 + c(l3_g1,l3_g2)*x3 + x4",
      "x1 ~~ c(theta_x1,theta_x1)*x1",
      "l2_g1 + l3_g1 == 1.60",
      sep = "\n"
    ),
    latent_regression = paste(
      "fy =~ y1 + y2 + y3",
      "fx =~ x1 + x2 + x3",
      "fy ~ fx",
      sep = "\n"
    ),
    stop("unknown structure: ", structure, call. = FALSE)
  )
}

geometry_row <- function(geometry, structure, stress_axis,
                         target_min_eigenvalue, theta_min, psi_min,
                         n_ratio, n_free, data_structure = structure,
                         n_groups = 1L, fit_plan = "full",
                         weight_condition_target = NA_real_,
                         misspecified = FALSE) {
  n <- as.integer(round(n_ratio * n_free))
  data.frame(
    geometry = geometry,
    structure = structure,
    data_structure = data_structure,
    stress_axis = stress_axis,
    target_min_eigenvalue = target_min_eigenvalue,
    theta_min = theta_min,
    psi_min = psi_min,
    n_ratio = n_ratio,
    n_free = n_free,
    n_groups = as.integer(n_groups),
    n = n,
    n_per_group = as.integer(round(n / n_groups)),
    fit_plan = fit_plan,
    weight_condition_target = weight_condition_target,
    misspecified = misspecified,
    stringsAsFactors = FALSE
  )
}

eigen_token <- function(value) {
  gsub("[.]", "p", format(value, scientific = FALSE, trim = TRUE))
}

pilot_geometry_catalog <- function() {
  targets <- c(0.20, 0.05, 0.01, 0.001, 0)
  rows <- lapply(targets, function(target) {
    geometry_row(
      paste0("cfa1_theta_", eigen_token(target)),
      "one_factor", "theta_boundary", target, target, 1, 10, 12
    )
  })
  rows <- c(rows, lapply(c(2, 5, 50), function(ratio) {
    geometry_row(
      paste0("cfa1_nratio_", ratio),
      "one_factor", "sample_size", 0.20, 0.20, 1, ratio, 12
    )
  }))
  rows[[length(rows) + 1L]] <- geometry_row(
    "cfa2_anchor", "two_factor", "anchor", 0.20, 0.20, 0.20, 10, 20
  )
  rows <- c(rows, lapply(targets[-1L], function(target) {
    geometry_row(
      paste0("cfa2_theta_", eigen_token(target)),
      "two_factor", "theta_boundary", target, target, 0.20, 10, 20
    )
  }))
  rows <- c(rows, lapply(targets[-1L], function(target) {
    geometry_row(
      paste0("cfa2_psi_", eigen_token(target)),
      "two_factor", "psi_boundary", target, 0.20, target, 10, 20
    )
  }))
  rows[[length(rows) + 1L]] <- geometry_row(
    "cfa2_joint_0p001", "two_factor", "joint_boundary", 0.001,
    0.001, 0.001, 10, 20
  )
  rows <- c(rows, lapply(targets, function(target) {
    geometry_row(
      paste0("regression_psi_", eigen_token(target)),
      "latent_regression", "psi_boundary", target, 0.35, target, 10, 19
    )
  }))
  rows[[length(rows) + 1L]] <- geometry_row(
    "cfa2_misspecified", "two_factor_misspecified", "misspecification",
    0.20, 0.20, 0.20, 10, 19, data_structure = "two_factor",
    misspecified = TRUE
  )
  rows[[length(rows) + 1L]] <- geometry_row(
    "two_group_anchor", "two_group_one_factor", "multi_group",
    0.20, 0.20, 1, 10, 22, data_structure = "two_group_one_factor",
    n_groups = 2L
  )
  rows[[length(rows) + 1L]] <- geometry_row(
    "two_group_theta_0p001", "two_group_one_factor", "multi_group_boundary",
    0.001, 0.001, 1, 10, 22, data_structure = "two_group_one_factor",
    n_groups = 2L
  )
  rows <- c(rows, lapply(c(1, 1e4, 1e8), function(condition) {
    geometry_row(
      paste0("fixed_wls_kappa_", format(condition, scientific = TRUE)),
      "one_factor", "weight_conditioning", 0.20, 0.20, 1, 10, 12,
      fit_plan = "fixed_wls", weight_condition_target = condition
    )
  }))
  do.call(rbind, rows)
}

geometry_catalog <- function(profile = "smoke") {
  if (identical(profile, "pilot")) return(pilot_geometry_catalog())
  if (!identical(profile, "smoke")) {
    stop("unknown profile: ", profile, call. = FALSE)
  }
  rbind(
    geometry_row(
      "interior", "one_factor", "anchor", 0.35, 0.35, 1, 10, 12
    ),
    geometry_row(
      "theta_near_boundary", "one_factor", "theta_boundary", 0.001,
      0.001, 1, 10, 12
    )
  )
}

simulation_seed <- function(seed_base, geometry, rep) {
  geometry_code <- sum(utf8ToInt(geometry) * seq_along(utf8ToInt(geometry)))
  as.integer(seed_base + 1009L * geometry_code + rep)
}

rmvn_psd <- function(n, covariance) {
  decomposition <- eigen(
    0.5 * (covariance + t(covariance)), symmetric = TRUE
  )
  if (min(decomposition$values) < -1e-10) {
    stop("simulation covariance is not positive semidefinite", call. = FALSE)
  }
  root <- decomposition$vectors %*%
    diag(sqrt(pmax(decomposition$values, 0)), nrow(covariance))
  matrix(stats::rnorm(n * nrow(covariance)), nrow = n) %*% t(root)
}

simulate_one_factor_data <- function(n, theta_min,
                                     loadings = c(1.0, 0.85, 0.75, 0.90),
                                     residual_variances = NULL,
                                     means = c(0.20, -0.10, 0.40, 0.80),
                                     latent_variance = 1) {
  eta <- stats::rnorm(n, sd = sqrt(latent_variance))
  residual_variances <- residual_variances %||%
    c(theta_min, 0.45, 0.55, 0.35)
  errors <- matrix(stats::rnorm(n * 4L), nrow = n, ncol = 4L)
  errors <- sweep(errors, 2L, sqrt(residual_variances), "*")
  out <- sweep(tcrossprod(eta, loadings) + errors, 2L, means, "+")
  out <- as.data.frame(out)
  names(out) <- paste0("x", seq_len(4L))
  out
}

two_group_population_moments <- function(theta_min) {
  loadings <- list(
    c(1.0, 0.85, 0.75, 0.90),
    c(1.0, 0.85, 0.70, 0.95)
  )
  residuals <- list(
    c(theta_min, 0.45, 0.55, 0.35),
    c(theta_min, 0.50, 0.40, 0.60)
  )
  latent_variances <- c(1.0, 1.25)
  means <- list(
    c(0.20, -0.10, 0.40, 0.80),
    c(-0.15, 0.25, 0.10, 0.55)
  )
  list(
    sigma = Map(function(lambda, theta, psi) {
      tcrossprod(lambda) * psi + diag(theta)
    }, loadings, residuals, latent_variances),
    mu = means,
    loadings = loadings,
    residuals = residuals,
    latent_variances = latent_variances
  )
}

simulate_two_group_one_factor_data <- function(n_per_group, theta_min) {
  population <- two_group_population_moments(theta_min)
  Map(function(lambda, residuals, means, latent_variance) {
    simulate_one_factor_data(
      n_per_group, theta_min, loadings = lambda,
      residual_variances = residuals, means = means,
      latent_variance = latent_variance
    )
  }, population$loadings, population$residuals, population$mu,
  population$latent_variances)
}

simulate_two_factor_data <- function(n, theta_min, psi_min) {
  psi <- matrix(c(1, 1 - psi_min, 1 - psi_min, 1), 2L, 2L)
  theta <- diag(c(0.50, 0.40, 0.40, 0.45, 0.55, 0.35))
  theta[2L, 3L] <- theta[3L, 2L] <- 0.40 - theta_min
  latent <- rmvn_psd(n, psi)
  errors <- rmvn_psd(n, theta)
  lambda <- matrix(0, 6L, 2L)
  lambda[1:3, 1L] <- c(1.0, 0.80, 0.70)
  lambda[4:6, 2L] <- c(1.0, 0.85, 0.75)
  means <- c(0.20, -0.10, 0.40, 0.80, -0.30, 0.10)
  out <- sweep(latent %*% t(lambda) + errors, 2L, means, "+")
  out <- as.data.frame(out)
  names(out) <- paste0("x", seq_len(6L))
  out
}

simulate_latent_regression_data <- function(n, psi_min) {
  eta_x <- stats::rnorm(n)
  eta_y <- 0.60 * eta_x + sqrt(psi_min) * stats::rnorm(n)
  latent <- cbind(eta_y, eta_x)
  lambda <- matrix(0, 6L, 2L)
  lambda[1:3, 1L] <- c(1.0, 0.85, 0.75)
  lambda[4:6, 2L] <- c(1.0, 0.80, 0.70)
  errors <- matrix(stats::rnorm(n * 6L), nrow = n)
  errors <- sweep(errors, 2L, sqrt(c(0.35, 0.45, 0.55, 0.40, 0.50, 0.60)), "*")
  means <- c(0.30, -0.10, 0.20, 0.50, 0, -0.20)
  out <- sweep(latent %*% t(lambda) + errors, 2L, means, "+")
  out <- as.data.frame(out)
  names(out) <- c(paste0("y", 1:3), paste0("x", 1:3))
  out
}

population_moments <- function(context) {
  if (identical(context$data_structure, "one_factor")) {
    lambda <- c(1.0, 0.85, 0.75, 0.90)
    sigma <- tcrossprod(lambda) +
      diag(c(context$theta_min, 0.45, 0.55, 0.35))
    return(list(sigma = sigma, mu = c(0.20, -0.10, 0.40, 0.80)))
  }
  if (identical(context$data_structure, "two_factor")) {
    psi <- matrix(
      c(1, 1 - context$psi_min, 1 - context$psi_min, 1), 2L, 2L
    )
    theta <- diag(c(0.50, 0.40, 0.40, 0.45, 0.55, 0.35))
    theta[2L, 3L] <- theta[3L, 2L] <- 0.40 - context$theta_min
    lambda <- matrix(0, 6L, 2L)
    lambda[1:3, 1L] <- c(1.0, 0.80, 0.70)
    lambda[4:6, 2L] <- c(1.0, 0.85, 0.75)
    return(list(
      sigma = lambda %*% psi %*% t(lambda) + theta,
      mu = c(0.20, -0.10, 0.40, 0.80, -0.30, 0.10)
    ))
  }
  if (identical(context$data_structure, "latent_regression")) {
    latent_covariance <- matrix(
      c(0.36 + context$psi_min, 0.60, 0.60, 1), 2L, 2L
    )
    lambda <- matrix(0, 6L, 2L)
    lambda[1:3, 1L] <- c(1.0, 0.85, 0.75)
    lambda[4:6, 2L] <- c(1.0, 0.80, 0.70)
    return(list(
      sigma = lambda %*% latent_covariance %*% t(lambda) +
        diag(c(0.35, 0.45, 0.55, 0.40, 0.50, 0.60)),
      mu = c(0.30, -0.10, 0.20, 0.50, 0, -0.20)
    ))
  }
  if (identical(context$data_structure, "two_group_one_factor")) {
    return(two_group_population_moments(context$theta_min)[c("sigma", "mu")])
  }
  stop("unknown data structure: ", context$data_structure, call. = FALSE)
}

simulate_geometry_data <- function(context, seed) {
  set.seed(seed)
  switch(
    context$data_structure,
    one_factor = simulate_one_factor_data(context$n, context$theta_min),
    two_factor = simulate_two_factor_data(
      context$n, context$theta_min, context$psi_min
    ),
    latent_regression = simulate_latent_regression_data(
      context$n, context$psi_min
    ),
    two_group_one_factor = simulate_two_group_one_factor_data(
      context$n_per_group, context$theta_min
    ),
    stop("unknown data structure: ", context$data_structure, call. = FALSE)
  )
}

apply_missingness <- function(data, seed) {
  set.seed(seed + 7919L)
  n <- nrow(data)
  out <- data
  out$x2[stats::runif(n) < 0.16] <- NA_real_
  out$x3[stats::runif(n) < 0.12] <- NA_real_
  p4 <- stats::plogis(-1.8 + 0.7 * as.numeric(scale(data$x1)))
  out$x4[stats::runif(n) < p4] <- NA_real_
  out
}

ordinalize <- function(data) {
  out <- lapply(data, function(x) {
    ordered(cut(x, c(-Inf, -0.5, 0.5, Inf), labels = FALSE))
  })
  as.data.frame(out)
}

mixed_ordinalize <- function(data) {
  out <- data
  out$x3 <- ordered(cut(out$x3, c(-Inf, -0.5, 0.5, Inf), labels = FALSE))
  out$x4 <- ordered(cut(out$x4, c(-Inf, -0.5, 0.5, Inf), labels = FALSE))
  out
}

continuous_spec <- function(structure = "one_factor") {
  if (identical(structure, "two_group_one_factor")) {
    return(model_spec(
      model_syntax(structure), meanstructure = TRUE,
      group = "group", group_labels = c("g1", "g2")
    ))
  }
  model_spec(model_syntax(structure), meanstructure = TRUE)
}

ordinal_spec <- function(parameterization = "delta") {
  model_spec(
    model_syntax(), ordered = paste0("x", seq_len(4L)),
    parameterization = parameterization
  )
}

mixed_spec <- function(parameterization = "delta") {
  model_spec(
    model_syntax(), ordered = c("x3", "x4"),
    parameterization = parameterization, meanstructure = TRUE
  )
}

fixed_wls_weight <- function(sample_stats, condition = 100) {
  lapply(seq_along(sample_stats$S), function(block) {
    p <- nrow(sample_stats$S[[block]])
    q <- if (length(sample_stats$mean) &&
             length(sample_stats$mean[[block]])) p else 0L
    q <- q + p * (p + 1L) / 2L
    diag(exp(seq(log(1), log(condition), length.out = q)))
  })
}

task_grid <- function(profile, reps, families = NULL, geometries = NULL) {
  catalog <- geometry_catalog(profile)
  if (!is.null(geometries)) {
    unknown <- setdiff(geometries, catalog$geometry)
    if (length(unknown)) {
      stop("unknown geometries: ", paste(unknown, collapse = ", "),
           call. = FALSE)
    }
    catalog <- catalog[catalog$geometry %in% geometries, , drop = FALSE]
  }
  family_catalog <- data.frame(
    family = c(
      "continuous", "fiml", "ml2s", "ordinal_delta", "ordinal_theta",
      "mixed_delta", "mixed_theta", "catml_nonpd"
    ),
    parameterization = c(
      "continuous", "continuous", "continuous", "delta", "theta",
      "delta", "theta", "delta"
    ),
    stringsAsFactors = FALSE
  )
  if (identical(profile, "pilot")) {
    family_catalog <- family_catalog[
      family_catalog$family == "continuous", , drop = FALSE
    ]
  }
  if (!is.null(families)) {
    unknown <- setdiff(families, family_catalog$family)
    if (length(unknown)) {
      stop("families unavailable in ", profile, " profile: ",
           paste(unknown, collapse = ", "), call. = FALSE)
    }
    family_catalog <- family_catalog[
      family_catalog$family %in% families, , drop = FALSE]
  }

  grid <- merge(family_catalog, catalog, by = NULL, sort = FALSE)
  if (identical(profile, "smoke")) {
    grid <- grid[!(grid$family %in% c("ordinal_theta", "mixed_theta") &
                     grid$geometry != "interior"), , drop = FALSE]
    grid <- grid[grid$family != "catml_nonpd" |
                   grid$geometry == "interior", , drop = FALSE]
  }
  grid <- merge(
    grid, data.frame(rep = seq_len(reps)), by = NULL, sort = FALSE
  )
  grid <- grid[order(grid$family, grid$geometry, grid$rep), , drop = FALSE]
  grid$task_id <- paste(grid$family, grid$geometry, grid$rep, sep = "__")
  rownames(grid) <- NULL
  grid
}
