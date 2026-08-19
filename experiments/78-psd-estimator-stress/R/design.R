model_syntax <- function() {
  "f =~ x1 + x2 + x3 + x4"
}

geometry_catalog <- function() {
  data.frame(
    geometry = c("interior", "theta_near_boundary"),
    theta_min = c(0.35, 0.001),
    stringsAsFactors = FALSE
  )
}

simulation_seed <- function(seed_base, geometry, rep) {
  geometry_index <- match(geometry, geometry_catalog()$geometry)
  as.integer(seed_base + 1009L * geometry_index + rep)
}

simulate_cfa_data <- function(n, theta_min, seed) {
  set.seed(seed)
  eta <- stats::rnorm(n)
  loadings <- c(1.0, 0.85, 0.75, 0.90)
  residual_variances <- c(theta_min, 0.45, 0.55, 0.35)
  means <- c(0.20, -0.10, 0.40, 0.80)
  errors <- matrix(stats::rnorm(n * 4L), nrow = n, ncol = 4L)
  errors <- sweep(errors, 2L, sqrt(residual_variances), "*")
  out <- sweep(tcrossprod(eta, loadings) + errors, 2L, means, "+")
  out <- as.data.frame(out)
  names(out) <- paste0("x", seq_len(4L))
  out
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

continuous_spec <- function() {
  model_spec(model_syntax(), meanstructure = TRUE)
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
  p <- nrow(sample_stats$S[[1L]])
  q <- if (length(sample_stats$mean) && length(sample_stats$mean[[1L]])) p else 0L
  q <- q + p * (p + 1L) / 2L
  list(diag(exp(seq(log(1), log(condition), length.out = q))))
}

task_grid <- function(reps, families = NULL, geometries = NULL) {
  catalog <- geometry_catalog()
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
    n = c(120L, 120L, 120L, 180L, 180L, 180L, 180L, 180L),
    stringsAsFactors = FALSE
  )
  if (!is.null(families)) {
    unknown <- setdiff(families, family_catalog$family)
    if (length(unknown)) {
      stop("unknown families: ", paste(unknown, collapse = ", "),
           call. = FALSE)
    }
    family_catalog <- family_catalog[
      family_catalog$family %in% families, , drop = FALSE]
  }

  grid <- merge(
    family_catalog, catalog, by = NULL, sort = FALSE
  )
  grid <- grid[!(grid$family %in% c("ordinal_theta", "mixed_theta") &
                   grid$geometry != "interior"), , drop = FALSE]
  grid <- grid[grid$family != "catml_nonpd" |
                 grid$geometry == "interior", , drop = FALSE]
  grid <- merge(
    grid, data.frame(rep = seq_len(reps)), by = NULL, sort = FALSE
  )
  grid <- grid[order(grid$family, grid$geometry, grid$rep), , drop = FALSE]
  grid$task_id <- paste(grid$family, grid$geometry, grid$rep, sep = "__")
  rownames(grid) <- NULL
  grid
}
