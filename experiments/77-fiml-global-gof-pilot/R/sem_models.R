sem_model_one_factor <- function() {
  lambda <- c(1.00, 0.80, 0.90, 0.70, 1.10, 0.85)
  theta <- c(0.50, 0.60, 0.55, 0.65, 0.50, 0.60)
  p <- length(lambda)
  list(
    model_id = "one_factor_6",
    model_label = "One-factor CFA (6 indicators)",
    p = p,
    expected_df = 9L,
    ov = paste0("x", seq_len(p)),
    mu = rep(0, p),
    Sigma = tcrossprod(lambda) + diag(theta),
    spec = magmaan::model_spec(
      paste0("f =~ ", paste0("x", seq_len(p), collapse = " + ")),
      meanstructure = TRUE, fixed_x = FALSE)
  )
}

sem_model_two_factor_fmg <- function() {
  lambda_1 <- c(0.4, 0.5, 0.6, 0.8, 0.4)
  lambda_2 <- c(0.4, 0.7, 0.6, 0.4, 0.8)
  p <- length(lambda_1) + length(lambda_2)
  Lambda <- matrix(0, p, 2L)
  Lambda[seq_along(lambda_1), 1L] <- lambda_1
  Lambda[length(lambda_1) + seq_along(lambda_2), 2L] <- lambda_2
  Phi <- matrix(c(1, 0.5, 0.5, 1), 2L, 2L)
  residual <- 1 - rowSums(Lambda^2)
  list(
    model_id = "two_factor_fmg_10",
    model_label = "FMG two-factor CFA (10 indicators)",
    p = p,
    expected_df = 34L,
    ov = paste0("x", seq_len(p)),
    mu = rep(0, p),
    Sigma = Lambda %*% Phi %*% t(Lambda) + diag(residual),
    spec = magmaan::model_spec(paste(
      "f1 =~ x1 + x2 + x3 + x4 + x5",
      "f2 =~ x6 + x7 + x8 + x9 + x10", sep = "\n"),
      meanstructure = TRUE, fixed_x = FALSE)
  )
}

sem_model_bifactor <- function() {
  per <- 4L
  group_loading <- c(0.50, 0.40, 0.45)
  p <- per * length(group_loading)
  general_loading <- rep(0.60, p)
  group_matrix <- matrix(0, p, length(group_loading))
  blocks <- split(seq_len(p), rep(seq_along(group_loading), each = per))
  for (group in seq_along(group_loading)) {
    group_matrix[blocks[[group]], group] <- group_loading[[group]]
  }
  Lambda <- cbind(general_loading, group_matrix)
  residual <- 1 - rowSums(Lambda^2)
  ov <- paste0("x", seq_len(p))
  syntax <- paste0(
    "G =~ ", paste(ov, collapse = " + "), "\n",
    paste(vapply(seq_along(blocks), function(group) {
      paste0("g", group, " =~ ", paste(ov[blocks[[group]]], collapse = " + "))
    }, character(1L)), collapse = "\n"))
  list(
    model_id = "bifactor_12",
    model_label = "Orthogonal bifactor (12 indicators)",
    p = p,
    expected_df = 42L,
    ov = ov,
    mu = rep(0, p),
    Sigma = tcrossprod(Lambda) + diag(residual),
    spec = magmaan::model_spec(
      syntax, orthogonal = TRUE, std_lv = TRUE,
      meanstructure = TRUE, fixed_x = FALSE)
  )
}

sem_model_three_factor_15 <- function() {
  loadings <- list(
    c(0.80, 0.70, 0.60, 0.50, 0.40),
    c(0.75, 0.65, 0.55, 0.45, 0.80),
    c(0.70, 0.60, 0.50, 0.40, 0.75))
  p <- sum(lengths(loadings))
  Lambda <- matrix(0, p, 3L)
  blocks <- split(seq_len(p), rep(seq_along(loadings), lengths(loadings)))
  for (factor in seq_along(loadings)) {
    Lambda[blocks[[factor]], factor] <- loadings[[factor]]
  }
  Phi <- matrix(c(
    1.00, 0.30, 0.40,
    0.30, 1.00, 0.50,
    0.40, 0.50, 1.00), 3L, 3L)
  residual <- 1 - rowSums(Lambda^2)
  ov <- paste0("x", seq_len(p))
  syntax <- paste(vapply(seq_along(blocks), function(factor) {
    paste0("f", factor, " =~ ", paste(ov[blocks[[factor]]], collapse = " + "))
  }, character(1L)), collapse = "\n")
  list(
    model_id = "three_factor_15",
    model_label = "Three-factor CFA (15 indicators)",
    p = p,
    expected_df = 87L,
    ov = ov,
    mu = rep(0, p),
    Sigma = Lambda %*% Phi %*% t(Lambda) + diag(residual),
    spec = magmaan::model_spec(
      syntax, std_lv = TRUE, meanstructure = TRUE, fixed_x = FALSE)
  )
}

sem_model_growth <- function() {
  time <- 0:4
  Lambda <- cbind(1, time)
  Phi <- matrix(c(1.00, 0.10, 0.10, 0.10), 2L, 2L)
  alpha <- c(1.00, 0.20)
  p <- length(time)
  ov <- paste0("y", seq_len(p))
  syntax <- paste(
    paste0("i =~ ", paste0("1*", ov, collapse = " + ")),
    paste0("s =~ ", paste0(time, "*", ov, collapse = " + ")),
    sep = "\n")
  list(
    model_id = "linear_growth_5",
    model_label = "Linear growth model (5 waves)",
    p = p,
    expected_df = 10L,
    ov = ov,
    mu = as.vector(Lambda %*% alpha),
    Sigma = Lambda %*% Phi %*% t(Lambda) + diag(0.50, p),
    spec = magmaan::model_spec(
      syntax, model_type = "growth", fixed_x = FALSE)
  )
}

sem_model_catalog <- function() {
  out <- list(
    sem_model_one_factor(), sem_model_two_factor_fmg(),
    sem_model_bifactor(), sem_model_three_factor_15(), sem_model_growth())
  names(out) <- vapply(out, `[[`, character(1L), "model_id")
  for (model in out) {
    if (length(model$mu) != model$p ||
        !identical(dim(model$Sigma), c(model$p, model$p)) ||
        min(eigen(model$Sigma, symmetric = TRUE, only.values = TRUE)$values) <= 0) {
      stop("invalid SEM population: ", model$model_id, call. = FALSE)
    }
  }
  out
}

sem_distribution_moments <- function(distribution, p) {
  severity <- if (identical(distribution, "normal")) 0L else
    as.integer(sub("^[a-z]+", "", distribution))
  target <- switch(as.character(severity),
    `0` = c(skew = 0, exkurt = 0),
    `1` = c(skew = 2, exkurt = 7),
    `2` = c(skew = 3, exkurt = 21),
    stop("unknown SEM distribution: ", distribution, call. = FALSE))
  list(skew = rep(target[["skew"]], p),
       exkurt = rep(target[["exkurt"]], p))
}

sem_calibrate_sampler <- function(model, distribution) {
  begin <- proc.time()[["elapsed"]]
  moments <- sem_distribution_moments(distribution, model$p)
  kind <- sub("[12]$", "", distribution)
  sds <- sqrt(diag(model$Sigma))
  calibration <- switch(kind,
    normal = NULL,
    vm = magmaan::magmaan_core$sim_vm_calibrate(
      stats::cov2cor(model$Sigma), moments$skew, moments$exkurt),
    ig = magmaan::magmaan_core$sim_ig_calibrate(
      model$Sigma, moments$skew, moments$exkurt,
      root = "symmetric", generator_family = "pearson",
      quadrature_points = 81L),
    stop("unknown SEM generator: ", distribution, call. = FALSE))
  list(
    kind = kind,
    distribution = distribution,
    Sigma = model$Sigma,
    mu = model$mu,
    chol = if (kind == "normal") chol(model$Sigma) else NULL,
    scale = if (kind == "vm") sds else rep(1, model$p),
    calibration = calibration,
    setup_seconds = proc.time()[["elapsed"]] - begin)
}

sem_seed <- function(seed) {
  as.integer(seed %% (.Machine$integer.max - 1L))
}

sem_draw <- function(model, sampler, n, seed) {
  seed <- sem_seed(seed)
  if (sampler$kind == "normal") {
    set.seed(seed)
    X <- matrix(stats::rnorm(n * model$p), nrow = n) %*% sampler$chol
  } else {
    batch <- switch(sampler$kind,
      vm = magmaan::magmaan_core$sim_vm_draw(
        sampler$calibration, n = n, reps = 1L, seed_base = seed),
      ig = magmaan::magmaan_core$sim_ig_draw(
        sampler$calibration, n = n, reps = 1L, seed_base = seed,
        quadrature_points = 81L))
    X <- sweep(batch$draws[[1L]], 2L, sampler$scale, "*")
  }
  X <- sweep(X, 2L, sampler$mu, "+")
  colnames(X) <- model$ov
  storage.mode(X) <- "double"
  X
}

sem_apply_missingness <- function(X, mechanism, rate = 0.30) {
  if (identical(mechanism, "complete")) return(X)
  if (identical(mechanism, "mar_30")) {
    masked <- sb2005_mar(
      as.data.frame(X), rate = rate, predictors = 1:2, calibrate = TRUE)
    out <- as.matrix(masked$data)
    storage.mode(out) <- "double"
    return(out)
  }
  if (identical(mechanism, "mcar_30")) {
    mask <- matrix(stats::runif(nrow(X) * (ncol(X) - 1L)) < rate,
                   nrow(X), ncol(X) - 1L)
    eligible <- X[, -1L, drop = FALSE]
    eligible[mask] <- NA_real_
    X[, -1L] <- eligible
    return(X)
  }
  stop("unknown SEM missingness mechanism: ", mechanism, call. = FALSE)
}
