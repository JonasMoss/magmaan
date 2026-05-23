convergence_sim_catalog <- function() {
  data.frame(
    design = c(
      "dejonckere_simple_2025",
      "dejonckere_simple_2022",
      "dejonckere_shrinkage_2023_study1",
      "dejonckere_shrinkage_2023_study2_6",
      "dejonckere_shrinkage_2023_study2_14",
      "dejonckere_shrinkage_2023_study2_18",
      "dejonckere_crossloading_2025",
      "dejonckere_crossloading_2022",
      "dejonckere_msst_2025",
      "ludtke_cfa_2021"
    ),
    reference = c(
      "De Jonckere and Rosseel (2025), Study 1",
      "De Jonckere and Rosseel (2022), Study 1",
      "De Jonckere and Rosseel (2023), Study 1",
      "De Jonckere and Rosseel (2023), Study 2, 6 observed variables",
      "De Jonckere and Rosseel (2023), Study 2, 14 observed variables",
      "De Jonckere and Rosseel (2023), Study 2, 18 observed variables",
      "De Jonckere and Rosseel (2025), Study 2",
      "De Jonckere and Rosseel (2022), Study 2",
      "De Jonckere and Rosseel (2025), Study 3",
      "Ludtke, Ulitzsch, and Robitzsch (2021), Simulation Study 1"
    ),
    default_n = c(20L, 20L, 20L, 20L, 20L, 20L, 20L, 20L, 10L, 30L),
    notes = c(
      "Two-factor latent regression, beta = 0.1, reliability-0.1 indicators.",
      "Two-factor latent regression, beta = 0.25, symmetric loadings.",
      "Two-factor latent regression, beta in {0.1, 0.3}; shrinkage-target paper setup.",
      "Two-factor latent regression, beta in {0, 0.25, 1, 2}; three indicators per factor.",
      "Two-factor latent regression, beta in {0, 0.25, 1, 2}; seven indicators per factor.",
      "Two-factor latent regression, beta in {0, 0.25, 1, 2}; nine indicators per factor.",
      "Three-factor chain SEM with three crossloadings and small latent residual variances.",
      "Three-factor chain SEM with three crossloadings and beta = c(0.6, 0.6).",
      "Second-order MSST helper using the OSF loadings and latent variance defaults.",
      "Two-factor CFA, standardized loading 0.5, rho in {0.3, 0.5, 0.7}."
    ),
    stringsAsFactors = FALSE
  )
}

convergence_sim <- function(design = convergence_sim_catalog()$design, ...) {
  design <- match.arg(design)
  switch(
    design,
    dejonckere_simple_2025 =
      convergence_sim_dejonckere_simple(variant = "2025", ...),
    dejonckere_simple_2022 =
      convergence_sim_dejonckere_simple(variant = "2022", ...),
    dejonckere_shrinkage_2023_study1 =
      convergence_sim_dejonckere_shrinkage(study = "study1", ...),
    dejonckere_shrinkage_2023_study2_6 =
      convergence_sim_dejonckere_shrinkage(study = "study2", indicators = 3L, ...),
    dejonckere_shrinkage_2023_study2_14 =
      convergence_sim_dejonckere_shrinkage(study = "study2", indicators = 7L, ...),
    dejonckere_shrinkage_2023_study2_18 =
      convergence_sim_dejonckere_shrinkage(study = "study2", indicators = 9L, ...),
    dejonckere_crossloading_2025 =
      convergence_sim_dejonckere_crossloading(variant = "2025", ...),
    dejonckere_crossloading_2022 =
      convergence_sim_dejonckere_crossloading(variant = "2022", ...),
    dejonckere_msst_2025 = convergence_sim_dejonckere_msst(...),
    ludtke_cfa_2021 = convergence_sim_ludtke_cfa(...)
  )
}

convergence_sim_dejonckere_simple <- function(n = 20L,
                                              variant = c("2025", "2022"),
                                              beta = NULL,
                                              seed = NULL) {
  variant <- match.arg(variant)
  if (is.null(beta)) beta <- if (identical(variant, "2025")) 0.1 else 0.25
  n <- .conv_positive_int(n, "n")
  beta <- .conv_scalar_numeric(beta, "beta")

  lambda_x <- c(1.0, 0.8, 0.6)
  lambda_y <- if (identical(variant, "2025")) c(1.0, 0.7, 0.9) else c(1.0, 0.8, 0.6)
  latent_cov <- matrix(c(1, beta, beta, 1 + beta^2), 2L, 2L)
  lambda <- matrix(0, nrow = 6L, ncol = 2L)
  lambda[1:3, 1L] <- lambda_x
  lambda[4:6, 2L] <- lambda_y
  residual_var <- if (identical(variant, "2025")) {
    c((1 - 0.1) * 1.0 * lambda_x^2 / 0.1,
      (1 - 0.1) * (1 + beta^2) * lambda_y^2 / 0.1)
  } else {
    rep(1, 6L)
  }
  theta <- diag(residual_var, 6L)
  ov_names <- c(paste0("x", 1:3), paste0("y", 1:3))
  sigma <- lambda %*% latent_cov %*% t(lambda) + theta
  dimnames(sigma) <- list(ov_names, ov_names)

  analysis <- paste(
    "X =~ x1 + x2 + x3",
    "Y =~ y1 + y2 + y3",
    "Y ~ X",
    sep = "\n"
  )
  residual_rows <- paste(
    paste0(ov_names, " ~~ ", format(residual_var, scientific = FALSE), "*", ov_names),
    collapse = "\n"
  )
  population <- paste(
    sprintf("X =~ 1*x1 + %.15g*x2 + %.15g*x3", lambda_x[2L], lambda_x[3L]),
    sprintf("Y =~ 1*y1 + %.15g*y2 + %.15g*y3", lambda_y[2L], lambda_y[3L]),
    sprintf("Y ~ %.15g*X", beta),
    "X ~~ 1*X",
    "Y ~~ 1*Y",
    residual_rows,
    sep = "\n"
  )
  .conv_pack(
    design = paste0("dejonckere_simple_", variant),
    reference = paste0("De Jonckere and Rosseel (", if (variant == "2025") "2025" else "2022", "), Study 1"),
    n = n,
    ov_names = ov_names,
    sigma = sigma,
    analysis_syntax = analysis,
    population_syntax = population,
    seed = seed,
    truth = list(beta = beta, lambda_x = lambda_x, lambda_y = lambda_y,
                 latent_cov = latent_cov, residual_var = residual_var)
  )
}

convergence_sim_dejonckere_crossloading <- function(n = 20L,
                                                    variant = c("2025", "2022"),
                                                    fit_model = c("correct",
                                                                  "omit_lambda8",
                                                                  "omit_lambda8_lambda9",
                                                                  "omit_all_crossloadings"),
                                                    beta = NULL,
                                                    seed = NULL) {
  variant <- match.arg(variant)
  fit_model <- match.arg(fit_model)
  n <- .conv_positive_int(n, "n")
  if (is.null(beta)) {
    beta <- if (identical(variant, "2025")) c(0.1, 0.2) else c(0.6, 0.6)
  }
  beta <- as.numeric(beta)
  if (length(beta) != 2L || any(!is.finite(beta))) {
    stop("`beta` must be a finite numeric vector of length 2", call. = FALSE)
  }

  B <- matrix(0, 3L, 3L)
  B[2L, 1L] <- beta[1L]
  B[3L, 2L] <- beta[2L]
  A <- solve(diag(3L) - B)
  latent_residual_var <- if (identical(variant, "2025")) {
    c(0.49, 0.3136, 0.3136)
  } else {
    rep(1, 3L)
  }
  psi <- diag(latent_residual_var, 3L)
  latent_cov <- A %*% psi %*% t(A)

  lambda <- matrix(0, nrow = 9L, ncol = 3L)
  lambda[1:3, 1L] <- 1
  lambda[4:6, 2L] <- 1
  lambda[7:9, 3L] <- 1
  lambda[4L, 1L] <- 0.3
  lambda[7L, 2L] <- 0.3
  lambda[6L, 3L] <- 0.3
  residual_var <- c(0.51, 0.51, 0.51, 0.2895, 0.51, 0.2895, 0.2895, 0.51, 0.51)
  ov_names <- paste0("y", 1:9)
  sigma <- lambda %*% latent_cov %*% t(lambda) + diag(residual_var, 9L)
  dimnames(sigma) <- list(ov_names, ov_names)

  analysis <- .conv_crossloading_syntax(fit_model)
  latent_variance_rows <- paste(
    paste0(paste0("eta", 1:3), " ~~ ",
           format(latent_residual_var, scientific = FALSE), "*",
           paste0("eta", 1:3)),
    collapse = "\n"
  )
  residual_rows <- paste(
    paste0(ov_names, " ~~ ", format(residual_var, scientific = FALSE), "*", ov_names),
    collapse = "\n"
  )
  population <- paste(
    "eta1 =~ 1*y1 + 1*y2 + 1*y3 + 0.3*y4",
    "eta2 =~ 1*y4 + 1*y5 + 1*y6 + 0.3*y7",
    "eta3 =~ 1*y7 + 1*y8 + 1*y9 + 0.3*y6",
    sprintf("eta2 ~ %.15g*eta1", beta[1L]),
    sprintf("eta3 ~ %.15g*eta2", beta[2L]),
    latent_variance_rows,
    residual_rows,
    sep = "\n"
  )
  .conv_pack(
    design = paste0("dejonckere_crossloading_", variant, "_", fit_model),
    reference = paste0("De Jonckere and Rosseel (", if (variant == "2025") "2025" else "2022", "), Study 2"),
    n = n,
    ov_names = ov_names,
    sigma = sigma,
    analysis_syntax = analysis,
    population_syntax = population,
    seed = seed,
    truth = list(beta = beta, lambda = lambda, latent_cov = latent_cov,
                 residual_var = residual_var,
                 latent_residual_var = latent_residual_var,
                 fit_model = fit_model)
  )
}

convergence_sim_dejonckere_shrinkage <- function(n = 20L,
                                                 study = c("study1", "study2"),
                                                 indicators = NULL,
                                                 beta = NULL,
                                                 seed = NULL) {
  study <- match.arg(study)
  n <- .conv_positive_int(n, "n")
  if (identical(study, "study1")) {
    if (is.null(indicators)) indicators <- 3L
    if (is.null(beta)) beta <- 0.1
    reference <- "De Jonckere and Rosseel (2023), Study 1"
    design <- "dejonckere_shrinkage_2023_study1"
  } else {
    if (is.null(indicators)) indicators <- 3L
    if (is.null(beta)) beta <- 0.25
    reference <- paste0(
      "De Jonckere and Rosseel (2023), Study 2, ",
      2L * as.integer(indicators), " observed variables"
    )
    design <- paste0("dejonckere_shrinkage_2023_study2_", 2L * as.integer(indicators))
  }

  indicators <- .conv_positive_int(indicators, "indicators")
  if (!indicators %in% c(3L, 7L, 9L)) {
    stop("`indicators` must be 3, 7, or 9", call. = FALSE)
  }
  beta <- .conv_scalar_numeric(beta, "beta")
  if (identical(study, "study1") && indicators != 3L) {
    stop("Study 1 uses `indicators = 3`", call. = FALSE)
  }

  loadings <- switch(
    as.character(indicators),
    `3` = c(1.0, 0.8, 0.6),
    `7` = c(1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3),
    `9` = c(1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.3, 0.2)
  )
  y_names <- paste0("y", seq_len(indicators))
  x_names <- paste0("x", seq_len(indicators))
  ov_names <- c(y_names, x_names)

  lambda <- matrix(0, nrow = length(ov_names), ncol = 2L)
  lambda[seq_len(indicators), 1L] <- loadings
  lambda[indicators + seq_len(indicators), 2L] <- loadings
  latent_cov <- matrix(c(1 + beta^2, beta, beta, 1), 2L, 2L)
  residual_var <- rep(1, length(ov_names))
  sigma <- lambda %*% latent_cov %*% t(lambda) + diag(residual_var, length(ov_names))
  dimnames(sigma) <- list(ov_names, ov_names)

  analysis <- paste(
    paste("Y =~", paste(y_names, collapse = " + ")),
    paste("X =~", paste(x_names, collapse = " + ")),
    "Y ~ X",
    sep = "\n"
  )
  population <- paste(
    .conv_loading_row("Y", y_names, loadings),
    .conv_loading_row("X", x_names, loadings),
    sprintf("Y ~ %.15g*X", beta),
    "Y ~~ 1*Y",
    "X ~~ 1*X",
    paste(paste0(ov_names, " ~~ 1*", ov_names), collapse = "\n"),
    sep = "\n"
  )
  .conv_pack(
    design = design,
    reference = reference,
    n = n,
    ov_names = ov_names,
    sigma = sigma,
    analysis_syntax = analysis,
    population_syntax = population,
    seed = seed,
    truth = list(beta = beta, loadings = loadings, latent_cov = latent_cov,
                 residual_var = residual_var,
                 shrinkage_targets = c("identity", "constant_correlation",
                                       "constant_variance", "model_based",
                                       "ledoit_wolf"))
  )
}

convergence_sim_dejonckere_msst <- function(n = 10L,
                                            state_loadings = c(1.0, 0.5, 1.3),
                                            trait_loadings = c(1.0, 0.5, 1.3),
                                            measurement_error = c(0.6, 0.25, 0.7),
                                            trait_var = 0.6,
                                            state_residual_var = c(0.6, 0.25, 0.7),
                                            seed = NULL) {
  n <- .conv_positive_int(n, "n")
  state_loadings <- .conv_numeric_vector(state_loadings, 3L, "state_loadings")
  trait_loadings <- .conv_numeric_vector(trait_loadings, 3L, "trait_loadings")
  measurement_error <- .conv_numeric_vector(measurement_error, 3L, "measurement_error")
  state_residual_var <- .conv_numeric_vector(state_residual_var, 3L, "state_residual_var")
  trait_var <- .conv_scalar_numeric(trait_var, "trait_var")
  if (trait_var <= 0 || any(state_residual_var <= 0) || any(measurement_error <= 0)) {
    stop("MSST variance arguments must be positive", call. = FALSE)
  }

  state_cov <- trait_var * tcrossprod(trait_loadings) + diag(state_residual_var, 3L)
  lambda <- matrix(0, nrow = 9L, ncol = 3L)
  for (time in seq_len(3L)) {
    rows <- time + 3L * (0:2)
    lambda[rows, time] <- state_loadings
  }
  residual_var <- rep(measurement_error, times = 3L)
  ov_names <- as.vector(outer(paste0("y", 1:3), paste0("t", 1:3), paste0))
  sigma <- lambda %*% state_cov %*% t(lambda) + diag(residual_var, 9L)
  dimnames(sigma) <- list(ov_names, ov_names)

  analysis <- paste(
    "s1 =~ y1t1 + y2t1 + y3t1",
    "s2 =~ y1t2 + y2t2 + y3t2",
    "s3 =~ y1t3 + y2t3 + y3t3",
    "trait =~ s1 + s2 + s3",
    sep = "\n"
  )
  population <- paste(
    sprintf("s1 =~ %.15g*y1t1 + %.15g*y2t1 + %.15g*y3t1", state_loadings[1L], state_loadings[2L], state_loadings[3L]),
    sprintf("s2 =~ %.15g*y1t2 + %.15g*y2t2 + %.15g*y3t2", state_loadings[1L], state_loadings[2L], state_loadings[3L]),
    sprintf("s3 =~ %.15g*y1t3 + %.15g*y2t3 + %.15g*y3t3", state_loadings[1L], state_loadings[2L], state_loadings[3L]),
    sprintf("trait =~ %.15g*s1 + %.15g*s2 + %.15g*s3", trait_loadings[1L], trait_loadings[2L], trait_loadings[3L]),
    sep = "\n"
  )
  .conv_pack(
    design = "dejonckere_msst_2025",
    reference = "De Jonckere and Rosseel (2025), Study 3",
    n = n,
    ov_names = ov_names,
    sigma = sigma,
    analysis_syntax = analysis,
    population_syntax = population,
    seed = seed,
    truth = list(state_loadings = state_loadings,
                 trait_loadings = trait_loadings,
                 measurement_error = measurement_error,
                 trait_var = trait_var,
                 state_residual_var = state_residual_var,
                 state_cov = state_cov,
                 note = paste(
                   "Defaults use the OSF script's state and trait loading",
                   "values plus its theta1/eta residual variance values.",
                   "The OSF syntax labels observed residual variances rather",
                   "than fixing them, so measurement_error remains explicit."
                 ))
  )
}

convergence_sim_ludtke_cfa <- function(n = 30L,
                                       rho = 0.3,
                                       loading = 0.5,
                                       seed = NULL) {
  n <- .conv_positive_int(n, "n")
  rho <- .conv_scalar_numeric(rho, "rho")
  loading <- .conv_scalar_numeric(loading, "loading")
  if (abs(rho) >= 1) stop("`rho` must be between -1 and 1", call. = FALSE)
  if (loading <= 0 || loading >= 1) {
    stop("`loading` must be between 0 and 1 for the standardized CFA design",
         call. = FALSE)
  }

  lambda <- matrix(0, nrow = 6L, ncol = 2L)
  lambda[1:3, 1L] <- loading
  lambda[4:6, 2L] <- loading
  latent_cov <- matrix(c(1, rho, rho, 1), 2L, 2L)
  residual_var <- rep(1 - loading^2, 6L)
  ov_names <- c(paste0("x", 1:3), paste0("y", 1:3))
  sigma <- lambda %*% latent_cov %*% t(lambda) + diag(residual_var, 6L)
  dimnames(sigma) <- list(ov_names, ov_names)

  analysis <- paste(
    "f1 =~ x1 + x2 + x3",
    "f2 =~ y1 + y2 + y3",
    "f1 ~~ f2",
    sep = "\n"
  )
  residual_rows <- paste(
    paste0(ov_names, " ~~ ", format(residual_var, scientific = FALSE), "*", ov_names),
    collapse = "\n"
  )
  population <- paste(
    sprintf("f1 =~ %.15g*x1 + %.15g*x2 + %.15g*x3", loading, loading, loading),
    sprintf("f2 =~ %.15g*y1 + %.15g*y2 + %.15g*y3", loading, loading, loading),
    sprintf("f1 ~~ %.15g*f2", rho),
    "f1 ~~ 1*f1",
    "f2 ~~ 1*f2",
    residual_rows,
    sep = "\n"
  )
  .conv_pack(
    design = "ludtke_cfa_2021",
    reference = "Ludtke, Ulitzsch, and Robitzsch (2021), Simulation Study 1",
    n = n,
    ov_names = ov_names,
    sigma = sigma,
    analysis_syntax = analysis,
    population_syntax = population,
    seed = seed,
    truth = list(rho = rho, loading = loading, latent_cov = latent_cov,
                 residual_var = residual_var)
  )
}

.conv_crossloading_syntax <- function(fit_model) {
  eta1 <- "eta1 =~ y1 + y2 + y3"
  eta2 <- "eta2 =~ y4 + y5 + y6"
  eta3 <- "eta3 =~ y7 + y8 + y9"
  if (fit_model != "omit_all_crossloadings") eta1 <- paste(eta1, "+ y4")
  if (!fit_model %in% c("omit_lambda8", "omit_lambda8_lambda9",
                        "omit_all_crossloadings")) {
    eta2 <- paste(eta2, "+ y7")
  }
  if (!fit_model %in% c("omit_lambda8_lambda9", "omit_all_crossloadings")) {
    eta3 <- paste(eta3, "+ y6")
  }
  paste(eta1, eta2, eta3, "eta2 ~ eta1", "eta3 ~ eta2", sep = "\n")
}

.conv_loading_row <- function(factor, ov_names, loadings) {
  terms <- paste0(format(loadings, scientific = FALSE), "*", ov_names)
  paste(factor, "=~", paste(terms, collapse = " + "))
}

.conv_pack <- function(design, reference, n, ov_names, sigma, analysis_syntax,
                       population_syntax, seed, truth) {
  sigma <- .conv_near_pd(sigma)
  mean <- stats::setNames(rep(0, length(ov_names)), ov_names)
  data <- .conv_simulate_mvn(n, mean = mean, cov = sigma, seed = seed)
  S <- stats::cov(data)
  sample_mean <- colMeans(data)
  out <- list(
    design = design,
    reference = reference,
    n = n,
    seed = seed,
    analysis_syntax = analysis_syntax,
    population_syntax = population_syntax,
    data = data,
    sample_stats = list(S = list(S), nobs = as.integer(n), mean = list(sample_mean)),
    population_cov = sigma,
    population_mean = mean,
    truth = truth
  )
  class(out) <- c("magmaan_convergence_sim", "list")
  out
}

print.magmaan_convergence_sim <- function(x, ...) {
  cat("<magmaan_convergence_sim>\n", sep = "")
  cat("  design:    ", x$design, "\n", sep = "")
  cat("  reference: ", x$reference, "\n", sep = "")
  cat("  n:         ", x$n, "\n", sep = "")
  cat("  variables: ", length(x$population_mean), "\n", sep = "")
  if (!is.null(x$seed)) cat("  seed:      ", x$seed, "\n", sep = "")
  invisible(x)
}

.conv_simulate_mvn <- function(n, mean, cov, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  z <- matrix(stats::rnorm(n * length(mean)), nrow = n)
  x <- sweep(z %*% chol(cov), 2L, mean, "+")
  out <- as.data.frame(x, check.names = FALSE)
  names(out) <- names(mean)
  out
}

.conv_near_pd <- function(S, eps = 1e-10) {
  S <- (S + t(S)) / 2
  eig <- eigen(S, symmetric = TRUE)
  vals <- pmax(eig$values, eps)
  out <- eig$vectors %*% diag(vals, length(vals)) %*% t(eig$vectors)
  dimnames(out) <- dimnames(S)
  (out + t(out)) / 2
}

.conv_positive_int <- function(x, name) {
  if (length(x) != 1L || !is.finite(x) || x < 1) {
    stop("`", name, "` must be a positive integer", call. = FALSE)
  }
  as.integer(x)
}

.conv_scalar_numeric <- function(x, name) {
  if (length(x) != 1L || !is.finite(x)) {
    stop("`", name, "` must be a finite scalar", call. = FALSE)
  }
  as.numeric(x)
}

.conv_numeric_vector <- function(x, len, name) {
  x <- as.numeric(x)
  if (length(x) != len || any(!is.finite(x))) {
    stop("`", name, "` must be a finite numeric vector of length ", len,
         call. = FALSE)
  }
  x
}
