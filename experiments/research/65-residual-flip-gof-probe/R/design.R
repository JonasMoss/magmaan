# Focused single-group GOF design using the Foldnes--Moss--Gronneberg severe
# marginal construction. This is an illustrative consumer, not a replication.

.gof_loadings <- c(
  .8, .9, 1.1, 1.4, .7, 1.4, 1.4, 1.2, 1.1, .6,
  .7, .7, 1.2, .9, 1.3, 1, 1.2, 1.5, .9, 1.3)

gof_population <- function(p, truth = c("null", "power"), rho = .25) {
  truth <- match.arg(truth)
  stopifnot(p %in% c(5L, 20L), is.finite(rho), abs(rho) < 1)
  lambda <- .gof_loadings[seq_len(p)]
  theta <- diag(p)
  if (truth == "power") theta[1L, 2L] <- theta[2L, 1L] <- rho
  Sigma <- tcrossprod(lambda) + theta
  list(p = p, truth = truth, lambda = lambda, theta = theta, Sigma = Sigma,
       rho = if (truth == "power") rho else 0,
       df = as.integer(p * (p - 3L) / 2L))
}

gof_model_spec <- function(p) {
  magmaan::model_spec(
    paste0("f =~ ", paste0("x", seq_len(p), collapse = " + ")),
    std_lv = FALSE, meanstructure = FALSE)
}

gof_design_grid <- function() {
  out <- expand.grid(
    p = c(5L, 20L), n = c(100L, 400L),
    distribution = c("normal", "pl"), truth = c("null", "power"),
    stringsAsFactors = FALSE)
  out$df <- as.integer(out$p * (out$p - 3L) / 2L)
  out$n_over_df <- out$n / out$df
  out$rho <- ifelse(out$truth == "power", .25, 0)
  out$cell_id <- seq_len(nrow(out))
  out
}

gof_validate_design <- function() {
  grid <- gof_design_grid()
  stopifnot(nrow(grid) == 16L, setequal(unique(grid$df), c(5L, 170L)))
  for (p in c(5L, 20L)) for (truth in c("null", "power")) {
    pop <- gof_population(p, truth)
    stopifnot(min(eigen(pop$theta, symmetric = TRUE,
                        only.values = TRUE)$values) > 0,
              min(eigen(pop$Sigma, symmetric = TRUE,
                        only.values = TRUE)$values) > 0)
  }
  invisible(TRUE)
}

gof_make_sampler <- function(pop, n_max, reps, distribution, seed_base) {
  begin <- proc.time()[["elapsed"]]
  p <- pop$p
  if (distribution == "normal") {
    L <- chol(pop$Sigma)
    draws <- lapply(seq_len(reps), function(rep_id) {
      set.seed(seed_base + rep_id)
      matrix(stats::rnorm(n_max * p), n_max, p) %*% L
    })
    calibration <- NULL
  } else if (distribution == "pl") {
    calibration <- magmaan:::sim_plsim_calibrate_impl(
      stats::cov2cor(pop$Sigma), rep(3, p), rep(21, p),
      method = "hermite_then_rectangle", num_segments = 12L,
      quadrature_points = 31L, hermite_order = 24L)
    batch <- magmaan:::sim_plsim_draw_impl(
      calibration, n = n_max, reps = reps, seed_base = seed_base)
    sds <- sqrt(diag(pop$Sigma))
    draws <- lapply(batch$draws, function(X) sweep(X, 2L, sds, "*"))
  } else {
    stop("unknown distribution: ", distribution, call. = FALSE)
  }
  varnames <- paste0("x", seq_len(p))
  draws <- lapply(draws, function(X) {
    colnames(X) <- varnames
    storage.mode(X) <- "double"
    X
  })
  list(
    setup_seconds = proc.time()[["elapsed"]] - begin,
    calibration = calibration,
    draw = function(rep_id, n) draws[[rep_id]][seq_len(n), , drop = FALSE])
}
