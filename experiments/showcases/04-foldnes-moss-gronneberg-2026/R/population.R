# Population model and non-normal data generation for the
# Foldnes-Moss-Gronneberg (2026) Study-1 design.
#
# These helpers are experiment-local: they reconstruct a *representative*
# five-factor population matching the verbal description in the paper (loadings
# ~ U[0.5, 1.5], unit latent and error variances, interfactor correlations in
# {-.3, -.2, 0, .1, .2, .3}). The paper's exact realized population matrices
# live in the OSF supplement (https://osf.io/h2y3n/) and are NOT reproduced
# here. Marginal non-normality is induced with experiment-local Vale-Maurelli
# code for VM cells and magmaan's independent-generator and piecewise-linear
# simulators for IG/PL cells, so the experiment needs no covsim dependency.
# The paper itself generates VM/IG/PL data with lavaan/covsim.

# ── Fleishman (1978) power-method coefficients ──────────────────────────────
# Solve for (b, c, d) with a = -c so that Y = a + bZ + cZ^2 + dZ^3, Z ~ N(0,1),
# has unit variance and the target marginal skewness and excess kurtosis.
fleishman_coef <- function(skew, kurt) {
  eqs <- function(p) {
    b <- p[[1L]]; c <- p[[2L]]; d <- p[[3L]]
    c(b^2 + 6 * b * d + 2 * c^2 + 15 * d^2 - 1,
      2 * c * (b^2 + 24 * b * d + 105 * d^2 + 2) - skew,
      24 * (b * d + c^2 * (1 + b^2 + 28 * b * d) +
              d^2 * (12 + 48 * b * d + 141 * c^2 + 225 * d^2)) - kurt)
  }
  obj <- function(p) sum(eqs(p)^2)
  inits <- list(c(1, 0, 0), c(0.95, 0.05, 0.03),
                c(1, skew / 10, kurt / 200))
  best <- NULL
  for (init in inits) {
    o <- stats::optim(init, obj, method = "BFGS",
                      control = list(reltol = 1e-14, maxit = 5000L))
    if (is.null(best) || o$value < best$value) best <- o
  }
  list(a = -best$par[[2L]], b = best$par[[1L]],
       c = best$par[[2L]], d = best$par[[3L]], resid = best$value)
}

# Intermediate (Z-scale) correlation that maps to a target Y-scale correlation
# under identical Fleishman coefficients for both variables:
#   rho_y = rho_z (b + 3d)^2 + rho_z^2 (2 c^2) + rho_z^3 (6 d^2)
# Solve the cubic and take the real root in [-1, 1] closest to rho_y.
intermediate_rho <- function(rho_y, fl) {
  if (abs(rho_y) < 1e-12) return(0.0)
  co <- c(-rho_y, (fl$b + 3 * fl$d)^2, 2 * fl$c^2, 6 * fl$d^2)
  rts <- polyroot(co)
  real <- Re(rts)[abs(Im(rts)) < 1e-7]
  real <- real[real >= -1 & real <= 1]
  if (!length(real)) return(NA_real_)
  real[which.min(abs(real - rho_y))]
}

# ── Five-factor population ──────────────────────────────────────────────────
# n_factors = 5, equal indicators per factor. Loadings drawn U[0.5, 1.5] with a
# deterministic seed; latent and error variances fixed at 1. Interfactor
# correlations cycle through {-.3, -.2, 0, .1, .2, .3} over the 10 factor pairs.
build_population_5factor <- function(p, seed = 20260530L) {
  n_factors <- 5L
  stopifnot(p %% n_factors == 0L)
  per_factor <- as.integer(p / n_factors)

  withr_seed <- .Random.seed_guard()
  set.seed(seed)
  lambdas <- stats::runif(p, 0.5, 1.5)
  withr_seed()

  Lambda <- matrix(0.0, p, n_factors)
  for (f in seq_len(n_factors)) {
    rows <- ((f - 1L) * per_factor + 1L):(f * per_factor)
    Lambda[rows, f] <- lambdas[rows]
  }

  rho_vals <- c(-0.3, -0.2, 0.0, 0.1, 0.2, 0.3)
  Phi <- diag(n_factors)
  pairs <- which(upper.tri(Phi), arr.ind = TRUE)
  for (i in seq_len(nrow(pairs))) {
    v <- rho_vals[((i - 1L) %% length(rho_vals)) + 1L]
    Phi[pairs[i, 1L], pairs[i, 2L]] <- v
    Phi[pairs[i, 2L], pairs[i, 1L]] <- v
  }
  if (min(eigen(Phi, symmetric = TRUE, only.values = TRUE)$values) <= 0) {
    stop("interfactor correlation matrix is not positive definite", call. = FALSE)
  }

  Theta <- diag(p)                       # unit error variances
  Sigma <- Lambda %*% Phi %*% t(Lambda) + Theta
  std_loadings <- lambdas / sqrt(lambdas^2 + 1)
  list(Sigma = Sigma, L = chol(Sigma), Lambda = Lambda, Phi = Phi,
       lambdas = lambdas, std_loadings = std_loadings,
       p = p, per_factor = per_factor, n_factors = n_factors)
}

# lavaan-style five-factor analysis syntax (correctly specified for Study 1).
build_5factor_syntax <- function(p, per_factor = NULL) {
  if (is.null(per_factor)) per_factor <- as.integer(p / 5L)
  vars <- paste0("x", seq_len(p))
  lines <- vapply(seq_len(5L), function(f) {
    rows <- ((f - 1L) * per_factor + 1L):(f * per_factor)
    paste0("f", f, " =~ ", paste(vars[rows], collapse = " + "))
  }, character(1))
  paste(lines, collapse = "\n")
}

# ── Data sampling ───────────────────────────────────────────────────────────
dist_family <- function(dist) sub("[12]$", "", dist)

# Vale-Maurelli setup: solve the intermediate normal correlation matrix once per
# cell, then reuse it across replicates.
.vm_setup <- function(pop, fl) {
  if (is.null(fl)) stop("Fleishman coefficients required for non-normal dist",
                        call. = FALSE)
  R <- stats::cov2cor(pop$Sigma)
  sds <- sqrt(diag(pop$Sigma))
  Rz <- R
  off <- which(upper.tri(R), arr.ind = TRUE)
  for (i in seq_len(nrow(off))) {
    rz <- intermediate_rho(R[off[i, 1L], off[i, 2L]], fl)
    if (is.na(rz)) stop("no valid intermediate correlation", call. = FALSE)
    Rz[off[i, 1L], off[i, 2L]] <- rz
    Rz[off[i, 2L], off[i, 1L]] <- rz
  }
  if (min(eigen(Rz, symmetric = TRUE, only.values = TRUE)$values) <= 0) {
    stop("intermediate correlation matrix not PD; VM infeasible for this cell",
         call. = FALSE)
  }
  list(Lz = chol(Rz), sds = sds)
}

.vm_draw <- function(N, p, setup, fl) {
  Z <- matrix(stats::rnorm(N * p), N, p) %*% setup$Lz
  Y <- fl$a + fl$b * Z + fl$c * Z^2 + fl$d * Z^3   # unit-variance, target moments
  sweep(Y, 2L, setup$sds, "*")
}

# Build a per-cell sampler. Cheap families draw fresh per replicate; IG and
# PLSIM calibrate once per cell and store a short batch of generated matrices.
make_cell_sampler <- function(pop, N, dist, reps, seed_base, moments,
                              fl = NULL, core = NULL,
                              sim_calibration = NULL,
                              ig_root = "symmetric",
                              ig_generator_family = "pearson",
                              ig_quadrature_points = 81L,
                              plsim_method = "hermite_then_rectangle",
                              plsim_num_segments = 12L,
                              plsim_quadrature_points = 31L,
                              plsim_hermite_order = 24L) {
  p <- pop$p
  fam <- dist_family(dist)
  t0 <- proc.time()[["elapsed"]]

  if (fam == "norm") {
    return(list(setup_seconds = proc.time()[["elapsed"]] - t0,
                draw = function(i) {
                  set.seed(seed_base + i)
                  matrix(stats::rnorm(N * p), N, p) %*% pop$L
                }))
  }

  if (fam == "vm") {
    setup <- .vm_setup(pop, fl)
    return(list(setup_seconds = proc.time()[["elapsed"]] - t0,
                draw = function(i) {
                  set.seed(seed_base + i)
                  .vm_draw(N, p, setup, fl)
                }))
  }

  if (fam == "ig") {
    if (is.null(core)) stop("IG cells require magmaan_core", call. = FALSE)
    mom <- moments[[dist]]
    if (is.null(mom)) stop("unknown IG distribution: ", dist, call. = FALSE)
    cal <- sim_calibration
    if (is.null(cal)) {
      cal <- core$sim_ig_calibrate(
        pop$Sigma,
        rep(mom[[1L]], p),
        rep(mom[[2L]], p),
        root = ig_root,
        generator_family = ig_generator_family,
        quadrature_points = ig_quadrature_points)
    }
    batch <- core$sim_ig_draw(
      cal, n = N, reps = reps, seed_base = seed_base,
      quadrature_points = ig_quadrature_points)
    return(list(setup_seconds = proc.time()[["elapsed"]] - t0,
                calibration = cal,
                ig = cal[c("root", "generator_skewness",
                           "generator_excess_kurtosis")],
                draw = function(i) batch$draws[[i]]))
  }

  if (fam == "pl") {
    if (is.null(core)) stop("PLSIM cells require magmaan_core", call. = FALSE)
    mom <- moments[[dist]]
    if (is.null(mom)) stop("unknown PLSIM distribution: ", dist, call. = FALSE)
    sds <- sqrt(diag(pop$Sigma))
    cal <- sim_calibration
    if (is.null(cal)) {
      cal <- core$sim_plsim_calibrate(
        stats::cov2cor(pop$Sigma),
        rep(mom[[1L]], p),
        rep(mom[[2L]], p),
        method = plsim_method,
        num_segments = plsim_num_segments,
        quadrature_points = plsim_quadrature_points,
        hermite_order = plsim_hermite_order)
    }
    batch <- core$sim_plsim_draw(
      cal, n = N, reps = reps, seed_base = seed_base)
    return(list(setup_seconds = proc.time()[["elapsed"]] - t0,
                calibration = cal,
                plsim = cal[c("intermediate_corr", "achieved_corr", "iterations")],
                draw = function(i) sweep(batch$draws[[i]], 2L, sds, "*")))
  }

  stop("unknown distribution family: ", dist, call. = FALSE)
}

# Save and restore the RNG state so a fixed population seed does not perturb the
# replication stream.
.Random.seed_guard <- function() {
  had <- exists(".Random.seed", envir = globalenv(), inherits = FALSE)
  saved <- if (had) get(".Random.seed", envir = globalenv()) else NULL
  function() {
    if (had) assign(".Random.seed", saved, envir = globalenv())
    else if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
      rm(".Random.seed", envir = globalenv())
    }
    invisible(NULL)
  }
}
