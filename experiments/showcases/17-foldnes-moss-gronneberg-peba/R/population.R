# Population model and non-normal data generation for the
# Foldnes-Moss-Gronneberg (2024) goodness-of-fit study.
#
# "Improved Goodness of Fit Procedures for Structural Equation Models",
# Structural Equation Modeling 32(1), 1-13.
#
# The design is a correctly-specified two-factor CFA at three sizes
# (p = 10, 20, 40 observed variables; df = 34, 169, 739). The population
# loading and residual matrices are the paper's *exact* values, transcribed
# verbatim from the OSF supplement (models.R at https://osf.io/6trwu/): MODEL1,
# MODEL2 and MODEL3 are the p = 10, 20 and 40 populations. Standardized loadings
# give unit-variance indicators; the two factors have unit variance and
# correlate .5. The models are nested: MODEL2's first five loadings per factor
# equal MODEL1's, and MODEL3's first ten per factor equal MODEL2's.
#
# Marginal non-normality spans the paper's three families at two severities,
# moderate (skew 2, kurt 7) and severe (skew 3, kurt 21):
#   - VM (Vale-Maurelli 1983): self-contained via the Fleishman power method.
#   - IG (independent generator, Foldnes-Olsson 2016) and PL (piecewise linear,
#     Foldnes-Gronneberg 2022): magmaan's own simulators (core$sim_ig_*,
#     core$sim_plsim_*), so the experiment needs no covsim dependency. The paper
#     itself generates IG/PL with covsim and VM with lavaan; magmaan reproduces
#     the IG/PL families natively (see experiment 18).

# ── Fleishman (1978) power-method coefficients (Vale-Maurelli) ──────────────
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
  inits <- list(c(1, 0, 0), c(0.95, 0.05, 0.03), c(1, skew / 10, kurt / 200))
  best <- NULL
  for (init in inits) {
    o <- stats::optim(init, obj, method = "BFGS",
                      control = list(reltol = 1e-14, maxit = 5000L))
    if (is.null(best) || o$value < best$value) best <- o
  }
  list(a = -best$par[[2L]], b = best$par[[1L]],
       c = best$par[[2L]], d = best$par[[3L]], resid = best$value)
}

# Intermediate (Z-scale) correlation mapping to a target Y-scale correlation
# under identical Fleishman coefficients on both variables.
intermediate_rho <- function(rho_y, fl) {
  if (abs(rho_y) < 1e-12) return(0.0)
  co <- c(-rho_y, (fl$b + 3 * fl$d)^2, 2 * fl$c^2, 6 * fl$d^2)
  rts <- polyroot(co)
  real <- Re(rts)[abs(Im(rts)) < 1e-7]
  real <- real[real >= -1 & real <= 1]
  if (!length(real)) return(NA_real_)
  real[which.min(abs(real - rho_y))]
}

# ── Exact two-factor populations (paper OSF supplement, models.R) ────────────
# Per-factor standardized loadings and residual variances for MODEL1/2/3. Each
# loading^2 + residual = 1 (unit-variance indicators); nesting is visible in the
# leading entries (MODEL2 extends MODEL1, MODEL3 extends MODEL2).
.osf_models <- list(
  `10` = list(
    f1_load  = c(0.4, 0.5, 0.6, 0.8, 0.4),
    f1_resid = c(0.84, 0.75, 0.64, 0.36, 0.84),
    f2_load  = c(0.4, 0.7, 0.6, 0.4, 0.8),
    f2_resid = c(0.84, 0.51, 0.64, 0.84, 0.36)),
  `20` = list(
    f1_load  = c(0.4, 0.5, 0.6, 0.8, 0.4, 0.7, 0.8, 0.6, 0.6, 0.3),
    f1_resid = c(0.84, 0.75, 0.64, 0.36, 0.84, 0.51, 0.36, 0.64, 0.64, 0.91),
    f2_load  = c(0.4, 0.7, 0.6, 0.4, 0.8, 0.8, 0.4, 0.7, 0.5, 0.6),
    f2_resid = c(0.84, 0.51, 0.64, 0.84, 0.36, 0.36, 0.84, 0.51, 0.75, 0.64)),
  `40` = list(
    f1_load  = c(0.4, 0.5, 0.6, 0.8, 0.4, 0.7, 0.8, 0.6, 0.6, 0.3,
                 0.4, 0.4, 0.6, 0.5, 0.7, 0.5, 0.7, 0.8, 0.5, 0.7),
    f1_resid = c(0.84, 0.75, 0.64, 0.36, 0.84, 0.51, 0.36, 0.64, 0.64, 0.91,
                 0.84, 0.84, 0.64, 0.75, 0.51, 0.75, 0.51, 0.36, 0.75, 0.51),
    f2_load  = c(0.4, 0.7, 0.6, 0.4, 0.8, 0.8, 0.4, 0.7, 0.5, 0.6,
                 0.6, 0.4, 0.7, 0.4, 0.5, 0.7, 0.8, 0.4, 0.5, 0.3),
    f2_resid = c(0.84, 0.51, 0.64, 0.84, 0.36, 0.36, 0.84, 0.51, 0.75, 0.64,
                 0.64, 0.84, 0.51, 0.84, 0.75, 0.51, 0.36, 0.84, 0.75, 0.91))
)

build_population_2factor <- function(p, interfactor = 0.5) {
  m <- .osf_models[[as.character(p)]]
  if (is.null(m)) {
    stop("no OSF population for p = ", p, " (have 10, 20, 40)", call. = FALSE)
  }
  per_factor <- as.integer(p / 2L)
  stopifnot(length(m$f1_load) == per_factor, length(m$f2_load) == per_factor)

  std   <- c(m$f1_load, m$f2_load)
  resid <- c(m$f1_resid, m$f2_resid)
  Lambda <- matrix(0.0, p, 2L)
  Lambda[seq_len(per_factor), 1L] <- m$f1_load
  Lambda[(per_factor + 1L):p, 2L] <- m$f2_load

  Phi   <- matrix(c(1, interfactor, interfactor, 1), 2L, 2L)
  Theta <- diag(resid)
  Sigma <- Lambda %*% Phi %*% t(Lambda) + Theta  # a correlation matrix

  # Indicators are unit-variance by construction; this doubles as a guard
  # against any transcription slip in .osf_models above.
  if (max(abs(diag(Sigma) - 1)) > 1e-8) {
    stop("OSF population for p = ", p,
         " is not unit-variance; check the transcription", call. = FALSE)
  }
  list(Sigma = Sigma, L = chol(Sigma), Lambda = Lambda, Phi = Phi,
       std_loadings = std, residuals = resid, p = p, per_factor = per_factor)
}

# Correctly-specified two-factor analysis syntax (marker identification, free
# factor variances and covariance -> df 34/169/739 for p 10/20/40).
build_2factor_syntax <- function(p) {
  per_factor <- as.integer(p / 2L)
  vars <- paste0("x", seq_len(p))
  sprintf("f1 =~ %s\nf2 =~ %s",
          paste(vars[seq_len(per_factor)], collapse = " + "),
          paste(vars[(per_factor + 1L):p], collapse = " + "))
}

# ── Data sampling ───────────────────────────────────────────────────────────
dist_family <- function(dist) sub("[12]$", "", dist)

# Vale-Maurelli setup: solve the intermediate (Z-scale) correlation matrix that
# maps to `pop$Sigma` under the Fleishman coefficients, and cache its Cholesky.
# Depends only on the population and `fl`, NOT on the random draw, so it is
# computed once per cell and reused across replicates.
.vm_setup <- function(pop, fl) {
  if (is.null(fl)) {
    stop("Fleishman coefficients required for non-normal dist", call. = FALSE)
  }
  R <- stats::cov2cor(pop$Sigma)
  sds <- sqrt(diag(pop$Sigma))
  Rz <- R
  off <- which(upper.tri(R), arr.ind = TRUE)
  for (i in seq_len(nrow(off))) {
    rz <- intermediate_rho(R[off[i, 1L], off[i, 2L]], fl)
    if (is.na(rz)) stop("VM: no valid intermediate correlation", call. = FALSE)
    Rz[off[i, 1L], off[i, 2L]] <- rz
    Rz[off[i, 2L], off[i, 1L]] <- rz
  }
  if (min(eigen(Rz, symmetric = TRUE, only.values = TRUE)$values) <= 0) {
    stop("VM: intermediate correlation matrix not PD for this cell", call. = FALSE)
  }
  list(Lz = chol(Rz), sds = sds)
}

# One VM replicate from a cached setup: Fleishman polynomial on a correlated
# normal.
.vm_draw <- function(N, p, setup, fl) {
  Z <- matrix(stats::rnorm(N * p), N, p) %*% setup$Lz
  Y <- fl$a + fl$b * Z + fl$c * Z^2 + fl$d * Z^3
  sweep(Y, 2L, setup$sds, "*")
}

# Build a per-cell sampler returning a list(draw, setup_seconds, calibration).
# Cheap families (norm, VM) draw fresh per replicate from a per-replicate seed.
# IG and PL calibrate once per cell with magmaan's native simulators and store a
# short batch of generated matrices; the calibration is population-only (no N),
# so the caller can reuse it across sample sizes.
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
    setup <- .vm_setup(pop, fl)               # once per cell, not per replicate
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
    batch <- core$sim_plsim_draw(cal, n = N, reps = reps, seed_base = seed_base)
    return(list(setup_seconds = proc.time()[["elapsed"]] - t0,
                calibration = cal,
                draw = function(i) sweep(batch$draws[[i]], 2L, sds, "*")))
  }

  stop("unknown distribution family: ", dist, call. = FALSE)
}

# Save/restore RNG state so a fixed population seed does not perturb the
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
