# Population model and non-normal data generation for the
# Foldnes-Moss-Gronneberg (2024) goodness-of-fit study.
#
# "Improved Goodness of Fit Procedures for Structural Equation Models",
# Structural Equation Modeling 32(1), 1-13.
#
# The design is a correctly-specified two-factor CFA at three sizes
# (p = 10, 20, 40 observed variables; df = 34, 169, 739). Standardized loadings
# are drawn ~ U[.3, .8] (Li 2016) and NESTED across sizes: the p=10 loadings are
# the first five per factor of the p=20 loadings, which are the first ten of the
# p=40 loadings. Residual variances give unit-variance indicators; the two
# factors correlate .5. The paper's exact realized loadings live on OSF
# (https://osf.io/6trwu/) and are NOT reproduced here; we reconstruct a
# representative population matching the verbal description with a fixed seed.
#
# Marginal non-normality spans the paper's three families at two severities:
#   moderate (skew 2, kurt 7) and severe (skew 3, kurt 21).
#   - VM (Vale-Maurelli 1983): self-contained via Fleishman power method.
#   - IG (independent generator) / PL (piecewise linear): via covsim, as the
#     paper itself generates them.

# ── Fleishman (1978) power-method coefficients (Vale-Maurelli) ──────────────
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

# ── Two-factor nested population ────────────────────────────────────────────
# A pool of `max_per_factor` standardized loadings per factor is drawn once with
# a fixed seed; a size-p model uses the first p/2 of each factor's pool, so
# smaller models are nested in larger ones (paper, Section 2.1).
build_population_2factor <- function(p, seed = 20240701L, interfactor = 0.5,
                                     max_per_factor = 20L,
                                     loading_range = c(0.3, 0.8)) {
  stopifnot(p %% 2L == 0L)
  per_factor <- as.integer(p / 2L)
  stopifnot(per_factor <= max_per_factor)

  restore <- .Random.seed_guard()
  set.seed(seed)
  pool1 <- stats::runif(max_per_factor, loading_range[[1L]], loading_range[[2L]])
  pool2 <- stats::runif(max_per_factor, loading_range[[1L]], loading_range[[2L]])
  restore()

  std <- c(pool1[seq_len(per_factor)], pool2[seq_len(per_factor)])
  Lambda <- matrix(0.0, p, 2L)
  Lambda[seq_len(per_factor), 1L]              <- std[seq_len(per_factor)]
  Lambda[(per_factor + 1L):p, 2L]              <- std[(per_factor + 1L):p]

  Phi <- matrix(c(1, interfactor, interfactor, 1), 2L, 2L)
  Theta <- diag(1 - std^2)                       # unit-variance indicators
  Sigma <- Lambda %*% Phi %*% t(Lambda) + Theta  # a correlation matrix

  list(Sigma = Sigma, L = chol(Sigma), Lambda = Lambda, Phi = Phi,
       std_loadings = std, p = p, per_factor = per_factor)
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
# Marginal moments per distribution label (moderate = *1, severe = *2).
dist_moments <- list(
  norm = c(0, 0),
  ig1  = c(2, 7),  ig2 = c(3, 21),
  pl1  = c(2, 7),  pl2 = c(3, 21),
  vm1  = c(2, 7),  vm2 = c(3, 21))

dist_family <- function(dist) sub("[12]$", "", dist)

# Vale-Maurelli setup: solve the intermediate (Z-scale) correlation matrix that
# maps to `pop$Sigma` under the Fleishman coefficients, and cache its Cholesky.
# This depends only on the population and `fl`, NOT on the random draw, so it is
# computed once per cell and reused across replicates (the per-pair polyroot +
# the PD check are the expensive part, especially as p grows).
.vm_setup <- function(pop, fl) {
  R <- stats::cov2cor(pop$Sigma)
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
  list(Lz = chol(Rz), sds = sqrt(diag(pop$Sigma)))
}

# One VM replicate from a cached setup: Fleishman polynomial on a correlated
# normal.
.vm_draw <- function(N, p, setup, fl) {
  Z <- matrix(stats::rnorm(N * p), N, p) %*% setup$Lz
  Y <- fl$a + fl$b * Z + fl$c * Z^2 + fl$d * Z^3
  sweep(Y, 2L, setup$sds, "*")
}

# Build a per-cell sampler: a closure draw(i) returning replicate i's N x p
# matrix. Cheap families (norm, VM) draw fresh per replicate from a per-replicate
# seed; covsim families (IG, PL) carry setup cost, so the whole batch of `reps`
# datasets is generated once per cell (seeded) and indexed. If covsim cannot fit
# the requested marginals (PL severe is the usual offender), draw(i) raises a
# clear error so the runner counts the cell's replicates as failed and moves on,
# rather than aborting the whole run.
make_cell_sampler <- function(pop, N, dist, reps, seed_base, fl = NULL,
                              pl_numsegments = 12L) {
  fam <- dist_family(dist)
  if (fam == "norm") {
    return(function(i) {
      set.seed(seed_base + i)
      matrix(stats::rnorm(N * pop$p), N, pop$p) %*% pop$L
    })
  }
  if (fam == "vm") {
    setup <- .vm_setup(pop, fl)                 # once per cell, not per replicate
    return(function(i) {
      set.seed(seed_base + i)
      .vm_draw(N, pop$p, setup, fl)
    })
  }
  mom <- dist_moments[[dist]]
  skew <- rep(mom[[1L]], pop$p); kurt <- rep(mom[[2L]], pop$p)
  set.seed(seed_base)
  batch <- tryCatch(
    if (fam == "ig") {
      covsim::rIG(N, pop$Sigma, skewness = skew, excesskurtosis = kurt, reps = reps)
    } else {
      covsim::rPLSIM(N, pop$Sigma, skewness = skew, excesskurtosis = kurt,
                     reps = reps, numsegments = pl_numsegments, verbose = FALSE)
    },
    error = function(e) e)
  if (inherits(batch, "error")) {
    msg <- conditionMessage(batch)
    return(function(i) stop("covsim ", toupper(fam), " generation failed: ", msg,
                            call. = FALSE))
  }
  # covsim returns a list of `reps` matrices (rPLSIM nests them one level).
  flat <- if (fam == "pl" && is.list(batch) && is.list(batch[[1L]])) batch[[1L]] else batch
  function(i) as.matrix(flat[[i]])
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
