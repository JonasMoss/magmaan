# Non-normal marginal generators for the two-group invariance population.
#
# The marginal families are experiment 17's: Vale-Maurelli (Fleishman power
# method), the independent generator (IG, Foldnes-Olsson 2016) and piecewise
# linear (PL, Foldnes-Gronneberg 2022). t5 is dropped on purpose: it is
# symmetric and barely stresses a mean-scaled robust test. IG and PL are
# magmaan's native simulators; VM is self-contained R. Each family is calibrated
# to *each group's* implied covariance (group 2 differs through psi and through
# any truth violation), then the per-group mean is swept in. Marginal skew and
# kurtosis are central/standardized, so the mean shift leaves them unchanged.

# Distribution labels and target marginal moments (excess kurtosis). Suffix 1 =
# moderate (skew 2, kurt 7), suffix 2 = severe (skew 3, kurt 21), matching
# experiment 17. "norm" carries zero excess.
dist_moments <- list(
  norm = c(0, 0),
  vm1  = c(2, 7),  vm2 = c(3, 21),
  ig1  = c(2, 7),  ig2 = c(3, 21),
  pl1  = c(2, 7),  pl2 = c(3, 21))

dist_family <- function(dist) sub("[12]$", "", dist)

# Generator tuning knobs (experiment 17 defaults). Kept in one place so a runner
# can override them per family.
default_gen_knobs <- function() list(
  ig_root                 = "symmetric",
  ig_generator_family     = "pearson",
  ig_quadrature_points    = 81L,
  plsim_method            = "hermite_then_rectangle",
  plsim_num_segments      = 12L,
  plsim_quadrature_points = 31L,
  plsim_hermite_order     = 24L)

# ── Fleishman (1978) power-method coefficients (Vale-Maurelli) ──────────────
# Solve for (b, c, d) with a = -c so Y = a + bZ + cZ^2 + dZ^3, Z ~ N(0,1) has
# unit variance and the target marginal skewness and excess kurtosis.
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

# Vale-Maurelli setup for one target covariance: solve the intermediate
# correlation matrix and cache its Cholesky plus the marginal sds.
vm_setup <- function(Sigma, fl) {
  R <- stats::cov2cor(Sigma)
  sds <- sqrt(diag(Sigma))
  Rz <- R
  off <- which(upper.tri(R), arr.ind = TRUE)
  for (i in seq_len(nrow(off))) {
    rz <- intermediate_rho(R[off[i, 1L], off[i, 2L]], fl)
    if (is.na(rz)) stop("VM: no valid intermediate correlation", call. = FALSE)
    Rz[off[i, 1L], off[i, 2L]] <- rz
    Rz[off[i, 2L], off[i, 1L]] <- rz
  }
  if (min(eigen(Rz, symmetric = TRUE, only.values = TRUE)$values) <= 0) {
    stop("VM: intermediate correlation matrix not PD for this group", call. = FALSE)
  }
  list(Lz = chol(Rz), sds = sds)
}

vm_draw_block <- function(n, p, setup, fl) {
  Z <- matrix(stats::rnorm(n * p), n, p) %*% setup$Lz
  Y <- fl$a + fl$b * Z + fl$c * Z^2 + fl$d * Z^3
  sweep(Y, 2L, setup$sds, "*")
}

# Add the per-group mean, name the columns, and tag the group.
.finish_block <- function(X, mu, ov, g) {
  X <- sweep(X, 2L, mu, "+")
  colnames(X) <- ov
  data.frame(X, school = if (g == 1L) "A" else "B", stringsAsFactors = FALSE)
}

# Build a per-cell sampler: returns list(draw, setup_seconds). `draw(i)` yields
# the full two-group data frame for replicate i (complete data; the caller
# injects MCAR). The truth `violate` axis enters through group_moments(), so the
# sampler is specific to (dist, violate). norm/VM draw fresh per replicate from a
# per-replicate seed; IG/PL calibrate once per group and pre-draw a reps-sized
# batch (their native draw is batched), indexed by replicate.
build_cell_sampler <- function(pop, dist, violate, n_per_group, reps, seed_base,
                               core = NULL, knobs = default_gen_knobs()) {
  fam <- dist_family(dist)
  mom <- dist_moments[[dist]]
  if (is.null(mom)) stop("unknown distribution: ", dist, call. = FALSE)
  p   <- length(pop$ov)
  n_g <- if (length(n_per_group) == 1L) rep(n_per_group, 2L) else n_per_group
  gm  <- lapply(seq_len(2L), function(g) group_moments(pop, g, violate))
  t0  <- proc.time()[["elapsed"]]

  if (fam == "norm") {
    L <- lapply(gm, function(m) chol(m$Sigma))
    draw <- function(i) {
      set.seed(seed_base + i)
      do.call(rbind, lapply(seq_len(2L), function(g)
        .finish_block(matrix(stats::rnorm(n_g[g] * p), n_g[g], p) %*% L[[g]],
                      gm[[g]]$mu, pop$ov, g)))
    }
  } else if (fam == "vm") {
    fl    <- fleishman_coef(mom[[1L]], mom[[2L]])
    setup <- lapply(gm, function(m) vm_setup(m$Sigma, fl))
    draw <- function(i) {
      set.seed(seed_base + i)
      do.call(rbind, lapply(seq_len(2L), function(g)
        .finish_block(vm_draw_block(n_g[g], p, setup[[g]], fl),
                      gm[[g]]$mu, pop$ov, g)))
    }
  } else if (fam == "ig") {
    if (is.null(core)) stop("IG cells require magmaan_core", call. = FALSE)
    cal <- lapply(gm, function(m) core$sim_ig_calibrate(
      m$Sigma, rep(mom[[1L]], p), rep(mom[[2L]], p),
      root = knobs$ig_root, generator_family = knobs$ig_generator_family,
      quadrature_points = knobs$ig_quadrature_points))
    batch <- lapply(seq_len(2L), function(g) core$sim_ig_draw(
      cal[[g]], n = n_g[g], reps = reps, seed_base = seed_base + g * 1000003L,
      quadrature_points = knobs$ig_quadrature_points))
    draw <- function(i) do.call(rbind, lapply(seq_len(2L), function(g)
      .finish_block(batch[[g]]$draws[[i]], gm[[g]]$mu, pop$ov, g)))
  } else if (fam == "pl") {
    if (is.null(core)) stop("PL cells require magmaan_core", call. = FALSE)
    sds <- lapply(gm, function(m) sqrt(diag(m$Sigma)))
    cal <- lapply(gm, function(m) core$sim_plsim_calibrate(
      stats::cov2cor(m$Sigma), rep(mom[[1L]], p), rep(mom[[2L]], p),
      method = knobs$plsim_method, num_segments = knobs$plsim_num_segments,
      quadrature_points = knobs$plsim_quadrature_points,
      hermite_order = knobs$plsim_hermite_order))
    batch <- lapply(seq_len(2L), function(g) core$sim_plsim_draw(
      cal[[g]], n = n_g[g], reps = reps, seed_base = seed_base + g * 1000003L))
    draw <- function(i) do.call(rbind, lapply(seq_len(2L), function(g)
      .finish_block(sweep(batch[[g]]$draws[[i]], 2L, sds[[g]], "*"),
                    gm[[g]]$mu, pop$ov, g)))
  } else {
    stop("unknown distribution family: ", fam, call. = FALSE)
  }

  list(draw = draw, setup_seconds = proc.time()[["elapsed"]] - t0)
}
