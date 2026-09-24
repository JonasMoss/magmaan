# Posterior sampling over the admissible (covariance-honest) set.
#
# Random-walk Metropolis in standardized-latent coordinates with a flat prior
# there, several chains from dispersed starts. The proposal covariance starts
# at the inverse expected information and is re-estimated twice during burn-in
# (with a Robbins-Monro scale), then frozen. The Jeffreys posterior is obtained
# by reweighting the thinned draws with sqrt(det I(phi)); the Jeffreys density
# is too expensive to evaluate at every step in R.

sampler_defaults <- function() {
  list(chains = 4L, burn = 4000L, keep = 12000L, thin = 12L,
       target_accept = 0.25, start_spread = 1.5)
}

log_post_flat <- function(st, S, n) {
  function(phi) {
    b <- build_std(st, phi)
    if (!b$ok) return(-Inf)
    loglik_sigma(b$Sigma, S, n)
  }
}

safe_chol <- function(V, q) {
  V <- (V + t(V)) / 2
  for (ridge in c(0, 1e-10, 1e-8, 1e-6, 1e-4)) {
    R <- tryCatch(chol(V + diag(ridge * max(1e-12, mean(diag(V))), q)),
                  error = function(e) NULL)
    if (!is.null(R)) return(R)
  }
  diag(sqrt(pmax(diag(V), 1e-8)), q)
}

run_chain <- function(lpost, phi0, V0, opts) {
  q <- length(phi0)
  iters <- opts$burn + opts$keep
  R <- safe_chol(V0 * 2.38^2 / q, q)
  log_scale <- 0
  phi <- phi0
  lp <- lpost(phi)
  hist <- matrix(NA_real_, iters, q)
  acc_window <- 0L
  acc_keep <- 0L
  for (t in seq_len(iters)) {
    prop <- phi + exp(log_scale) * drop(stats::rnorm(q) %*% R)
    lpp <- lpost(prop)
    if (is.finite(lpp) && log(stats::runif(1)) < lpp - lp) {
      phi <- prop
      lp <- lpp
      if (t > opts$burn) acc_keep <- acc_keep + 1L else acc_window <- acc_window + 1L
    }
    hist[t, ] <- phi
    if (t <= opts$burn && t %% 200L == 0L) {
      log_scale <- log_scale + 2 * (acc_window / 200 - opts$target_accept)
      log_scale <- min(max(log_scale, -6), 2)
      acc_window <- 0L
    }
    if (t == opts$burn %/% 2L || t == opts$burn) {
      from <- max(1L, t %/% 3L)
      V <- stats::cov(hist[from:t, , drop = FALSE])
      if (all(is.finite(V)) && min(diag(V)) > 0) {
        R <- safe_chol(V * 2.38^2 / q, q)
        log_scale <- 0
      }
    }
  }
  kept <- hist[(opts$burn + 1L):iters, , drop = FALSE]
  list(draws = kept[seq(opts$thin, nrow(kept), by = opts$thin), , drop = FALSE],
       all = kept, accept = acc_keep / opts$keep)
}

# Dispersed admissible start around phi0.
dispersed_start <- function(lpost, phi0, V0, spread) {
  R <- safe_chol(V0, length(phi0))
  for (k in seq_len(200L)) {
    cand <- phi0 + spread * drop(stats::rnorm(length(phi0)) %*% R)
    if (is.finite(lpost(cand))) return(cand)
  }
  phi0
}

split_rhat <- function(chains) {       # list of numeric vectors (equal length)
  half <- floor(length(chains[[1L]]) / 2)
  parts <- unlist(lapply(chains, function(x) list(x[seq_len(half)], x[half + seq_len(half)])),
                  recursive = FALSE)
  means <- vapply(parts, mean, numeric(1))
  vars <- vapply(parts, stats::var, numeric(1))
  W <- mean(vars)
  if (!is.finite(W) || W <= 0) return(if (stats::var(means) > 0) Inf else 1)
  Bn <- stats::var(means)
  sqrt(((half - 1) / half * W + Bn) / W)
}

batch_ess <- function(x, b = 100L) {
  k <- length(x) %/% b
  if (k < 5L) return(NA_real_)
  bm <- colMeans(matrix(x[seq_len(k * b)], b))
  v <- stats::var(x)
  vb <- stats::var(bm)
  if (!is.finite(vb) || vb <= 0) return(length(x))
  min(length(x), length(x) * v / (b * vb))
}

weighted_quantile <- function(x, w, probs) {
  o <- order(x)
  x <- x[o]
  cw <- cumsum(w[o]) / sum(w)
  vapply(probs, function(p) x[min(which(cw >= p))], numeric(1))
}

summarize_draws <- function(inv, w) {
  w <- w / sum(w)
  apply(inv, 2, function(x) {
    q <- weighted_quantile(x, w, c(0.025, 0.5, 0.975))
    c(mean = sum(w * x), median = q[[2]], lower = q[[1]], upper = q[[3]])
  })
}

# Full posterior pass for one dataset. `phi0` is an interior start, `V0` the
# inverse expected information (total, not per observation) at phi0.
posterior_fit <- function(st, S, n, phi0, key, seed, opts = sampler_defaults()) {
  set.seed(seed)
  lpost <- log_post_flat(st, S, n)
  I0 <- information_std(st, phi0)
  V0 <- tryCatch(solve(n * I0), error = function(e) NULL)
  if (is.null(V0)) V0 <- diag(1e-3, length(phi0))
  t0 <- proc.time()[["elapsed"]]
  chains <- lapply(seq_len(opts$chains), function(k) {
    start <- dispersed_start(lpost, phi0, V0, opts$start_spread)
    run_chain(lpost, start, V0, opts)
  })
  t_mcmc <- proc.time()[["elapsed"]] - t0

  # Convergence on the key invariants, computed from the unthinned kept draws.
  key_trace <- lapply(chains, function(ch) {
    sub <- ch$all[seq(1L, nrow(ch$all), by = 4L), , drop = FALSE]
    matrix(apply(sub, 1, function(ph) invariants_std(st, ph)[key]),
           ncol = length(key), byrow = TRUE)
  })
  rhat_key <- max(vapply(seq_along(key), function(j) {
    split_rhat(lapply(key_trace, function(m) m[, j]))
  }, numeric(1)))
  ess_key <- min(vapply(seq_along(key), function(j) {
    sum(vapply(key_trace, function(m) batch_ess(m[, j], 25L), numeric(1)))
  }, numeric(1)))
  rhat_phi <- max(vapply(seq_along(phi0), function(j) {
    split_rhat(lapply(chains, function(ch) ch$all[, j]))
  }, numeric(1)))

  draws <- do.call(rbind, lapply(chains, `[[`, "draws"))
  inv <- t(apply(draws, 1, function(ph) invariants_std(st, ph)))
  t1 <- proc.time()[["elapsed"]]
  lj <- apply(draws, 1, function(ph) log_jeffreys(st, ph))
  t_jeff <- proc.time()[["elapsed"]] - t1
  wj <- exp(lj - max(lj[is.finite(lj)]))
  wj[!is.finite(wj)] <- 0
  kish <- sum(wj)^2 / sum(wj^2)

  list(
    flat = summarize_draws(inv, rep(1, nrow(inv))),
    jeffreys = summarize_draws(inv, wj),
    diagnostics = data.frame(
      accept = mean(vapply(chains, `[[`, 0, "accept")),
      rhat_key = rhat_key, rhat_phi = rhat_phi, ess_key = ess_key,
      draws = nrow(draws), jeffreys_ess = kish,
      seconds_mcmc = t_mcmc, seconds_jeffreys = t_jeff))
}
