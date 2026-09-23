# Point estimators (magmaan fits) and the local truncated-normal approximation.

point_methods <- function() {
  list(
    ml = function(spec, dat) magmaan_core$fit_ml(spec, dat),
    psd = function(spec, dat) frontier_fit_ml_psd(spec, dat),
    pen_l010 = function(spec, dat) frontier_fit_ml_multiinfo(spec, dat, weight = 0.1),
    pen_l025 = function(spec, dat) frontier_fit_ml_multiinfo(spec, dat, weight = 0.25),
    pen_l050 = function(spec, dat) frontier_fit_ml_multiinfo(spec, dat, weight = 0.5),
    pen_l100 = function(spec, dat) frontier_fit_ml_multiinfo(spec, dat, weight = 1))
}

posterior_methods <- c("flat_mean", "flat_median", "jeff_mean", "jeff_median")

all_methods <- function() c(names(point_methods()), "trunc_normal", posterior_methods)

boundary_tol <- 1e-6

classify_fit <- function(fit) {
  adm <- fit$diagnostics$admissibility
  eig <- c(vapply(adm$theta, function(b) b$min_eigenvalue %||% NA_real_, numeric(1)),
           vapply(adm$psi, function(b) b$min_eigenvalue %||% NA_real_, numeric(1)))
  if (!isTRUE(adm$admissible)) "improper"
  else if (min(eig, na.rm = TRUE) <= boundary_tol) "boundary"
  else "interior"
}

`%||%` <- function(x, y) if (is.null(x)) y else x

run_point <- function(fun, spec, dat, st) {
  t0 <- proc.time()[["elapsed"]]
  fit <- tryCatch(suppressWarnings(fun(spec, dat)), error = function(e) e)
  sec <- proc.time()[["elapsed"]] - t0
  if (inherits(fit, "error")) {
    return(list(fit = NULL, status = "error", converged = NA, class = NA_character_,
                inv = NULL, seconds = sec))
  }
  pt <- fit$partable
  mats <- marker_builder(pt, st)(pt$est[free_order(pt)])
  list(fit = fit, status = "ok", converged = isTRUE(fit$converged),
       class = classify_fit(fit), inv = invariants_marker(st, mats), seconds = sec)
}

# Mean of the invariants over N(theta_ML, V_ML) cut to the admissible set in
# magmaan's marker coordinates. Uses no likelihood evaluations.
trunc_normal <- function(ml_fit, st, draws = 4000L, max_draws = 40000L, min_keep = 1000L) {
  if (is.null(ml_fit) || !isTRUE(ml_fit$converged)) return(NULL)
  pt <- ml_fit$partable
  mu <- pt$est[free_order(pt)]
  V <- tryCatch(
    magmaan_core$inference_vcov(magmaan_core$inference_information_expected(ml_fit), ml_fit),
    error = function(e) NULL)
  if (is.null(V) || any(!is.finite(V))) return(NULL)
  R <- tryCatch(chol((V + t(V)) / 2), error = function(e) NULL)
  if (is.null(R)) return(NULL)
  build <- marker_builder(pt, st)
  kept <- list()
  n_kept <- 0L
  total <- 0L
  while (n_kept < min_keep && total < max_draws) {
    Z <- matrix(stats::rnorm(draws * length(mu)), draws) %*% R
    Th <- sweep(Z, 2, mu, `+`)
    total <- total + draws
    for (i in seq_len(draws)) {
      mats <- build(Th[i, ])
      if (!marker_admissible(mats)) next
      inv <- invariants_marker(st, mats)
      if (is.null(inv)) next
      n_kept <- n_kept + 1L
      kept[[n_kept]] <- inv
    }
  }
  if (n_kept < 100L) return(NULL)
  inv <- do.call(rbind, kept)
  list(inv = colMeans(inv), admissible_share = n_kept / total)
}
