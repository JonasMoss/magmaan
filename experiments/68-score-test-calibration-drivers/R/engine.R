# Self-contained draw / model / test machinery for the score-test calibration
# study. Deliberately duplicates a small amount of generic multigroup-draw code
# rather than sourcing a sibling experiment (layering: no sibling-leaf edges).

# Generic non-marker loading pattern; sliced to p. Values are arbitrary but fixed.
sbd_null_loadings <- c(
  .8, .9, 1.1, 1.4, .7, 1.4, 1.4, 1.2, 1.1, .6,
  .7, .7, 1.2, .9, 1.3, 1, 1.2, 1.5, .9, 1.3)

sbd_methods <- c(
  "score_chisq", "score_sb", "score_mv", "score_ss",
  "score_peba4", "score_pall", "score_sandwich",
  "flip_basic", "flip_effective", "flip_standardized")

# ---- marginal generators (homogeneous skew/kurtosis unless overridden) --------
.sbd_calibrate_group <- function(Sigma, distribution, skew, exkurt) {
  if (distribution == "normal")
    return(list(distribution = "normal", L = chol(Sigma), Sigma = Sigma))
  cal <- switch(distribution,
    vm = magmaan:::sim_vm_calibrate_impl(stats::cov2cor(Sigma), skew, exkurt),
    ig = magmaan:::sim_ig_calibrate_impl(Sigma, skew, exkurt, root = "symmetric",
      generator_family = "pearson", quadrature_points = 81L),
    pl = magmaan:::sim_plsim_calibrate_impl(stats::cov2cor(Sigma), skew, exkurt,
      method = "hermite_then_rectangle", num_segments = 12L,
      quadrature_points = 31L, hermite_order = 24L),
    stop("unknown distribution: ", distribution, call. = FALSE))
  list(distribution = distribution, calibration = cal, Sigma = Sigma)
}

sbd_calibrate_sampler <- function(Sigma_list, distribution, skew, exkurt) {
  cache <- new.env(parent = emptyenv())
  states <- lapply(Sigma_list, function(Sigma) {
    key <- paste(formatC(as.vector(Sigma), digits = 17, format = "fg"), collapse = "|")
    if (!exists(key, envir = cache, inherits = FALSE))
      assign(key, .sbd_calibrate_group(Sigma, distribution, skew, exkurt), envir = cache)
    get(key, envir = cache, inherits = FALSE)
  })
  list(p = ncol(Sigma_list[[1L]]), groups = length(Sigma_list),
       distribution = distribution, states = states)
}

.sbd_draw_group <- function(state, n, seed) {
  p <- ncol(state$Sigma); dist <- state$distribution
  if (dist == "normal") { set.seed(seed + 1L); return(matrix(stats::rnorm(n * p), n, p) %*% state$L) }
  batch <- switch(dist,
    vm = magmaan:::sim_vm_draw_impl(state$calibration, n = n, reps = 1L, seed_base = seed),
    ig = magmaan:::sim_ig_draw_impl(state$calibration, n = n, reps = 1L, seed_base = seed,
      quadrature_points = 81L),
    pl = magmaan:::sim_plsim_draw_impl(state$calibration, n = n, reps = 1L, seed_base = seed))
  X <- batch$draws[[1L]]
  if (dist %in% c("vm", "pl")) X <- sweep(X, 2L, sqrt(diag(state$Sigma)), "*")
  X
}

sbd_draw <- function(sampler, group_sizes, seed) {
  varnames <- paste0("x", seq_len(sampler$p)); labels <- paste0("g", seq_len(sampler$groups))
  blocks <- lapply(seq_len(sampler$groups), function(g) {
    X <- .sbd_draw_group(sampler$states[[g]], group_sizes[[g]], seed + g * 1000003L)
    colnames(X) <- varnames; storage.mode(X) <- "double"; X
  })
  data <- data.frame(do.call(rbind, blocks),
    group = factor(rep(labels, times = group_sizes), levels = labels), check.names = FALSE)
  list(data = data, blocks = blocks)
}

# ---- populations ---------------------------------------------------------------
# Equal loadings across groups => the equality null is TRUE. phi (length G) and
# theta (list of length-p vectors) carry the structural information geometry.
sbd_population <- function(G, p, phi, theta) {
  base <- sbd_null_loadings[seq_len(p)]
  L <- matrix(rep(base, G), nrow = G, byrow = TRUE)
  Sigma <- lapply(seq_len(G), function(g) phi[[g]] * tcrossprod(L[g, ]) + diag(theta[[g]], p))
  pd <- vapply(Sigma, function(S) min(eigen(S, symmetric = TRUE, only.values = TRUE)$values), numeric(1))
  list(Sigma = Sigma, min_pd = min(pd), G = G, p = p)
}

# Named structural regimes used by the structure study (all keep the null true).
sbd_structure <- function(name, G, p) {
  rep_t <- function(v) lapply(seq_len(G), function(g) v)
  swap_t <- function(a, b) lapply(seq_len(G), function(g)
    if (g %% 2L == 1L) rep(c(a, b), length.out = p) else rep(c(b, a), length.out = p))
  switch(name,
    homogeneous  = sbd_population(G, p, rep(1, G),                        rep_t(rep(1, p))),
    geometry     = sbd_population(G, p, rep(c(3, .45), length.out = G),   swap_t(.4, 2.2)),
    concentrated = sbd_population(G, p, c(15, 15, rep(.2, G - 2)),        swap_t(.10, .55)),
    stop("unknown structure: ", name, call. = FALSE))
}

# ---- model specs: single factor, test equality of q loadings across G groups ---
sbd_specs <- function(G, p, q_tested) {
  labels <- paste0("g", seq_len(G))
  args <- list(syntax = paste0("f =~ ", paste0("x", seq_len(p), collapse = " + ")),
               std_lv = FALSE, meanstructure = FALSE, group = "group", group_labels = labels)
  h1 <- do.call(magmaan::model_spec, args)
  tested <- paste0("x", 2:(q_tested + 1)); released <- setdiff(paste0("x", 2:p), tested)
  h0a <- c(args, list(group_equal = "loadings"))
  if (length(released)) h0a$group_partial <- paste("f =~", released)  # VECTOR: one per released loading
  list(H1 = h1, H0 = do.call(magmaan::model_spec, h0a), df = q_tested * (G - 1))
}

# ---- one replication -> named p-values + spectrum summaries --------------------
sbd_empty <- function() stats::setNames(rep(NA_real_, length(sbd_methods)), sbd_methods)

sbd_one_rep <- function(specs, sampler, group_sizes, dgp_seed, flip_seed, flips) {
  samp <- tryCatch(sbd_draw(sampler, group_sizes, dgp_seed), error = function(e) e)
  if (inherits(samp, "error")) return(NULL)
  fits <- tryCatch(lapply(list(H1 = specs$H1, H0 = specs$H0), function(sp) magmaan::magmaan(
    sp, samp$data, estimator = "ML", optimizer = "nlopt-lbfgs-slsqp-fallback",
    se = "none", test = "none")), error = function(e) e)
  if (inherits(fits, "error")) return(NULL)
  if (!all(vapply(fits, function(x) isTRUE(x$converged), logical(1)))) return(NULL)
  flip <- tryCatch(magmaan::score_flip_test(fits$H1, fits$H0, samp$blocks,
    n_flips = flips, seed = flip_seed), error = function(e) e)
  if (inherits(flip, "error")) return(NULL)
  ev <- flip$eigenvalues
  pk <- function(md, pm = 4) tryCatch(magmaan:::infer_fmg_test(
    flip$statistic_effective, flip$df, ev, method = md, param = pm)$p_value,
    error = function(e) NA_real_)
  v <- sbd_empty()
  v[["score_chisq"]]      <- flip$p_chisq
  v[["score_sb"]]         <- flip$p_mean_scaled
  v[["score_sandwich"]]   <- flip$p_sandwich
  v[["flip_basic"]]       <- flip$p_basic
  v[["flip_effective"]]   <- flip$p_effective
  v[["flip_standardized"]] <- flip$p_standardized
  v[["score_mv"]]         <- pk("mv")
  v[["score_ss"]]         <- pk("ss")
  v[["score_peba4"]]      <- pk("peba", 4)
  v[["score_pall"]]       <- pk("penalized_all")
  data.frame(df_obs = as.integer(flip$df),
             eigen_cv = if (length(ev) > 1L) stats::sd(ev) / mean(ev) else 0,
             eigen_ratio = if (length(ev) > 1L) max(ev) / min(ev) else 1,
             as.data.frame(as.list(v)), stringsAsFactors = FALSE)
}
