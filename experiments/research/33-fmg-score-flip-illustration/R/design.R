# Focused weak-invariance design from Foldnes, Gronneberg & Moss (2026),
# Behavior Research Methods 58:107. Exact Study 2 null loadings and the selected
# Study 3 power loadings are transcribed from https://osf.io/h2y3n/. This leaf
# intentionally owns its paper-specific constants; experiments may not source
# one another.

.fmg_null_loadings <- c(
  .8, .9, 1.1, 1.4, .7, 1.4, 1.4, 1.2, 1.1, .6,
  .7, .7, 1.2, .9, 1.3, 1, 1.2, 1.5, .9, 1.3)

.fmg_power_loadings <- list(
  `5_2` = rbind(
    c(.68, .48, .72, .72, .88),
    c(.72, .52, .68, .72, .88)),
  `5_8` = rbind(
    c(.88, .72, .62, .62, .78),
    c(.92, .72, .62, .58, .82),
    c(.92, .72, .62, .58, .82),
    c(.88, .72, .62, .58, .78),
    c(.88, .68, .62, .62, .78),
    c(.92, .72, .58, .62, .82),
    c(.88, .68, .62, .58, .82),
    c(.88, .72, .58, .58, .82)),
  `20_2` = rbind(
    c(.64, .74, .56, .76, 1.14, .56, 1.26, 1.36, .56, .74,
      1.54, .86, .76, 1.06, .74, 1.26, 1.24, .86, 1.36, .86),
    c(.56, .66, .56, .76, 1.14, .56, 1.34, 1.44, .56, .66,
      1.46, .94, .84, 1.14, .66, 1.26, 1.24, .94, 1.44, .94)),
  `20_8` = rbind(
    c(1.06, .46, .76, .76, 1.34, .84, 1.16, 1.36, 1.44, .64,
      1.26, .84, .56, 1.46, .86, 1.04, 1.46, 1.14, 1.54, 1.26),
    c(1.14, .46, .76, .84, 1.34, .76, 1.24, 1.36, 1.44, .64,
      1.34, .84, .56, 1.54, .86, 1.04, 1.54, 1.14, 1.54, 1.34),
    c(1.06, .46, .76, .76, 1.34, .84, 1.24, 1.44, 1.44, .64,
      1.34, .84, .64, 1.54, .86, 1.04, 1.54, 1.06, 1.46, 1.34),
    c(1.14, .46, .76, .76, 1.34, .84, 1.24, 1.36, 1.36, .56,
      1.34, .76, .56, 1.54, .86, 1.04, 1.54, 1.14, 1.46, 1.34),
    c(1.06, .54, .84, .76, 1.26, .84, 1.24, 1.44, 1.44, .64,
      1.26, .76, .64, 1.46, .86, 1.04, 1.46, 1.14, 1.54, 1.34),
    c(1.06, .46, .84, .76, 1.26, .84, 1.16, 1.36, 1.44, .56,
      1.26, .76, .56, 1.46, .86, .96, 1.54, 1.14, 1.54, 1.34),
    c(1.14, .54, .76, .76, 1.26, .76, 1.16, 1.36, 1.36, .56,
      1.34, .84, .56, 1.46, .86, .96, 1.46, 1.14, 1.46, 1.26),
    c(1.06, .46, .84, .76, 1.34, .84, 1.16, 1.36, 1.36, .64,
      1.26, .76, .64, 1.54, .86, 1.04, 1.46, 1.06, 1.54, 1.26)))

fmg_loading_matrix <- function(p, groups, truth = c("null", "power")) {
  truth <- match.arg(truth)
  stopifnot(p %in% c(5L, 20L), groups %in% c(2L, 8L))
  if (truth == "null") {
    return(matrix(rep(.fmg_null_loadings[seq_len(p)], groups),
                  nrow = groups, byrow = TRUE))
  }
  .fmg_power_loadings[[paste(p, groups, sep = "_")]]
}

fmg_population <- function(p, groups, truth = c("null", "power")) {
  truth <- match.arg(truth)
  loadings <- fmg_loading_matrix(p, groups, truth)
  sigmas <- lapply(seq_len(groups), function(g) {
    lambda <- loadings[g, ]
    tcrossprod(lambda) + diag(p)
  })
  list(p = p, groups = groups, truth = truth, loadings = loadings,
       Sigma = sigmas, df = as.integer((groups - 1L) * (p - 1L)))
}

fmg_analysis_syntax <- function(p) {
  paste0("f =~ ", paste0("x", seq_len(p), collapse = " + "))
}

fmg_model_specs <- function(p, groups) {
  labels <- paste0("g", seq_len(groups))
  # The paper fits std.lv weak invariance, where lavaan frees later-group factor
  # variances after equating loadings. Marker identification describes the same
  # covariance-model pair with (G - 1)(p - 1) affine loading restrictions, and
  # keeps H0 literally nested in H1 as score_flip_test() requires.
  args <- list(syntax = fmg_analysis_syntax(p), std_lv = FALSE,
               meanstructure = FALSE, group = "group", group_labels = labels)
  list(
    H1 = do.call(magmaan::model_spec, args),
    H0 = do.call(magmaan::model_spec, c(args, list(group_equal = "loadings"))))
}

fmg_design_grid <- function() {
  out <- expand.grid(
    p = c(5L, 20L), groups = c(2L, 8L), n_group = 400L,
    distribution = c("normal", "vm", "ig", "pl"),
    truth = c("null", "power"), stringsAsFactors = FALSE)
  out$df <- with(out, as.integer((groups - 1L) * (p - 1L)))
  out$n_total <- out$groups * out$n_group
  # Interleave easy and hard cells under dynamic scheduling while keeping a
  # stable cell identifier independent of execution order.
  out$cell_id <- seq_len(nrow(out))
  out
}

fmg_validate_design <- function() {
  stopifnot(length(.fmg_null_loadings) == 20L)
  expected_dims <- list(`5_2` = c(2L, 5L), `5_8` = c(8L, 5L),
                        `20_2` = c(2L, 20L), `20_8` = c(8L, 20L))
  for (key in names(expected_dims)) {
    stopifnot(identical(dim(.fmg_power_loadings[[key]]), expected_dims[[key]]))
  }
  stopifnot(identical(.fmg_null_loadings[1:5], c(.8, .9, 1.1, 1.4, .7)))
  stopifnot(identical(.fmg_null_loadings[16:20], c(1, 1.2, 1.5, .9, 1.3)))
  stopifnot(identical(.fmg_power_loadings$`5_2`[2, ],
                      c(.72, .52, .68, .72, .88)))
  stopifnot(identical(.fmg_power_loadings$`20_8`[8, c(1, 10, 20)],
                      c(1.06, .64, 1.26)))
  grid <- fmg_design_grid()
  stopifnot(nrow(grid) == 32L, setequal(unique(grid$df), c(4L, 19L, 28L, 133L)))
  for (p in c(5L, 20L)) for (groups in c(2L, 8L)) {
    for (truth in c("null", "power")) {
      pop <- fmg_population(p, groups, truth)
      stopifnot(nrow(pop$loadings) == groups, ncol(pop$loadings) == p,
                length(pop$Sigma) == groups)
      stopifnot(all(vapply(pop$Sigma, function(S)
        min(eigen(S, symmetric = TRUE, only.values = TRUE)$values) > .999,
        logical(1))))
    }
  }
  invisible(TRUE)
}

.fmg_draws_one_group <- function(Sigma, n, reps, distribution, seed_base) {
  p <- ncol(Sigma)
  if (distribution == "normal") {
    L <- chol(Sigma)
    draws <- lapply(seq_len(reps), function(r) {
      set.seed(seed_base + r)
      matrix(stats::rnorm(n * p), n, p) %*% L
    })
    return(list(draws = draws, calibration = NULL))
  }

  skew <- rep(3, p)
  exkurt <- rep(21, p)
  if (distribution == "vm") {
    cal <- magmaan:::sim_vm_calibrate_impl(stats::cov2cor(Sigma), skew, exkurt)
    batch <- magmaan:::sim_vm_draw_impl(cal, n = n, reps = reps,
                                        seed_base = seed_base)
    sds <- sqrt(diag(Sigma))
    batch$draws <- lapply(batch$draws, function(X) sweep(X, 2L, sds, "*"))
  } else if (distribution == "ig") {
    cal <- magmaan:::sim_ig_calibrate_impl(
      Sigma, skew, exkurt, root = "symmetric", generator_family = "pearson",
      quadrature_points = 81L)
    batch <- magmaan:::sim_ig_draw_impl(
      cal, n = n, reps = reps, seed_base = seed_base,
      quadrature_points = 81L)
  } else if (distribution == "pl") {
    cal <- magmaan:::sim_plsim_calibrate_impl(
      stats::cov2cor(Sigma), skew, exkurt,
      method = "hermite_then_rectangle", num_segments = 12L,
      quadrature_points = 31L, hermite_order = 24L)
    batch <- magmaan:::sim_plsim_draw_impl(
      cal, n = n, reps = reps, seed_base = seed_base)
    sds <- sqrt(diag(Sigma))
    batch$draws <- lapply(batch$draws, function(X) sweep(X, 2L, sds, "*"))
  } else {
    stop("unknown distribution: ", distribution, call. = FALSE)
  }
  list(draws = batch$draws, calibration = cal)
}

fmg_make_sampler <- function(pop, n, reps, distribution, seed_base) {
  begin <- proc.time()[["elapsed"]]
  group_batches <- lapply(seq_len(pop$groups), function(g) {
    .fmg_draws_one_group(
      pop$Sigma[[g]], n, reps, distribution,
      seed_base + g * 1000003L)
  })
  setup_seconds <- proc.time()[["elapsed"]] - begin
  varnames <- paste0("x", seq_len(pop$p))
  labels <- paste0("g", seq_len(pop$groups))
  list(
    setup_seconds = setup_seconds,
    draw = function(rep_id) {
      blocks <- lapply(group_batches, function(x) x$draws[[rep_id]])
      blocks <- lapply(blocks, function(X) {
        colnames(X) <- varnames
        storage.mode(X) <- "double"
        X
      })
      dat <- data.frame(do.call(rbind, blocks),
                        group = factor(rep(labels, each = n), levels = labels),
                        check.names = FALSE)
      list(data = dat, blocks = blocks)
    })
}
