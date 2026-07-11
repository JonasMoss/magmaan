# magmaan-backed data generation for experiment 61.

draw_matrix <- function(draw) {
  if (is.matrix(draw) || is.data.frame(draw)) return(as.matrix(draw))
  if (is.list(draw) && !is.null(draw$X)) return(as.matrix(draw$X))
  stop("unsupported magmaan simulation draw shape", call. = FALSE)
}

ordinal_base_probabilities <- function(categories, threshold_profile) {
  categories <- as.integer(categories)
  if (threshold_profile == "symmetric") {
    return(switch(as.character(categories),
      `3` = c(.20, .60, .20),
      `5` = c(.10, .20, .40, .20, .10),
      `7` = c(.05, .10, .20, .30, .20, .10, .05),
      stop("unsupported symmetric category count: ", categories, call. = FALSE)))
  }
  switch(as.character(categories),
    `3` = c(.08, .27, .65),
    `5` = c(.04, .10, .21, .30, .35),
    `7` = c(.02, .05, .09, .16, .23, .25, .20),
    stop("unsupported asymmetric category count: ", categories, call. = FALSE))
}

ordinal_group_marginals <- function(pop, cell) {
  base_prob <- ordinal_base_probabilities(cell$categories[[1L]],
                                          cell$thresholds[[1L]])
  m1 <- pop$moments[[1L]]
  raw_thresholds <- lapply(seq_len(pop$p), function(j) {
    m1$mean[[j]] + sqrt(m1$Sigma[j, j]) *
      stats::qnorm(cumsum(base_prob)[-length(base_prob)])
  })
  lapply(pop$moments, function(m) {
    lapply(seq_len(pop$p), function(j) {
      z <- (raw_thresholds[[j]] - m$mean[[j]]) / sqrt(m$Sigma[j, j])
      cdf <- c(0, stats::pnorm(z), 1)
      p <- diff(cdf)
      pmax(p / sum(p), .Machine$double.eps)
    })
  })
}

calibrate_cell_sampler <- function(pop, cell) {
  generator <- cell$generator[[1L]]
  if (generator == "normal") {
    calibration <- lapply(pop$moments, function(m) {
      magmaan::sim_ordcorr_calibrate(
        stats::cov2cor(m$Sigma), rep(list(NULL), pop$p), metric = "polychoric",
        matrix_repair = "none")
    })
    return(list(kind = "normal", calibration = calibration, pop = pop))
  }
  if (startsWith(generator, "ig")) {
    target <- if (generator == "ig1") c(2, 7) else c(3, 21)
    calibration <- lapply(pop$moments, function(m) {
      magmaan::magmaan_core$sim_ig_calibrate(
        m$Sigma, target_skewness = rep(target[[1L]], pop$p),
        target_excess_kurtosis = rep(target[[2L]], pop$p),
        root = "symmetric", generator_family = "pearson",
        quadrature_points = 81L)
    })
    return(list(kind = "ig", calibration = calibration, pop = pop))
  }
  if (startsWith(generator, "ordinal_")) {
    marginals <- ordinal_group_marginals(pop, cell)
    calibration <- lapply(seq_len(2L), function(g) {
      magmaan::sim_ordcorr_calibrate(
        stats::cov2cor(pop$moments[[g]]$Sigma), marginals[[g]],
        metric = "polychoric", matrix_repair = "none")
    })
    return(list(kind = "ordinal", calibration = calibration, pop = pop,
                marginals = marginals))
  }
  stop("unknown generator: ", generator, call. = FALSE)
}

sampler_cache_key <- function(pop, cell) {
  moments <- unlist(lapply(pop$moments, function(m) c(m$mean, m$Sigma)),
                    use.names = FALSE)
  paste(cell$generator[[1L]], cell$categories[[1L]], cell$thresholds[[1L]],
        paste(format(round(moments, 10), scientific = FALSE, trim = TRUE),
              collapse = ","), sep = "|")
}

draw_cell_replication <- function(sampler, n, seed) {
  blocks <- lapply(seq_len(2L), function(g) {
    block_seed <- as.numeric(seed + g * 1000003)
    if (sampler$kind == "ig") {
      draw <- magmaan::magmaan_core$sim_ig_draw(
        sampler$calibration[[g]], n = as.integer(n[[g]]), reps = 1L,
        seed_base = block_seed, quadrature_points = 81L)$draws[[1L]]
      X <- draw_matrix(draw)
      X <- sweep(X, 2L, sampler$pop$moments[[g]]$mean, "+")
    } else {
      draw <- magmaan::sim_ordcorr_draw(
        sampler$calibration[[g]], n = as.integer(n[[g]]), reps = 1L,
        seed_base = block_seed)$draws[[1L]]
      X <- draw_matrix(draw)
      if (sampler$kind == "normal") {
        X <- sweep(X, 2L, sqrt(diag(sampler$pop$moments[[g]]$Sigma)), "*")
        X <- sweep(X, 2L, sampler$pop$moments[[g]]$mean, "+")
      }
    }
    colnames(X) <- sampler$pop$ov
    X
  })
  blocks
}
