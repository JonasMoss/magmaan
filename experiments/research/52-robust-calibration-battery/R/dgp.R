# Data-generating processes. Every generator reproduces the population mean and
# covariance of each group exactly, so H1 and H0 hold in every condition.
#
#   normal : multivariate normal.
#   vm     : Vale-Maurelli/Fleishman, every margin skewness 2, excess kurtosis 7.
#   ig     : Foldnes-Olsson independent generator, same marginal targets, Pearson
#            marginal family, Cholesky root (the heterogeneous-spectrum case).
#   disc   : 5-point items. A Gaussian copula is thresholded at the marginal
#            category probabilities, with the latent correlations calibrated so
#            that the Pearson correlations of the integer codes equal the
#            population correlations. Each item is then mapped affinely from
#            its code scale to the population mean and standard deviation. The
#            data stay 5-valued per item; no polychoric step anywhere.
dgps <- c("normal", "vm", "ig", "disc")
skew_target <- 2
kurt_target <- 7

# Two 5-category shapes. By default they alternate over items so each model
# mixes them; a population may ask for the skewed shape throughout.
disc_shapes <- list(symmetric = c(.10, .20, .40, .20, .10),
                    skewed    = c(.35, .30, .18, .11, .06))
disc_marginals <- function(p, pattern = "alternate") {
  if (identical(pattern, "skewed")) return(rep(list(disc_shapes$skewed), p))
  lapply(seq_len(p), function(j) disc_shapes[[1L + (j %% 2L == 0L)]])
}
code_moments <- function(prob) {
  k <- seq_along(prob)  # the generator's integer codes are 1..K
  m <- sum(k * prob)
  c(mean = m, sd = sqrt(sum((k - m)^2 * prob)))
}

core <- function() magmaanlab::magmaan_core
`%||%` <- function(a, b) if (is.null(a)) b else a

calibrate_group <- function(group, dgp, pattern = "alternate") {
  S <- group$sigma
  sds <- sqrt(diag(S))
  R <- cov2cor(S)
  p <- nrow(S)
  cal <- switch(dgp,
    normal = list(root = chol(S)),
    vm = list(state = core()$sim_vm_calibrate(R, rep(skew_target, p), rep(kurt_target, p))),
    ig = list(state = core()$sim_ig_calibrate(S, rep(skew_target, p), rep(kurt_target, p),
                                              root = "cholesky", generator_family = "pearson")),
    disc = {
      marg <- disc_marginals(p, pattern)
      list(state = core()$sim_ordcorr_calibrate(R, marg, metric = "pearson_codes"),
           code = t(vapply(marg, code_moments, numeric(2))))
    })
  cal$sds <- sds
  cal$mu <- group$mu
  cal$ov <- group$ov
  cal
}

calibrate_population <- function(pop, dgp) {
  lapply(pop$groups, calibrate_group, dgp = dgp, pattern = pop$disc_pattern %||% "alternate")
}

draw_group <- function(cal, n, dgp, seed) {
  p <- length(cal$mu)
  X <- switch(dgp,
    normal = {set.seed(seed); matrix(rnorm(n * p), n, p) %*% cal$root},
    vm = sweep(core()$sim_vm_draw(cal$state, n = n, reps = 1L, seed_base = seed)$draws[[1L]],
               2, cal$sds, "*"),
    ig = core()$sim_ig_draw(cal$state, n = n, reps = 1L, seed_base = seed)$draws[[1L]],
    disc = {
      Y <- core()$sim_ordcorr_draw(cal$state, n = n, reps = 1L, seed_base = seed)$draws[[1L]]$X
      Z <- sweep(sweep(Y, 2, cal$code[, "mean"], "-"), 2, cal$code[, "sd"], "/")
      sweep(Z, 2, cal$sds, "*")
    })
  X <- sweep(X, 2, cal$mu, "+")
  colnames(X) <- cal$ov
  X
}

group_sizes <- function(pop, n) {
  props <- vapply(pop$groups, `[[`, numeric(1), "proportion")
  sizes <- floor(n * props)
  sizes[1] <- sizes[1] + n - sum(sizes)
  sizes
}

draw_sample <- function(pop, cals, n, dgp, seed) {
  sizes <- group_sizes(pop, n)
  blocks <- lapply(seq_along(cals), function(g)
    draw_group(cals[[g]], sizes[g], dgp, seed + 50000L * (g - 1L)))
  d <- as.data.frame(do.call(rbind, blocks))
  if (length(cals) > 1L)
    d$group <- rep(vapply(pop$groups, `[[`, character(1), "label"), sizes)
  d
}
