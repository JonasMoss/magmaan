normal_moment <- function(k) {
  if (k %% 2) return(0)
  if (k == 0) return(1)
  prod(seq(1, k - 1, by = 2))
}
poly_product <- function(a, b) {
  out <- numeric(length(a) + length(b) - 1L)
  for (i in seq_along(a)) for (j in seq_along(b)) out[i+j-1L] <- out[i+j-1L] + a[i]*b[j]
  out
}
poly_moment <- function(coef, power) {
  p <- 1
  for (i in seq_len(power)) p <- poly_product(p, coef)
  sum(p * vapply(seq_along(p) - 1L, normal_moment, 0.))
}
fleishman <- function(skewness, kurtosis) {
  objective <- function(x) {
    coef <- c(-x[2], x)
    errors <- c(poly_moment(coef, 2)-1, poly_moment(coef, 3)-skewness,
                poly_moment(coef, 4)-3-kurtosis)
    sum(errors^2)
  }
  fit <- optim(c(1, skewness/6, kurtosis/24), objective, method = "BFGS",
               control = list(reltol = 1e-14, maxit = 2000, ndeps = rep(1e-6, 3)))
  if (fit$value > 1e-10) stop("Infeasible Fleishman moments")
  c(-fit$par[2], fit$par)
}

calibrate_generator <- function(population, cell) {
  coefficients <- if (cell$generator %in% c("vm", "independent"))
    fleishman(cell$skewness, cell$kurtosis) else NULL
  groups <- lapply(seq_along(population$stats$S), function(g) {
    sigma <- population$stats$S[[g]]; corr <- cov2cor(sigma)
    mu <- population$stats$mean[[g]]
    if (cell$generator == "vm") {
      b <- coefficients[2]; c <- coefficients[3]; d <- coefficients[4]
      latent <- corr
      for (i in seq_len(nrow(corr))) for (j in seq_len(i-1L)) {
        # Covariance of two centered cubic polynomials in correlated normals.
        latent[i,j] <- latent[j,i] <- uniroot(function(r)
          r*(b*b + 6*b*d + 9*d*d) + 2*c*c*r*r + 6*d*d*r^3 - corr[i,j],
          c(-1,1), tol = 1e-12)$root
      }
      factor <- chol(latent) # No silent repair: infeasible intermediate correlations fail.
    } else if (cell$generator == "discretized") {
      probabilities <- rep(1/cell$categories, cell$categories)
      calibration <- magmaanlab::sim_ordcorr_calibrate(corr,
        rep(list(probabilities), nrow(corr)), metric = "pearson", matrix_repair = "none")
      score_mean <- sum(seq_along(probabilities)*probabilities)
      score_variance <- sum((seq_along(probabilities)-score_mean)^2*probabilities)
      sigma <- corr * score_variance; mu <- rep(score_mean, nrow(corr))
      dimnames(sigma) <- dimnames(population$stats$S[[g]])
      return(list(calibration = calibration, sigma = sigma, mu = mu,
                  thresholds = qnorm(cumsum(probabilities)[-length(probabilities)])))
    } else factor <- chol(if (cell$generator == "independent") sigma else corr)
    list(factor = factor, sigma = sigma, mu = mu)
  })
  core_normal <- if (cell$generator == "normal" && length(groups) == 1)
    magmaanlab::sim_model_calibrate(population$generating) else NULL
  list(name = cell$generator, core_normal = core_normal, coefficients = coefficients, groups = groups,
       proportions = population$stats$nobs/sum(population$stats$nobs),
       labels = vapply(population$record$groups, `[[`, "", "label"))
}

group_sizes <- function(n, proportions) {
  sizes <- floor(n*proportions)
  sizes[seq_len(n-sum(sizes))] <- sizes[seq_len(n-sum(sizes))] + 1L
  sizes
}
draw_generator <- function(calibration, n, seed) {
  set.seed(seed)
  sizes <- group_sizes(n, calibration$proportions)
  groups <- lapply(seq_along(sizes), function(g) {
    spec <- calibration$groups[[g]]; count <- sizes[g]; p <- nrow(spec$sigma)
    if (!is.null(calibration$core_normal)) {
      x <- magmaanlab::sim_model_draw(calibration$core_normal, count, 1L, seed)$draws[[1]]$groups[[1]]$X
    } else if (calibration$name == "discretized") {
      x <- magmaanlab::sim_ordcorr_draw(spec$calibration, count, 1L,
                                      seed + g)$draws[[1]]
      if (is.list(x) && !is.matrix(x) && !is.data.frame(x)) x <- x$X
      x <- as.matrix(x)
    } else {
      z <- matrix(rnorm(count*p), count, p)
      if (calibration$name == "independent") {
        a <- calibration$coefficients
        z <- a[1] + a[2]*z + a[3]*z^2 + a[4]*z^3
        x <- z %*% spec$factor
      } else {
        z <- z %*% spec$factor
        if (calibration$name == "vm") {
          a <- calibration$coefficients
          z <- a[1] + a[2]*z + a[3]*z^2 + a[4]*z^3
        }
        x <- sweep(z, 2, sqrt(diag(spec$sigma)), "*")
      }
      x <- sweep(x, 2, spec$mu, "+")
    }
    colnames(x) <- colnames(spec$sigma)
    data <- as.data.frame(x)
    if (length(sizes) > 1L) data$study_group <- calibration$labels[g]
    data
  })
  do.call(rbind, groups)
}

validate_generator <- function(calibration, seed, n = 1000000L) {
  data <- draw_generator(calibration, n, seed)
  checks <- lapply(seq_along(calibration$groups), function(g) {
    x <- if (length(calibration$groups) == 1) data else
      data[data$study_group == calibration$labels[g], setdiff(names(data), "study_group")]
    sigma <- calibration$groups[[g]]$sigma
    error <- sqrt(sum((cov(x)-sigma)^2)/sum(sigma^2))
    data.frame(group = calibration$labels[g], n = nrow(x), relative_error = error,
               passed = error < 1e-2)
  })
  out <- do.call(rbind, checks)
  if (!all(out$passed)) stop("Generator covariance check failed: ", max(out$relative_error))
  out
}

apply_mar <- function(data, cell, seed) {
  variables <- setdiff(names(data), "study_group")
  cause <- if (nzchar(cell$cause)) cell$cause else variables[1]
  targets <- if (nzchar(cell$missing_targets)) strsplit(cell$missing_targets, ";", fixed = TRUE)[[1]] else
    tail(setdiff(variables, cause), floor(length(variables)/2))
  if ("study_group" %in% names(data)) {
    pieces <- lapply(unique(data$study_group), function(g)
      single_cause_mar(data[data$study_group == g, ], cause, targets, cell$missing_rate,
                       seed + match(g, unique(data$study_group))))
    do.call(rbind, pieces)
  } else single_cause_mar(data, cause, targets, cell$missing_rate, seed)
}
