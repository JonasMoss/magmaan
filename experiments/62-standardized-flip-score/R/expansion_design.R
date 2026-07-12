flip_expansion_model <- paste(
  "f1 =~ x1 + x2 + x3 + x4 + x5",
  "f2 =~ x6 + x7 + x8 + x9 + x10",
  "f1 ~~ f2", sep = "\n")

flip_expansion_ov <- paste0("x", 1:10)
flip_expansion_loading_tokens <- c(
  paste("f1 =~", paste0("x", 2:5)),
  paste("f2 =~", paste0("x", 7:10)))

flip_expansion_specs <- function(df) {
  stopifnot(df %in% c(1L, 4L, 8L))
  tested <- flip_expansion_loading_tokens[seq_len(df)]
  partial <- setdiff(flip_expansion_loading_tokens, tested)
  list(
    configural = model_spec(
      flip_expansion_model, meanstructure = TRUE, group = "school",
      group_labels = c("A", "B")),
    restricted = model_spec(
      flip_expansion_model, meanstructure = TRUE, group = "school",
      group_labels = c("A", "B"), group_equal = "loadings",
      group_partial = partial))
}

flip_expansion_population <- function(heterogeneity) {
  loading <- matrix(0, 10, 2)
  loading[1:5, 1] <- c(1, .9, .75, 1.1, .65)
  loading[6:10, 2] <- c(1, .8, 1.15, .7, .95)
  theta1 <- diag(c(.50, .65, .80, .45, .90,
                   .55, .75, .40, .85, .60))
  phi1 <- matrix(c(1, .30, .30, 1), 2, 2)

  if (heterogeneity == "homogeneous") {
    phi2 <- phi1
    theta2 <- theta1
  } else if (heterogeneity == "scale") {
    phi2 <- 3 * phi1
    theta2 <- theta1
  } else if (heterogeneity == "geometry") {
    latent_var <- c(3, .45)
    phi2 <- matrix(c(latent_var[1], .30 * sqrt(prod(latent_var)),
                     .30 * sqrt(prod(latent_var)), latent_var[2]), 2, 2)
    residual_multiplier <- rep(c(.40, 2.20), length.out = 10)
    theta2 <- diag(diag(theta1) * residual_multiplier)
  } else stop("unknown heterogeneity: ", heterogeneity, call. = FALSE)

  intercept <- seq(.15, 1.50, length.out = 10)
  alpha2 <- c(.45, -.30)
  list(
    mu = list(intercept, intercept + as.vector(loading %*% alpha2)),
    Sigma = list(loading %*% phi1 %*% t(loading) + theta1,
                 loading %*% phi2 %*% t(loading) + theta2))
}

flip_expansion_group_sizes <- function(n_total, balance) {
  if (balance == "1:1") return(c(n_total %/% 2L, n_total - n_total %/% 2L))
  if (balance == "1:3") return(c(n_total %/% 4L, n_total - n_total %/% 4L))
  stop("unknown balance: ", balance, call. = FALSE)
}

flip_expansion_draw_block <- function(n, mu, Sigma, distribution) {
  p <- length(mu)
  if (distribution == "normal") {
    z <- matrix(rnorm(n * p), n, p)
  } else if (distribution == "t5") {
    z <- matrix(rnorm(n * p), n, p)
    z <- z * sqrt(3 / 5) / sqrt(rchisq(n, 5) / 5)
  } else if (distribution == "skew") {
    z <- matrix(rexp(n * p) - 1, n, p)
  } else stop("unknown distribution: ", distribution, call. = FALSE)
  sweep(z %*% chol(Sigma), 2, mu, "+")
}

flip_expansion_draw_data <- function(n_total, balance, heterogeneity,
                                     distribution, seed) {
  set.seed(seed)
  pop <- flip_expansion_population(heterogeneity)
  sizes <- flip_expansion_group_sizes(n_total, balance)
  blocks <- lapply(1:2, function(g) flip_expansion_draw_block(
    sizes[g], pop$mu[[g]], pop$Sigma[[g]], distribution))
  blocks <- lapply(blocks, function(x) {
    colnames(x) <- flip_expansion_ov
    x
  })
  data.frame(
    rbind(blocks[[1]], blocks[[2]]),
    school = rep(c("A", "B"), sizes), check.names = FALSE)
}
