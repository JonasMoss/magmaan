source(file.path("docs", "research", "sims", "r", "common.R"))

li2021_thresholds <- function() {
  list(
    symmetric = list(
      `2` = 0,
      `3` = c(-.84, .84),
      `4` = c(-1.282, 0, 1.282),
      `5` = c(-1.282, -.524, .524, 1.282),
      `6` = c(-1.645, -.806, 0, .806, 1.645),
      `7` = c(-1.645, -.954, -.385, .385, .954, 1.645)
    ),
    slight = list(
      `2` = -.553,
      `3` = c(-1.282, -.202),
      `4` = c(-1.645, -1.08, .412),
      `5` = c(-1.751, -1.341, -.524, .706),
      `6` = c(-1.751, -1.341, -1.08, 0, .878),
      `7` = c(-1.751, -1.341, -1.036, -.613, .496, 1.341)
    )
  )
}

li2021_design_grid <- function() {
  expand.grid(
    model = c("cfa", "sem"),
    distribution = c("symmetric", "slight"),
    categories = 2:7,
    n = c(200L, 500L, 1000L),
    KEEP.OUT.ATTRS = FALSE
  )
}

li2021_cfa_model <- function() {
  paste(
    "f1 =~ x1 + x2 + x3 + y1",
    "f2 =~ x4 + x5 + y2 + y3",
    "f3 =~ x6 + y4 + y5 + y6",
    "f1 ~~ f2 + f3",
    "f2 ~~ f3",
    sep = "\n"
  )
}

li2021_sem_model <- function() {
  paste(
    "xi1 =~ x1 + x2 + y1 + y2",
    "xi2 =~ x3 + x4 + y3 + y4",
    "eta1 =~ x5 + x6 + y5 + y6",
    "eta2 =~ x7 + x8 + y7 + y8",
    "eta3 =~ x9 + x10 + y9 + y10",
    "eta2 ~ eta1",
    "eta3 ~ eta1 + eta2",
    "eta1 ~ xi1 + xi2",
    "eta2 ~ xi1 + xi2",
    "eta3 ~ xi1 + xi2",
    "xi1 ~~ xi2",
    sep = "\n"
  )
}

li2021_generate_cfa <- function(n, categories, distribution = "symmetric",
                                seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  th <- li2021_thresholds()[[distribution]][[as.character(categories)]]
  if (is.null(th)) stop("Unsupported Li 2021 CFA condition", call. = FALSE)

  factor_cov <- matrix(.3, 3, 3)
  diag(factor_cov) <- 1
  loadings <- list(
    f1 = c(.8, .7, .6, .7),
    f2 = c(.8, .7, .8, .7),
    f3 = c(.7, .8, .7, .6)
  )
  names <- c("x1", "x2", "x3", "y1",
             "x4", "x5", "y2", "y3",
             "x6", "y4", "y5", "y6")
  ystar <- sim_generate_indicators(n, loadings, factor_cov, names)
  data <- as.data.frame(ystar[, grep("^x", names), drop = FALSE])
  for (nm in grep("^y", names, value = TRUE)) {
    data[[nm]] <- sim_ordinalize(ystar[, nm], th)
  }
  data <- data[names]
  list(
    paper = "Li 2021",
    model_kind = "cfa",
    condition = list(n = n, categories = categories,
                     distribution = distribution),
    data = data,
    ordered = grep("^y", names, value = TRUE),
    model = li2021_cfa_model(),
    thresholds = th
  )
}

li2021_generate_sem <- function(n, categories, distribution = "symmetric",
                                seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  th <- li2021_thresholds()[[distribution]][[as.character(categories)]]
  if (is.null(th)) stop("Unsupported Li 2021 SEM condition", call. = FALSE)

  loadings <- rep(list(c(.8, .6, .8, .6)), 5L)
  names(loadings) <- c("xi1", "xi2", "eta1", "eta2", "eta3")
  variable_names <- c("x1", "x2", "y1", "y2",
                      "x3", "x4", "y3", "y4",
                      "x5", "x6", "y5", "y6",
                      "x7", "x8", "y7", "y8",
                      "x9", "x10", "y9", "y10")
  ystar <- sim_generate_indicators(
    n, loadings, sim_li_structural_latent_cov(), variable_names
  )
  data <- as.data.frame(ystar[, grep("^x", variable_names), drop = FALSE])
  for (nm in grep("^y", variable_names, value = TRUE)) {
    data[[nm]] <- sim_ordinalize(ystar[, nm], th)
  }
  data <- data[variable_names]
  list(
    paper = "Li 2021",
    model_kind = "sem",
    condition = list(n = n, categories = categories,
                     distribution = distribution),
    data = data,
    ordered = grep("^y", variable_names, value = TRUE),
    model = li2021_sem_model(),
    thresholds = th,
    latent_cov = sim_li_structural_latent_cov()
  )
}

li2021_generate <- function(model = c("cfa", "sem"), n, categories,
                            distribution = "symmetric", seed = NULL) {
  model <- match.arg(model)
  if (model == "cfa") {
    li2021_generate_cfa(n, categories, distribution, seed)
  } else {
    li2021_generate_sem(n, categories, distribution, seed)
  }
}
