source(file.path("resources", "sims", "r", "common.R"))

li2016_thresholds <- function() {
  list(
    symmetric = list(
      `4` = sim_thresholds_from_probs(c(.10, .40, .40, .10)),
      `5` = sim_thresholds_from_probs(c(.10, .20, .40, .20, .10)),
      `6` = sim_thresholds_from_probs(c(.05, .16, .29, .29, .16, .05)),
      `7` = sim_thresholds_from_probs(c(.05, .12, .18, .30, .18, .12, .05))
    ),
    slight = list(
      `4` = sim_thresholds_from_probs(c(.05, .09, .52, .34)),
      `5` = sim_thresholds_from_probs(c(.04, .05, .21, .46, .24)),
      `6` = sim_thresholds_from_probs(c(.04, .05, .05, .36, .31, .19)),
      `7` = sim_thresholds_from_probs(c(.04, .05, .06, .12, .42, .22, .09))
    ),
    moderate = list(
      `4` = sim_thresholds_from_probs(c(.05, .09, .26, .60)),
      `5` = sim_thresholds_from_probs(c(.04, .06, .10, .32, .48)),
      `6` = sim_thresholds_from_probs(c(.04, .05, .06, .10, .33, .42)),
      `7` = sim_thresholds_from_probs(c(.04, .04, .05, .06, .10, .32, .39))
    )
  )
}

li2016_design_grid <- function() {
  expand.grid(
    distribution = c("symmetric", "slight", "moderate"),
    categories = 4:7,
    n = c(200L, 300L, 400L, 500L, 750L, 1000L, 1500L),
    KEEP.OUT.ATTRS = FALSE
  )
}

li2016_sem_latent_cov <- function() {
  sim_li_structural_latent_cov()
}

li2016_model <- function() {
  paste(
    "xi1 =~ y1 + y2 + y3 + y4",
    "xi2 =~ y5 + y6 + y7 + y8",
    "eta1 =~ y9 + y10 + y11 + y12",
    "eta2 =~ y13 + y14 + y15 + y16",
    "eta3 =~ y17 + y18 + y19 + y20",
    "eta2 ~ eta1",
    "eta3 ~ eta1 + eta2",
    "eta1 ~ xi1 + xi2",
    "eta2 ~ xi1 + xi2",
    "eta3 ~ xi1 + xi2",
    "xi1 ~~ xi2",
    sep = "\n"
  )
}

li2016_generate <- function(n, categories, distribution = "symmetric",
                            seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  th <- li2016_thresholds()[[distribution]][[as.character(categories)]]
  if (is.null(th)) stop("Unsupported Li 2016 condition", call. = FALSE)

  loadings <- rep(list(c(.8, .7, .6, .5)), 5L)
  names(loadings) <- c("xi1", "xi2", "eta1", "eta2", "eta3")
  ystar <- sim_generate_indicators(
    n, loadings, li2016_sem_latent_cov(), paste0("y", 1:20)
  )
  data <- sim_ordinalize_df(ystar, rep(list(th), ncol(ystar)))
  list(
    paper = "Li 2016",
    condition = list(n = n, categories = categories,
                     distribution = distribution),
    data = data,
    ordered = names(data),
    model = li2016_model(),
    thresholds = th,
    latent_cov = li2016_sem_latent_cov()
  )
}
