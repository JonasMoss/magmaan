source(file.path("resources", "sims", "r", "common.R"))

rhemtulla_design_grid <- function() {
  expand.grid(
    model_size = c(10L, 20L),
    underlying = c("normal", "nonnormal"),
    categories = 2:7,
    threshold_pattern = c("symmetric", "moderate", "moderate_alternating",
                          "extreme", "extreme_alternating"),
    n = c(100L, 150L, 350L, 600L),
    KEEP.OUT.ATTRS = FALSE
  )
}

rhemtulla_model <- function(model_size = c(10L, 20L)) {
  model_size <- match.arg(as.character(model_size), c("10", "20"))
  per_factor <- as.integer(model_size) / 2L
  f1 <- paste0("y", seq_len(per_factor), collapse = " + ")
  f2 <- paste0("y", per_factor + seq_len(per_factor), collapse = " + ")
  paste("f1 =~", f1, "\n", "f2 =~", f2, "\n", "f1 ~~ f2")
}

rhemtulla_thresholds <- function(categories, pattern, item_index = 1L) {
  if (pattern == "symmetric") {
    return(seq(-2.5, 2.5, length.out = categories + 1L)[-c(1L, categories + 1L)])
  }

  base_pattern <- sub("_alternating$", "", pattern)
  if (base_pattern == "moderate") {
    loc <- seq(-1.6, 1.4, length.out = categories)
    probs <- stats::dnorm(loc, mean = -1.0, sd = .9)
  } else if (base_pattern == "extreme") {
    probs <- exp(-seq(0, categories - 1L) * .85)
  } else {
    stop("Unsupported Rhemtulla threshold pattern", call. = FALSE)
  }
  probs <- probs / sum(probs)
  if (grepl("_alternating$", pattern) && item_index %% 2L == 1L) {
    probs <- rev(probs)
  }
  sim_thresholds_from_probs(probs)
}

rhemtulla_nonnormalize <- function(z) {
  out <- stats::qchisq(stats::pnorm(z), df = 2)
  scale(out)[, 1]
}

rhemtulla_generate <- function(n, model_size = 10L, categories,
                               underlying = "normal",
                               threshold_pattern = "symmetric",
                               seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  per_factor <- model_size / 2L
  loading_values <- rep(c(.3, .4, .5, .6, .7),
                        length.out = per_factor)
  loadings <- list(f1 = loading_values, f2 = loading_values)
  ystar <- sim_generate_indicators(
    n, loadings, matrix(c(1, .3, .3, 1), 2, 2), paste0("y", 1:model_size)
  )
  if (underlying == "nonnormal") {
    ystar <- apply(ystar, 2L, rhemtulla_nonnormalize)
    colnames(ystar) <- paste0("y", 1:model_size)
  }

  thresholds <- lapply(seq_len(model_size), function(j) {
    rhemtulla_thresholds(categories, threshold_pattern, item_index = j)
  })
  names(thresholds) <- paste0("y", 1:model_size)
  data <- sim_ordinalize_df(ystar, thresholds)
  list(
    paper = "Rhemtulla et al. 2012",
    condition = list(n = n, model_size = model_size,
                     categories = categories, underlying = underlying,
                     threshold_pattern = threshold_pattern),
    data = data,
    ordered = names(data),
    model = rhemtulla_model(model_size),
    thresholds = thresholds
  )
}
