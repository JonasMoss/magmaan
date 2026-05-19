source(file.path("resources", "sims", "r", "common.R"))

lei2020_thresholds <- function() {
  list(
    symmetrical = list(
      `2` = 0,
      `3` = c(-.9730, .9730),
      `4` = c(-1.7507, 0, 1.7507),
      `5` = c(-1.3408, -.6745, .6745, 1.3408)
    ),
    moderate = list(
      `2` = -.9652,
      `3` = c(-1.0460, -.8670),
      `4` = c(-1.1735, -.9287, -.7768),
      `5` = c(-1.6330, -1.0660, -.7130, 1.3030)
    ),
    extreme = list(
      `2` = -1.3784,
      `3` = c(-1.7412, -1.1216),
      `4` = c(-1.7472, -1.3460, 1.7429),
      `5` = c(-1.8806, -1.5113, -1.1388, 1.7465)
    )
  )
}

lei2020_mar_slopes <- function() {
  list(
    `5` = rbind(
      symmetrical = c(`2` = 2.4038, `3` = 1.4550,
                      `4` = 1.0422, `5` = .7513),
      moderate = c(`2` = 1.9176, `3` = .9659,
                   `4` = .6480, `5` = .6130),
      extreme = c(`2` = 1.9869, `3` = 1.0057,
                  `4` = .6676, `5` = .5071)
    ),
    `30` = rbind(
      symmetrical = c(`2` = 3.1830, `3` = 1.9027,
                      `4` = 1.3011, `5` = .9615),
      moderate = c(`2` = 2.2947, `3` = 1.1568,
                   `4` = .7764, `5` = .7379),
      extreme = c(`2` = 2.1675, `3` = 1.9063,
                  `4` = 1.0463, `5` = .7141)
    )
  )
}

lei2020_design_grid <- function() {
  expand.grid(
    missing = c("none", "mcar_5", "mcar_30", "mar_5", "mar_30"),
    n = c(100L, 200L, 500L, 1000L),
    categories = 2:5,
    distribution = c("symmetrical", "moderate", "extreme"),
    KEEP.OUT.ATTRS = FALSE
  )
}

lei2020_model <- function() {
  paste(
    "f1 =~ y1 + y2 + y3 + y4 + y5",
    "f2 =~ y6 + y7 + y8 + y9 + y10",
    "f1 ~~ f2",
    sep = "\n"
  )
}

lei2020_apply_missing <- function(data, mechanism, categories, distribution,
                                  seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  if (mechanism == "none") return(data)

  parts <- strsplit(mechanism, "_", fixed = TRUE)[[1]]
  kind <- parts[[1]]
  amount <- as.integer(parts[[2]])
  beta0 <- if (amount == 5L) -2.83427 else -.6933
  if (kind == "mar") beta0 <- if (amount == 5L) -4.5951 else -2.7515

  out <- data
  for (nm in names(out)[-1L]) {
    p <- if (kind == "mcar") {
      rep(stats::plogis(beta0), nrow(out))
    } else {
      beta1 <- lei2020_mar_slopes()[[as.character(amount)]][
        distribution, as.character(categories)
      ]
      c0 <- as.integer(out$y1) - 1L
      stats::plogis(beta0 + beta1 * c0)
    }
    out[[nm]][stats::runif(nrow(out)) < p] <- NA
  }
  out
}

lei2020_generate <- function(n, categories, distribution = "symmetrical",
                             missing = "none", seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  th <- lei2020_thresholds()[[distribution]][[as.character(categories)]]
  if (is.null(th)) stop("Unsupported Lei/Shiverdecker condition", call. = FALSE)

  factor_cov <- matrix(c(1, .3, .3, 1), 2, 2)
  loadings <- list(f1 = rep(.7, 5L), f2 = rep(.7, 5L))
  ystar <- sim_generate_indicators(n, loadings, factor_cov, paste0("y", 1:10))
  data <- sim_ordinalize_df(ystar, rep(list(th), ncol(ystar)))
  data <- lei2020_apply_missing(
    data, missing, categories, distribution,
    seed = if (is.null(seed)) NULL else seed + 1L
  )

  list(
    paper = "Lei and Shiverdecker 2020",
    condition = list(n = n, categories = categories,
                     distribution = distribution, missing = missing),
    data = data,
    ordered = names(data),
    model = lei2020_model(),
    thresholds = th
  )
}
