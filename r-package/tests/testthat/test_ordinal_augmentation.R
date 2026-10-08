# Ordinal augmentation adds threshold and scale rows from category counts.
# augment_model_spec() adds them once, from declared categories; fitting the
# result must give exactly the fit that per-fit augmentation gives.

augmentation_data <- function(levels, continuous = character(), n = 600L, seed = 4601L) {
  set.seed(seed)
  # Two factors correlated 0.5, matching augmentation_syntax.
  f1 <- rnorm(n)
  f2 <- 0.5 * f1 + sqrt(0.75) * rnorm(n)
  group <- rep(c("A", "B"), each = n / 2L)
  columns <- lapply(seq_along(levels), function(j) {
    f <- if (j <= 3L) f1 else f2
    z <- 0.7 * f + 0.3 * (group == "B") + sqrt(0.51) * rnorm(n)
    if (paste0("y", j) %in% continuous) return(z)
    cuts <- stats::qnorm(seq_len(levels[[j]] - 1L) / levels[[j]])
    ordered(findInterval(z, cuts) + 1L, levels = seq_len(levels[[j]]))
  })
  d <- as.data.frame(columns)
  names(d) <- paste0("y", seq_along(levels))
  d$g <- group
  d
}
augmentation_syntax <- "f1 =~ y1 + y2 + y3\nf2 =~ y4 + y5 + y6"

augmentation_spec <- function(d, continuous = character(), ...) {
  ordered <- setdiff(paste0("y", 1:6), continuous)
  model_spec(augmentation_syntax, ordered = ordered, meanstructure = TRUE, ...)
}
category_levels <- function(d, spec) lapply(d[spec$ordered], levels)
row_key <- function(pt) paste(pt$lhs, pt$op, pt$rhs, pt$group)

test_that("an augmented specification fits exactly as per-fit augmentation does", {
  d <- augmentation_data(c(2, 3, 5, 4, 2, 4))
  cases <- list(
    list(spec = augmentation_spec(d), estimators = c("DWLS", "WLS", "ULS", "ML")),
    list(spec = augmentation_spec(d, parameterization = "theta"), estimators = c("DWLS", "ULS")),
    list(spec = augmentation_spec(d, group = "g", group_labels = c("A", "B"),
                                  group_equal = c("loadings", "thresholds")),
         estimators = c("DWLS", "WLS")))
  for (case in cases) {
    augmented <- augment_model_spec(case$spec, category_levels(d, case$spec))
    for (estimator in case$estimators) for (covariance in c("unrestricted", "psd", "barrier")) {
      label <- paste(estimator, covariance)
      per_fit <- fit_model(case$spec, d, estimator = estimator, covariance = covariance)
      once <- fit_model(augmented, d, estimator = estimator, covariance = covariance)
      expect_identical(once$theta, per_fit$theta, label = label)
      expect_identical(once$fmin, per_fit$fmin, label = label)
      expect_identical(row_key(once$partable), row_key(per_fit$partable), label = label)
      expect_identical(once$partable$free, per_fit$partable$free, label = label)
    }
  }
})

test_that("mixed continuous and ordered models augment the ordered subset", {
  continuous <- c("y3", "y6")
  d <- augmentation_data(c(4, 3, 4, 4, 5, 4), continuous = continuous)
  spec <- augmentation_spec(d, continuous = continuous)
  augmented <- augment_model_spec(spec, category_levels(d, spec))
  expect_setequal(unique(augmented$partable$lhs[augmented$partable$op == "|"]), spec$ordered)
  for (estimator in c("DWLS", "WLS")) for (covariance in c("unrestricted", "psd")) {
    expect_identical(fit_model(augmented, d, estimator = estimator, covariance = covariance)$theta,
                     fit_model(spec, d, estimator = estimator, covariance = covariance)$theta,
                     label = paste(estimator, covariance))
  }
})

test_that("augmentation uses category counts only and has lavaan's rows", {
  d <- augmentation_data(c(2, 3, 5, 4, 2, 7))
  spec <- augmentation_spec(d)
  augmented <- augment_model_spec(spec, category_levels(d, spec))
  pt <- augmented$partable
  thresholds <- pt[pt$op == "|", ]
  expect_equal(as.vector(table(factor(thresholds$lhs, levels = spec$ordered))),
               c(1L, 2L, 4L, 3L, 1L, 6L))
  # No data enter the table: threshold starts stay missing, as in lavaan's.
  expect_true(all(is.na(thresholds$ustart)))
  expect_identical(augmented$categories, lapply(category_levels(d, spec), as.character))
  expect_identical(augment_model_spec(augmented, category_levels(d, spec)), augmented)
  lav <- lavaan::parTable(lavaan::cfa(augmentation_syntax, d[spec$ordered], ordered = spec$ordered,
                                      meanstructure = TRUE, do.fit = FALSE))
  keep <- function(x) x[x$op != "==", , drop = FALSE]
  expect_setequal(row_key(keep(pt)), row_key(keep(lav)))
})

test_that("data with other category counts than declared fail clearly", {
  d4 <- augmentation_data(rep(4, 6))
  spec <- augmentation_spec(d4)
  augmented <- augment_model_spec(spec, category_levels(d4, spec))
  d5 <- augmentation_data(rep(5, 6))
  expect_error(fit_model(augmented, d5, estimator = "DWLS"),
               "declares 4 categories for y1, but the data have 5")
  d3 <- augmentation_data(c(4, 3, 4, 4, 4, 4))
  for (estimator in c("DWLS", "ML")) {
    expect_error(fit_model(augmented, d3, estimator = estimator),
                 "declares 4 categories for y2, but the data have 3")
  }
  expect_error(augment_model_spec(model_spec(augmentation_syntax), list()), "no ordered variables")
  expect_error(augment_model_spec(spec, category_levels(d4, spec)[-1]), "named by the ordered")
  expect_error(augment_model_spec(spec, lapply(category_levels(d4, spec), `[`, 1L)),
               "at least two categories")
})


test_that("augmentation preserves source scale rows and prepared schema-only starts", {
  d <- augmentation_data(rep(4, 6))
  spec <- augmentation_spec(d)
  pt <- spec$partable
  scale <- pt[1L, , drop = FALSE]
  scale$id <- nrow(pt) + 1L
  scale$lhs <- scale$rhs <- "y1"
  scale$op <- "~*~"
  scale$free <- 0L
  scale$ustart <- 1
  scale$label <- scale$plabel <- ""
  spec$partable <- rbind(pt, scale)
  augmented <- augment_model_spec(spec, category_levels(d, spec))
  scales <- augmented$partable$op == "~*~" & augmented$partable$lhs == "y1"
  expect_equal(sum(scales), 1L)
  # rbind uses character row names on the reference route too.
  expected_scale <- spec$partable[nrow(spec$partable), , drop = FALSE]
  attr(expected_scale, "row.names") <- as.character(attr(expected_scale, "row.names"))
  expect_identical(augmented$partable[scales, , drop = FALSE], expected_scale)
  # A zero-row prototype supplies exactly the same factor schema.
  prepared <- prepare_model(spec, prototype = d[0L, ])
  expect_identical(prepared$spec$partable, augmented$partable)
  expect_identical(prepared$categories, augmented$categories)
})
