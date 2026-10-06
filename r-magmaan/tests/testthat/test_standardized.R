# Compare at the same parameter point: optimizer endpoint differences are
# separately gated by the fitting tests and must not obscure the delta map.
standardized_parity <- function(fit, lav, bundle) {
  rows <- fit$lab$partable
  target <- lavaan::parTable(lav)
  idx <- match(.key(rows, fit$lab$group_labels), .key(target, .lav_labels(lav)))
  free <- rows$free > 0 & !is.na(idx)
  fit$lab$theta[rows$free[free]] <- target$est[idx[free]]
  # Standardization parity needs identical covariance inputs. Bundle covariance
  # parity has separate convention gates (and fitting-weight contracts).
  V <- vcov(fit, lavaan_compat = bundle)
  pt <- target[target$free > 0 & !duplicated(target$free), ]
  own <- rows[rows$free > 0 & !duplicated(rows$free), ]
  own <- own[order(own$free), ]
  vi <- match(.key(own, fit$lab$group_labels), .key(pt, .lav_labels(lav)))
  lav@vcov$vcov[pt$free[vi], pt$free[vi]] <- unname(V)
  s <- summary(fit, standardized = TRUE, lavaan_compat = bundle)
  p <- coef(s)
  for (type in c("std.lv", "std.all")) {
    ref <- lavaan::standardizedSolution(lav, type = type)
    hit <- match(.key(p, fit$lab$group_labels), .key(ref, .lav_labels(lav)))
    keep <- !is.na(hit) & p$op != ":="
    expect_equal(p[[type]][keep], ref$est.std[hit[keep]], tolerance = 1e-6)
    actual_se <- p[[paste0(type, ".se")]][keep]
    reference_se <- ref$se[hit[keep]]
    nonzero <- reference_se > 1e-8
    expect_lt(max(abs(actual_se[nonzero] / reference_se[nonzero] - 1)), 1e-5)
    expect_lt(max(abs(actual_se[!nonzero] - reference_se[!nonzero])), 1e-8)
  }
  ref_r2 <- lavaan::lavInspect(lav, "r2")
  if (!is.list(ref_r2)) ref_r2 <- list(ref_r2)
  for (g in seq_along(ref_r2)) {
    z <- s$r2
    if (!is.null(z$group)) {
      own_group <- match(.lav_labels(lav)[g], fit$lab$group_labels)
      z <- z[z$group == own_group, ]
    }
    expect_setequal(z$variable, names(ref_r2[[g]]))
    expect_equal(z$r2, as.numeric(ref_r2[[g]][z$variable]), tolerance = 1e-6)
  }
  invisible(fit)
}

standardized_delta <- function(fit) {
  s <- summary(fit, standardized = TRUE)
  V <- vcov(fit)
  point <- fit$lab
  n <- length(point$theta)
  zero <- matrix(0, n, n)
  base <- magmaanlab::standardized_rows(point, zero)
  for (type in c("std.lv", "std.all")) {
    J <- matrix(0, length(base[[type]]), n)
    for (k in seq_len(n)) {
      h <- 1e-5 * (1 + abs(point$theta[k]))
      plus <- minus <- point
      plus$theta[k] <- point$theta[k] + h
      minus$theta[k] <- point$theta[k] - h
      J[, k] <- (magmaanlab::standardized_rows(plus, zero)[[type]] -
                   magmaanlab::standardized_rows(minus, zero)[[type]]) / (2 * h)
    }
    expected <- sqrt(pmax(0, rowSums((J %*% V) * J)))
    keep <- !point$partable$op %in% magmaan:::.constraint_ops
    actual <- coef(s)[[paste0(type, ".se")]]
    expected <- expected[keep]
    finite <- is.finite(expected) & expected > 1e-8
    expect_lt(max(abs(actual[finite] / expected[finite] - 1)), 1e-6)
  }
  p <- coef(s)
  residual <- p[p$op == "~~" & p$lhs == p$rhs, ]
  hit <- match(paste(s$r2$group %||% 1, s$r2$variable),
               paste(residual$group %||% 1, residual$lhs))
  expect_equal(s$r2$r2, 1 - residual$std.all[hit])
  expect_equal(s$r2$r2.se, residual$std.all.se[hit])
}

`%||%` <- function(x, y) if (is.null(x)) y else x

test_that("ordinary ML standardized rows match lavaan and the active delta covariance", {
  d <- hs()
  for (grouped in c(FALSE, TRUE)) {
    group <- if (grouped) "school" else NULL
    fit <- magmaan(magmaan_model(cfa, prototype = d, group = group), d)
    lav <- lav_cfa(cfa, d, group = group, fixed.x = FALSE)
    standardized_parity(fit, lav, "ML")
    standardized_delta(fit)
  }
  d <- lavaan::PoliticalDemocracy
  model <- 'ind60 =~ x1 + x2 + x3
    dem60 =~ y1 + y2 + y3 + y4
    dem65 =~ y5 + y6 + y7 + y8
    dem60 ~ ind60
    dem65 ~ ind60 + dem60
    y1 ~~ y5
    y2 ~~ y4 + y6
    y3 ~~ y7
    y4 ~~ y8
    y6 ~~ y8'
  fit <- magmaan(model, d)
  lav <- lavaan::sem(model, d, meanstructure = TRUE, fixed.x = FALSE)
  standardized_parity(fit, lav, "ML")
  standardized_delta(fit)
})

test_that("ordinal and mixed standardized rows include thresholds and fixed residuals", {
  for (mixed in c(FALSE, TRUE)) for (par in c("delta", "theta")) for (grouped in c(FALSE, TRUE)) {
    group <- if (grouped) "school" else NULL
    d <- ordinal_hs()
    ordered <- paste0("x", 1:6)
    if (mixed) {
      d$x6 <- hs()$x6
      ordered <- ordered[1:5]
    }
    fit <- magmaan(magmaan_model(cfa, prototype = d, ordered = ordered,
                                  parameterization = par, group = group), d, estimator = "DWLS",
                    options = list(preset = "lavaan-0.7.2"))
    lav <- lav_cfa(cfa, d, ordered = ordered, parameterization = par,
                   estimator = "WLSMV", fixed.x = FALSE, group = group)
    standardized_parity(fit, lav, "WLSMV")
    standardized_delta(fit)
  }
})

test_that("FIML, grouped equality and unavailable inference retain standardization", {
  d <- hs()
  fit <- magmaan(magmaan_model(cfa, prototype = d, group = "school",
                                group.equal = "loadings"), d)
  lav <- lav_cfa(cfa, d, group = "school", group.equal = "loadings", fixed.x = FALSE)
  standardized_parity(fit, lav, "ML")
  standardized_delta(fit)
  d$x2[seq(1, nrow(d), 7)] <- NA
  fit <- magmaan(cfa, d, estimator = "FIML")
  standardized_delta(fit)
  grouped <- magmaan(magmaan_model(cfa, prototype = d, group = "school"),
                      d, estimator = "FIML")
  standardized_delta(grouped)
  plain <- magmaan(cfa, hs(), inference = FALSE)
  s <- summary(plain, standardized = TRUE)
  expect_true(all(is.finite(coef(s)$std.all)))
  expect_true(all(is.na(coef(s)$std.all.se)))
  expect_true(all(is.na(s$r2$r2.se)))
  expect_identical(attr(coef(s), "inference_reason"), "not_computed")
  expect_identical(attr(s$r2, "inference_reason"), "not_computed")
  expect_null(summary(plain)$r2)
  expect_output(print(s), "R-squared")
  expect_error(summary(plain, standardized = NA), "TRUE or FALSE")
})

test_that("defined standardized scales are explicitly unavailable", {
  fit <- magmaan(paste(cfa, 'visual ~~ rho*textual\nassociation := rho', sep = '\n'), hs())
  s <- summary(fit, standardized = TRUE)
  p <- coef(s)
  expect_true(any(p$op == ':='))
  expect_true(all(is.na(p$std.lv[p$op == ':='])))
  expect_true(all(is.na(p$std.all.se[p$op == ':='])))
  expect_identical(attr(p, 'defined_reason'), 'unsupported_defined_scale')
})
