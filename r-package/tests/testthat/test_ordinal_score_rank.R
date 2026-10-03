ordinal_score_rank_data <- function(grouped = FALSE) {
  set.seed(101)
  n <- 700L
  eta <- rnorm(n)
  d <- as.data.frame(sapply(c(1, 0.8, 0.7, 0.9), function(l) {
    as.integer(cut(l * eta + rnorm(n, sd = 0.7), c(-Inf, -0.5, 0.5, Inf)))
  }))
  names(d) <- paste0("x", 1:4)
  if (grouped) d$g <- rep(c("a", "b"), c(400L, 300L))
  d
}

ordinal_score_rank_key <- function(x) paste(x$kind, x$lhs, x$op, x$rhs, x$group)

test_that("ordinal ordinary and robust ranks agree across weight scales", {
  for (parameterization in c("delta", "theta")) {
    for (grouped in c(FALSE, TRUE)) {
      d <- ordinal_score_rank_data(grouped)
      for (estimator in c("DWLS", "WLS")) {
        fit <- fit_model("f =~ x1+x2+x3+x4", d, estimator = estimator,
                         ordered = paste0("x", 1:4), parameterization = parameterization,
                         groups = if (grouped) "g" else NULL)
        expect_true(fit$converged)
        baseline <- modification_indices(fit)
        baseline_robust <- modification_indices_robust(fit, estimated_weight = FALSE)
        expect_equal(nrow(baseline), if (grouped) 12L else 6L)
        expect_true(all(baseline$op == "~~" & baseline$lhs != baseline$rhs))
        for (scale in c(1e-16, 1, 1e12)) {
          scaled <- fit
          slot <- if (estimator == "DWLS") "W_dwls" else "W_wls"
          scaled$ordinal_stats[[slot]] <- lapply(fit$ordinal_stats[[slot]], function(w) scale * w)
          ordinary <- modification_indices(scaled)
          robust <- modification_indices_robust(scaled, estimated_weight = FALSE)
          expect_identical(ordinal_score_rank_key(ordinary), ordinal_score_rank_key(baseline))
          expect_identical(ordinal_score_rank_key(robust), ordinal_score_rank_key(ordinary))
          expect_equal(ordinary$mi / scale, baseline$mi, tolerance = 1e-7)
          expect_equal(ordinary$epc, baseline$epc, tolerance = 1e-7)
          expect_equal(ordinary$mi, robust$mi, tolerance = 1e-8)
          expect_equal(robust$mi.scaled, baseline_robust$mi.scaled, tolerance = 1e-7)
        }
      }
    }
  }
})

test_that("ordinal fixed rows and equality releases use the same rank and moment scale", {
  syntax <- paste("f =~ x1+1*x2+a*x3+b*x4", "a == b",
                   "x1 | 0*t1+t2", "x2 | t1+t2", "x3 | t1+t2", "x4 | t1+t2",
                   "x1 ~~ 0*x3", sep = "\n")
  for (parameterization in c("delta", "theta")) {
    for (grouped in c(FALSE, TRUE)) {
      d <- ordinal_score_rank_data(grouped)
      for (estimator in c("ULS", "DWLS", "WLS")) {
        fit <- fit_model(syntax, d, estimator = estimator, ordered = paste0("x", 1:4),
                         parameterization = parameterization,
                         groups = if (grouped) "g" else NULL)
        expect_true(fit$converged)
        for (release in c(FALSE, TRUE)) {
          ordinary <- if (release) score_tests(fit) else modification_indices(fit)
          robust <- if (release) score_tests_robust(fit, estimated_weight = FALSE) else modification_indices_robust(fit, estimated_weight = FALSE)
          expect_gt(nrow(ordinary), 0L)
          expect_identical(ordinal_score_rank_key(ordinary), ordinal_score_rank_key(robust))
          expect_equal(ordinary$mi, robust$mi, tolerance = 1e-8)
          expect_equal(ordinary$epc, robust$epc, tolerance = 1e-8)
          if (estimator == "WLS") {
            expect_equal(robust$mi.scaled, ordinary$mi, tolerance = 1e-7)
          }
        }
        if (estimator != "ULS") {
          base_release <- score_tests(fit)
          base_robust <- score_tests_robust(fit, estimated_weight = FALSE)
          slot <- if (estimator == "DWLS") "W_dwls" else "W_wls"
          for (scale in c(1e-16, 1e12)) {
            scaled <- fit
            scaled$ordinal_stats[[slot]] <- lapply(fit$ordinal_stats[[slot]], function(w) scale * w)
            ordinary <- score_tests(scaled)
            robust <- score_tests_robust(scaled, estimated_weight = FALSE)
            expect_identical(ordinal_score_rank_key(ordinary), ordinal_score_rank_key(base_release))
            expect_identical(ordinal_score_rank_key(robust), ordinal_score_rank_key(ordinary))
            expect_equal(ordinary$mi / scale, base_release$mi, tolerance = 1e-7)
            expect_equal(robust$mi.scaled, base_robust$mi.scaled, tolerance = 1e-7)
          }
        }
        mi <- modification_indices(fit)
        expect_true(any(mi$op == "=~" & mi$rhs == "x1"))
        expect_true(any(mi$op == "=~" & mi$rhs == "x2"))
        expect_true(any(mi$op == "|" & mi$lhs == "x1"))
        # An explicit fixed-zero row and an absent row have identical moment units.
        absent <- fit
        pt <- absent$partable
        absent$partable <- pt[!(pt$op == "~~" & pt$lhs == "x1" & pt$rhs == "x3"), ]
        mi_absent <- modification_indices(absent)
        key <- ordinal_score_rank_key(mi)
        index <- match(key, ordinal_score_rank_key(mi_absent))
        expect_false(anyNA(index))
        expect_equal(mi$mi, mi_absent$mi[index], tolerance = 1e-8)
        expect_equal(mi$epc, mi_absent$epc[index], tolerance = 1e-8)
      }
    }
  }
})
