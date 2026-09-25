# Frontier sphere chart: frontier_fit_sphere() reproduces the ordinary fit
# whenever the model's identification contains the estimate, signals a classed
# condition when it does not, and frontier_reidentify() re-expresses fits.

sphere_sim <- function(n = 300L, seed = 7L, loadings = c(1, 0.8, 0.6, 0.7)) {
  set.seed(seed)
  eta <- rnorm(n, sd = 1.1)
  x <- sapply(seq_along(loadings), function(j)
    0.1 * j + loadings[j] * eta + rnorm(n, sd = 0.7))
  colnames(x) <- paste0("x", seq_along(loadings))
  as.data.frame(x)
}

ernst_sim <- function(n = 200L, seed = 3L) {
  set.seed(seed)
  X <- rnorm(n)
  Y <- 0.4 * X + rnorm(n)
  lam <- c(1, 0.8, 0.6)
  out <- cbind(sapply(lam, function(l) l * X + rnorm(n)),
               sapply(lam, function(l) l * Y + rnorm(n)))
  colnames(out) <- c("x1", "x2", "x3", "y1", "y2", "y3")
  as.data.frame(out)
}

ernst <- "X =~ x1 + x2 + x3\n Y =~ y1 + y2 + y3\n Y ~ X"

test_that("sphere ML reproduces the ordinary ML fit", {
  dat <- ernst_sim()
  ord <- fit_model(ernst, dat)
  sph <- frontier_fit_sphere(ernst, dat)
  expect_s3_class(sph, "magmaan_fit")
  expect_true(sph$converged)
  expect_equal(sph$options$chart, "sphere")
  expect_equal(nrow(sph$gauge$units), 2L)
  expect_equal(nrow(sph$gauge$passthrough), 0L)
  expect_equal(sph$fmin, ord$fmin, tolerance = 1e-8)
  expect_equal(sph$partable$est, ord$partable$est, tolerance = 1e-5)
  expect_lt(sph$gauge$pin_residual, 1e-6)
})

test_that("std.lv and multi-group metric invariance round-trip", {
  dat <- ernst_sim()
  ord <- fit_model(ernst, dat, std_lv = TRUE)
  sph <- frontier_fit_sphere(ernst, dat, std_lv = TRUE)
  expect_equal(sph$gauge$units$kind, c("linear", "linear"))
  expect_equal(sph$partable$est, ord$partable$est, tolerance = 1e-5)

  g <- rbind(cbind(sphere_sim(seed = 1L), g = "a"),
             cbind(sphere_sim(seed = 2L, loadings = c(1, 0.8, 0.6, 0.7) * 1.2),
                   g = "b"))
  m <- "f =~ x1 + x2 + x3 + x4"
  ord <- fit_model(m, g, groups = "g", group_equal = "loadings")
  sph <- frontier_fit_sphere(m, g, groups = "g", group_equal = "loadings")
  expect_equal(sph$gauge$units$blocks, "1,2")
  expect_equal(sph$partable$est, ord$partable$est, tolerance = 1e-5)
})

test_that("ULS, GLS, FIML and psd = TRUE reproduce their ordinary fits", {
  dat <- ernst_sim()
  for (est in c("ULS", "GLS")) {
    ord <- fit_model(ernst, dat, estimator = est)
    sph <- frontier_fit_sphere(ernst, dat, estimator = est)
    expect_equal(sph$partable$est, ord$partable$est, tolerance = 1e-5)
  }
  miss <- dat
  miss$x2[seq(3, nrow(miss), by = 7)] <- NA
  miss$y3[seq(5, nrow(miss), by = 11)] <- NA
  ord <- fit_model(ernst, miss, estimator = "FIML")
  sph <- frontier_fit_sphere(ernst, miss, estimator = "FIML")
  expect_equal(sph$fmin, ord$fmin, tolerance = 1e-8)
  expect_equal(sph$partable$est, ord$partable$est, tolerance = 1e-4)

  psd <- frontier_fit_ml_psd(ernst, dat, preconditioning = "none")
  sph <- frontier_fit_sphere(ernst, dat, psd = TRUE)
  expect_equal(sph$fmin, psd$fmin, tolerance = 1e-7)
  expect_equal(sph$partable$est, psd$partable$est, tolerance = 1e-4)
})

test_that("a marker at a pole signals a classed condition with the sphere solution", {
  dat <- sphere_sim()
  set.seed(99)
  dat$x1 <- rnorm(nrow(dat))
  S <- stats::cov(dat)
  S[1, -1] <- S[-1, 1] <- 0
  data <- list(S = list(S), nobs = nrow(dat))
  cond <- tryCatch(frontier_fit_sphere("f =~ x1 + x2 + x3 + x4", data),
                   magmaan_user_chart_singular = function(e) e)
  expect_s3_class(cond, "magmaan_user_chart_singular")
  expect_true(cond$gauge$units$singular)
  expect_lt(abs(cond$gauge$units$direction_level), 1e-6)

  moved <- frontier_reidentify(cond, "f =~ NA*x1 + 1*x2 + x3 + x4")
  expect_true(all(is.finite(moved$theta)))
  expect_error(frontier_reidentify(cond, "f =~ x1 + x2 + x3 + x4"),
               "does not contain")
})

test_that("frontier_reidentify moves a fit between identifications", {
  dat <- ernst_sim()
  ord <- fit_model(ernst, dat)
  std <- frontier_reidentify(ord, ernst, std_lv = TRUE)
  direct <- fit_model(ernst, dat, std_lv = TRUE)
  expect_equal(std$partable$est, direct$partable$est, tolerance = 1e-5)
  expect_error(frontier_reidentify(ord, "X =~ x1 + 0.5*x2 + x3\n Y =~ y1 + y2 + y3\n Y ~ X"),
               "different models")
})

test_that("refitting helpers refit a sphere fit through the sphere", {
  dat <- ernst_sim(n = 120L)
  ord <- fit_model(ernst, dat)
  sph <- frontier_fit_sphere(ernst, dat)
  cr_s <- case_rerun(sph, dat, to_rerun = 1:3)
  cr_o <- case_rerun(ord, dat, to_rerun = 1:3)
  expect_true(all(vapply(cr_s$rerun, function(f) identical(f$options$chart, "sphere"),
                         logical(1))))
  expect_equal(sapply(cr_s$rerun, `[[`, "theta"), sapply(cr_o$rerun, `[[`, "theta"),
               tolerance = 1e-5)
  mi_s <- modification_indices_lrt(sph, dat, candidates = "loadings", robust = FALSE)
  mi_o <- modification_indices_lrt(ord, dat, candidates = "loadings", robust = FALSE)
  key <- function(x) x[order(x$lhs, x$rhs), c("mi", "lrt")]
  expect_equal(key(mi_s), key(mi_o), tolerance = 1e-5, ignore_attr = TRUE)
})

test_that("a vanishing endogenous residual variance is the std.lv pole", {
  # Y = 0.8 X exactly: std.lv's fixed residual variance cannot hold the point,
  # the marker identification can.
  L <- rbind(cbind(c(1, 0.8, 0.6), 0), cbind(0, c(1, 0.8, 0.6)))
  Phi <- matrix(c(1, 0.8, 0.8, 0.64), 2)
  S <- L %*% Phi %*% t(L) + diag(0.5, 6)
  # Data whose sample covariance is S exactly.
  set.seed(11)
  z <- scale(matrix(rnorm(400 * 6), 400), scale = FALSE)
  z <- z %*% solve(chol(cov(z))) %*% chol(S)
  dat <- as.data.frame(z)
  names(dat) <- c("x1", "x2", "x3", "y1", "y2", "y3")
  cond <- tryCatch(frontier_fit_sphere(ernst, dat, std_lv = TRUE),
                   magmaan_user_chart_singular = function(e) e)
  expect_s3_class(cond, "magmaan_user_chart_singular")
  expect_true(cond$gauge$units$singular[cond$gauge$units$latent == "Y"])
  expect_lt(cond$gauge$fmin_sphere, 1e-8)
  moved <- frontier_reidentify(cond, ernst)
  expect_lt(max(abs(moved$theta)), 10)
})

test_that("unsupported inputs are refused, not fitted in the user chart", {
  dat <- ernst_sim()
  expect_error(frontier_fit_sphere(ernst, dat, cluster = "g"), "two-level")
  expect_error(frontier_fit_sphere(ernst, dat, estimator = "ULS", psd = TRUE),
               "supports estimator = 'ML' only")
})

test_that("the canonical start gives one sphere solution for every identification", {
  dat <- ernst_sim(n = 150L)
  m2 <- "X =~ NA*x1 + 1*x2 + x3\n Y =~ NA*y1 + 1*y2 + y3\n Y ~ X"
  fits <- list(marker = frontier_fit_sphere(ernst, dat),
               marker2 = frontier_fit_sphere(m2, dat),
               std_lv = frontier_fit_sphere(ernst, dat, std_lv = TRUE),
               effect = frontier_fit_sphere(ernst, dat, effect_coding = TRUE))
  for (f in fits) expect_identical(f$gauge$start, "canonical")
  sphere_abs <- function(f) {
    p <- f$gauge$sphere_partable
    abs(p$est[p$free > 0 & p$op != "=="][order(paste(p$lhs, p$op, p$rhs)[p$free > 0 & p$op != "=="])])
  }
  # Same objective to rounding; parameters to optimizer precision along the
  # flattest direction.
  for (f in fits[-1]) {
    expect_equal(f$gauge$fmin_sphere, fits$marker$gauge$fmin_sphere, tolerance = 1e-12)
    expect_equal(sphere_abs(f), sphere_abs(fits$marker), tolerance = 1e-7)
  }
  for (f in fits) expect_true(f$gauge$driven_scaled)
  moved <- frontier_reidentify(fits$marker, ernst, std_lv = TRUE)
  expect_equal(moved$theta, fits$std_lv$theta, tolerance = 1e-6)
  # Explicit start values are in the user's identification: user start.
  hinted <- frontier_fit_sphere("X =~ x1 + start(0.8)*x2 + x3\n Y =~ y1 + y2 + y3\n Y ~ X", dat)
  expect_identical(hinted$gauge$start, "user")
  expect_identical(frontier_fit_sphere(ernst, dat, start = "user")$gauge$start, "user")
})
