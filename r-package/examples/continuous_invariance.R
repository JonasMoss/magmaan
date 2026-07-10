## Continuous measurement invariance: configural -> metric -> scalar.
## The scalar helper adds `f ~ c(0, NA)*1` before it constrains observed
## intercepts, matching the latent-mean identification required by lavaan.

suppressMessages(library(magmaan))

set.seed(601)
n <- c(180L, 220L)
lambda <- c(1, .8, .9, .7, .85, .75)
intercept <- c(0, .2, -.1, .15, -.2, .1)
draw_group <- function(n, latent_mean) {
  eta <- rnorm(n, mean = latent_mean)
  x <- vapply(seq_along(lambda), function(j) {
    intercept[j] + lambda[j] * eta + rnorm(n, sd = .6)
  }, numeric(n))
  colnames(x) <- paste0("x", seq_along(lambda))
  as.data.frame(x)
}
dat <- rbind(draw_group(n[1], 0), draw_group(n[2], .35))
dat$school <- rep(c("A", "B"), n)

res <- continuous_invariance(
  "f =~ x1 + x2 + x3 + x4 + x5 + x6",
  dat, group = "school", estimator = "ML")

stopifnot(identical(res$tests$A.method, c("exact", "delta")))
scalar_pt <- res$fits$scalar$partable
latent_mean_g2 <- scalar_pt$lhs == "f" & scalar_pt$op == "~1" &
  scalar_pt$group == 2L
stopifnot(sum(latent_mean_g2) == 1L, scalar_pt$free[latent_mean_g2] > 0L)

print(res)
cat("continuous_invariance(): ok\n")
