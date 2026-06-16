# Non-normal samplers for the invariance study.
#
# IMPLEMENTED: `norm` (per-group multivariate normal). This is all the smoke
# needs -- the smoke validates the pipeline wiring (fits, every p-value, the
# output schema), not the non-normality science, and normal data is the cleanest
# possible pipeline check.
#
# TODO (port from the experiment-17 families, lifting into experiments/_support
# so this and exp 23/24 stop carrying parallel copies): vm1/2 (Vale-Maurelli),
# ig1/2 (independent generator), pl1/2 (piecewise-linear, Foldnes & Gronneberg
# 2022). Until ported, requesting them errors loudly rather than silently
# falling back to normal.

`%||%` <- function(a, b) if (is.null(a)) b else a

# Build a sampler bound to a population, per-group sample sizes, and an optional
# violation. `draw(rep_i)` returns a stacked data.frame: x1..xp plus `school`
# (group label "A"/"B"). Seeding is per-replicate and deterministic.
make_sampler <- function(pop, ns, dist = "norm", violate = NULL,
                         seed_base = 0L) {
  if (!identical(dist, "norm")) {
    stop("non-normal family '", dist, "' not ported yet; smoke uses dist=norm ",
         "(see report.qmd / generators.R TODO)", call. = FALSE)
  }
  moms <- lapply(1:2, function(g) group_moments(pop, g, violate))
  labels <- c("A", "B")
  list(
    draw = function(rep_i) {
      set.seed(seed_base + rep_i)
      parts <- lapply(1:2, function(g) {
        X <- MASS::mvrnorm(ns[g], mu = moms[[g]]$mu, Sigma = moms[[g]]$Sigma)
        df <- as.data.frame(X)
        names(df) <- pop$ov
        df$school <- labels[g]
        df
      })
      do.call(rbind, parts)
    }
  )
}
