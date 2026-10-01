#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(magmaanlab))

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name, default) {
  prefix <- paste0("--", name, "=")
  hit <- grep(paste0("^", prefix), args, value = TRUE)
  if (length(hit) == 0L) return(default)
  sub(prefix, "", hit[[1L]], fixed = TRUE)
}

n <- as.integer(get_arg("n", "3000"))
p <- as.integer(get_arg("p", "40"))
reps <- as.integer(get_arg("reps", "10"))
rounds <- as.integer(get_arg("rounds", "7"))
seed_base <- as.integer(get_arg("seed-base", "20260601"))

# Self-contained 2-factor OSF populations (p in {10, 20, 40}). Inlined so this
# advisory check does not source another leaf (an experiment); the same numbers
# live in experiments/17's population.R, but cpp/tests/ must stay a sink.
.osf_models <- list(
  `10` = list(
    f1_load  = c(0.4, 0.5, 0.6, 0.8, 0.4),
    f1_resid = c(0.84, 0.75, 0.64, 0.36, 0.84),
    f2_load  = c(0.4, 0.7, 0.6, 0.4, 0.8),
    f2_resid = c(0.84, 0.51, 0.64, 0.84, 0.36)),
  `20` = list(
    f1_load  = c(0.4, 0.5, 0.6, 0.8, 0.4, 0.7, 0.8, 0.6, 0.6, 0.3),
    f1_resid = c(0.84, 0.75, 0.64, 0.36, 0.84, 0.51, 0.36, 0.64, 0.64, 0.91),
    f2_load  = c(0.4, 0.7, 0.6, 0.4, 0.8, 0.8, 0.4, 0.7, 0.5, 0.6),
    f2_resid = c(0.84, 0.51, 0.64, 0.84, 0.36, 0.36, 0.84, 0.51, 0.75, 0.64)),
  `40` = list(
    f1_load  = c(0.4, 0.5, 0.6, 0.8, 0.4, 0.7, 0.8, 0.6, 0.6, 0.3,
                 0.4, 0.4, 0.6, 0.5, 0.7, 0.5, 0.7, 0.8, 0.5, 0.7),
    f1_resid = c(0.84, 0.75, 0.64, 0.36, 0.84, 0.51, 0.36, 0.64, 0.64, 0.91,
                 0.84, 0.84, 0.64, 0.75, 0.51, 0.75, 0.51, 0.36, 0.75, 0.51),
    f2_load  = c(0.4, 0.7, 0.6, 0.4, 0.8, 0.8, 0.4, 0.7, 0.5, 0.6,
                 0.6, 0.4, 0.7, 0.4, 0.5, 0.7, 0.8, 0.4, 0.5, 0.3),
    f2_resid = c(0.84, 0.51, 0.64, 0.84, 0.36, 0.36, 0.84, 0.51, 0.75, 0.64,
                 0.64, 0.84, 0.51, 0.84, 0.75, 0.51, 0.36, 0.84, 0.75, 0.91))
)
build_population_2factor <- function(p, interfactor = 0.5) {
  m <- .osf_models[[as.character(p)]]
  if (is.null(m)) stop("no OSF population for p = ", p, call. = FALSE)
  per_factor <- as.integer(p / 2L)
  Lambda <- matrix(0.0, p, 2L)
  Lambda[seq_len(per_factor), 1L] <- m$f1_load
  Lambda[(per_factor + 1L):p, 2L] <- m$f2_load
  Phi <- matrix(c(1, interfactor, interfactor, 1), 2L, 2L)
  Sigma <- Lambda %*% Phi %*% t(Lambda) + diag(c(m$f1_resid, m$f2_resid))
  list(Sigma = Sigma, p = p)
}
pop <- build_population_2factor(p)
core <- magmaanlab::magmaan_core

cal <- core$sim_ig_calibrate(
  pop$Sigma, rep(3, pop$p), rep(21, pop$p),
  root = "symmetric", generator_family = "pearson",
  quadrature_points = 81L
)

time_magmaan <- numeric(rounds)
for (i in seq_len(rounds)) {
  time_magmaan[[i]] <- system.time(core$sim_ig_draw(
    cal, n = n, reps = reps, seed_base = seed_base + 1000L * i,
    quadrature_points = 81L
  ))[["elapsed"]]
}

cat(sprintf("case: p=%d n=%d reps=%d rounds=%d\n", p, n, reps, rounds))
cat("magmaan pearson types:\n")
print(table(vapply(cal$generator_marginals, `[[`, integer(1), "pearson_type")))
cat(sprintf(
  "magmaan median warm: %.4f s total, %.4f s/rep\n",
  median(time_magmaan[-1L]), median(time_magmaan[-1L]) / reps
))

covsim_path <- "external/r_source/covsim/R/IG.R"
if (file.exists(covsim_path)) {
  source(covsim_path)
  time_covsim <- numeric(rounds)
  for (i in seq_len(rounds)) {
    set.seed(seed_base + 2000L * i)
    time_covsim[[i]] <- system.time(rIG(
      N = n, sigma.target = pop$Sigma, skewness = rep(3, pop$p),
      excesskurtosis = rep(21, pop$p), reps = reps, typeA = "symm"
    ))[["elapsed"]]
  }
  cat(sprintf(
    "covsim median warm:  %.4f s total, %.4f s/rep\n",
    median(time_covsim[-1L]), median(time_covsim[-1L]) / reps
  ))
  cat(sprintf(
    "speedup: %.2fx\n",
    median(time_covsim[-1L]) / median(time_magmaan[-1L])
  ))
} else {
  cat(sprintf("covsim source not found at %s; skipped comparison\n", covsim_path))
}

# Foldnes-Gronneberg longitudinal IG3: two of ten Cholesky sources are IV.
# Keep this population local to the check; no paper files are read.
k <- 5L
loadings <- cbind(c(rep(0.7, k), rep(0, k)), c(rep(0, k), rep(0.7, k)))
residual <- diag(rep(0.51, 2L * k))
for (j in seq_len(k)) residual[j, k + j] <- residual[k + j, j] <- 0.15 * 0.51
sigma <- loadings %*% matrix(c(1, 0.3, 0.3, 1), 2L) %*% t(loadings) + residual
type4_cal <- core$sim_ig_calibrate(
  sigma, rep(2, 2L * k), rep(10, 2L * k),
  root = "cholesky", generator_family = "pearson"
)
types <- vapply(type4_cal$generator_marginals, `[[`, integer(1), "pearson_type")
stopifnot(sum(types == 4L) == 2L)
time_type4 <- vapply(seq_len(rounds), function(i) {
  system.time(core$sim_ig_draw(
    type4_cal, n = n, reps = reps, seed_base = seed_base + 3000L * i
  ))[["elapsed"]]
}, numeric(1))
cat(sprintf("longitudinal IG3: p=10 n=%d reps=%d (2 type IV, 8 type VI)\n", n, reps))
cat(sprintf("magmaan median warm: %.4f s total, %.6f s/rep\n",
            median(time_type4[-1L]), median(time_type4[-1L]) / reps))

# Single-source IG isolates draw cost at each family, with enough values to
# resolve sub-microsecond costs using R's elapsed timer.
for (excess in c(10, 7)) {
  scalar_cal <- core$sim_ig_calibrate(
    matrix(1), 2, excess, root = "cholesky", generator_family = "pearson"
  )
  scalar_n <- max(n, 100000L)
  times <- vapply(seq_len(rounds), function(i) {
    system.time(core$sim_ig_draw(
      scalar_cal, n = scalar_n, reps = 1L, seed_base = seed_base + 4000L * i
    ))[["elapsed"]]
  }, numeric(1))
  cat(sprintf("type %d: %.3f microseconds/value (%d values)\n",
              scalar_cal$generator_marginals[[1L]]$pearson_type,
              1e6 * median(times[-1L]) / scalar_n, scalar_n))
}
