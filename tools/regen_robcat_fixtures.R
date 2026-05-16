#!/usr/bin/env Rscript
# Generate frozen robcat-parity fixtures for the robust-polychoric sanity suite.
#
# robcat (Welz, Mair & Alfons, 2026) is treated as the *canonical* robust
# polychoric estimator. Each case here gets:
#
#   tests/fixtures/robcat/<id>.json  -- a Kx x Ky contingency table plus
#                                       robcat's MLE and robust C-estimator
#                                       outputs (rho, thresholds, objective,
#                                       stderr, vcov).
#
# The C++ golden test tests/golden/robcat_parity_golden_test.cpp consumes these
# with no R at run time. This is a manual developer step; CI never runs R. It
# is the sibling of tools/regen_oracle.R and tools/regen_parity_fixtures.R.
#
# Estimator equivalence (verified, see the C++ test header):
#   robcat polycor(c=C)  minimises  sum_k p_k * rho_fun(f_hat_k / p_k)  with
#   rho_fun(t) = t*log(t) for t <= C+1 and t*(log(C+1)+1) - (C+1) beyond.
#   magmaan WmaHardCap phi(t) is identical with cap k = C+1. Hence
#   robcat polycor(c=C)  == magmaan WmaHardCap(k = C+1), and
#   robcat polycor_mle   == magmaan ML (= WmaHardCap, k = Inf).
#
# Usage:
#   Rscript tools/regen_robcat_fixtures.R                  # all cases
#   Rscript tools/regen_robcat_fixtures.R clean_3x3_moderate ...   # named cases
#
# Requires the pinned robcat version (tests/fixtures/robcat_version.txt). If
# robcat is not installed:
#   R CMD INSTALL resources/robcat_0.2.tar.gz
# (its dependencies -- Rcpp, ggplot2, mvtnorm, stringr, numDeriv, pracma,
# Matrix -- must be available first.)

suppressMessages({
  library(robcat)
  library(mvtnorm)
  library(jsonlite)
})

repo <- normalizePath(file.path(dirname(normalizePath(sub(
  "^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))), ".."))

# --- robcat version pin ----------------------------------------------------
pin_path <- file.path(repo, "tests", "fixtures", "robcat_version.txt")
pinned <- trimws(readLines(pin_path, warn = FALSE)[1])
installed <- as.character(utils::packageVersion("robcat"))
if (!identical(pinned, installed)) {
  stop(sprintf(paste0("robcat version mismatch: pinned %s, installed %s.\n",
                      "  Reinstall the pinned build:\n",
                      "    R CMD INSTALL resources/robcat_0.2.tar.gz"),
               pinned, installed), call. = FALSE)
}

out_dir <- file.path(repo, "tests", "fixtures", "robcat")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# robcat tuning sweep. magmaan's equivalent WMA hard cap is k = c + 1.
c_values <- c(0.0, 0.3, 0.6, 1.0)

# === case table ============================================================
# Each case draws a latent bivariate normal, discretises with finite interior
# thresholds (-Inf/+Inf appended implicitly), optionally injects contamination,
# and tallies a Kx x Ky contingency table. Cases are kept well-conditioned
# (large N, well-separated thresholds, no empty categories) so robcat's
# unconstrained optimiser and magmaan's bounded joint fitter land on the same
# interior optimum -- the comparison then tests the estimator, not the
# constraint handling.
cases <- list(
  clean_3x3_moderate = list(
    seed = 101, n = 2500, rho = 0.40,
    thr_x = c(-0.65, 0.55), thr_y = c(-0.55, 0.60)),

  clean_5x5_strong = list(
    seed = 202, n = 4000, rho = 0.65,
    thr_x = c(-1.10, -0.35, 0.40, 1.15),
    thr_y = c(-1.05, -0.30, 0.45, 1.10)),

  clean_4x4_negative = list(
    seed = 303, n = 2500, rho = -0.35,
    thr_x = c(-0.80, 0.00, 0.85), thr_y = c(-0.75, 0.05, 0.80)),

  clean_2x2 = list(
    seed = 404, n = 2000, rho = 0.50,
    thr_x = c(0.10), thr_y = c(-0.15)),

  skewed_5x5 = list(
    seed = 505, n = 4000, rho = 0.50,
    thr_x = c(-1.60, -0.90, -0.20, 0.60),
    thr_y = c(-1.50, -0.80, -0.10, 0.70)),

  contam_5x5_corner = list(
    seed = 606, n = 4000, rho = 0.60,
    thr_x = c(-1.10, -0.35, 0.40, 1.10),
    thr_y = c(-1.10, -0.35, 0.40, 1.10),
    contamination = list(eps = 0.10, kind = "corner")),

  contam_4x4_uniform = list(
    seed = 707, n = 3000, rho = 0.50,
    thr_x = c(-0.80, 0.00, 0.80), thr_y = c(-0.75, 0.05, 0.75),
    contamination = list(eps = 0.15, kind = "uniform")),

  mild_3x3_lowcorr = list(
    seed = 808, n = 2500, rho = 0.10,
    thr_x = c(-0.60, 0.55), thr_y = c(-0.55, 0.60))
)

# === data generation =======================================================

# Draw a seeded contingency table for one case. Contamination, when present,
# replaces a fraction eps of the responses: "corner" puts careless responders
# in the model-discordant (x = Kx, y = 1) cell; "uniform" scatters them over
# the whole grid.
make_table <- function(id, case) {
  set.seed(case$seed)
  Kx <- length(case$thr_x) + 1L
  Ky <- length(case$thr_y) + 1L
  S <- matrix(c(1.0, case$rho, case$rho, 1.0), 2L, 2L)
  z <- mvtnorm::rmvnorm(case$n, sigma = S)
  x <- as.integer(cut(z[, 1], c(-Inf, case$thr_x, Inf)))
  y <- as.integer(cut(z[, 2], c(-Inf, case$thr_y, Inf)))

  contam <- case$contamination
  if (!is.null(contam)) {
    m <- round(contam$eps * case$n)
    idx <- sample.int(case$n, m)
    if (identical(contam$kind, "corner")) {
      x[idx] <- Kx
      y[idx] <- 1L
    } else if (identical(contam$kind, "uniform")) {
      x[idx] <- sample.int(Kx, m, replace = TRUE)
      y[idx] <- sample.int(Ky, m, replace = TRUE)
    } else {
      stop("unknown contamination kind: ", contam$kind, call. = FALSE)
    }
  }

  tab <- table(factor(x, levels = seq_len(Kx)),
               factor(y, levels = seq_len(Ky)))
  if (any(rowSums(tab) == 0) || any(colSums(tab) == 0)) {
    stop(id, ": generated an empty category -- adjust seed/n/thresholds",
         call. = FALSE)
  }
  tab
}

# Pull rho / thresholds / objective / stderr / vcov out of a robcat result.
# robcat orders parameters [rho, a1..a_{Kx-1}, b1..b_{Ky-1}].
extract <- function(res, Kx, Ky) {
  th <- res$thetahat
  list(
    rho          = unname(th[["rho"]]),
    thresholds_x = unname(th[2:Kx]),
    thresholds_y = unname(th[(Kx + 1L):(Kx + Ky - 1L)]),
    objective    = unname(res$objective),
    stderr       = unname(res$stderr),
    vcov         = unname(as.matrix(res$vcov)))
}

# === emit ==================================================================

emit <- function(id) {
  case <- cases[[id]]
  if (is.null(case)) stop("unknown robcat case: ", id, call. = FALSE)

  tab <- make_table(id, case)
  Kx <- nrow(tab)
  Ky <- ncol(tab)

  mle <- robcat::polycor_mle(tab, variance = TRUE)
  robust <- lapply(c_values, function(cc) {
    r <- robcat::polycor(tab, c = cc, variance = TRUE)
    c(list(c = cc, k_equiv = cc + 1.0), extract(r, Kx, Ky))
  })

  contamination <- if (is.null(case$contamination)) NULL else
    list(eps = case$contamination$eps, kind = case$contamination$kind)

  payload <- list(
    `_meta` = list(
      format_version = 1L,
      fixture_kind   = "robcat_polychoric",
      case_id        = id,
      tool           = "robcat::polycor / polycor_mle",
      robcat_version = installed,
      seed           = case$seed),
    dgp = list(
      rho          = case$rho,
      thresholds_x = case$thr_x,
      thresholds_y = case$thr_y,
      n_obs        = case$n,
      contamination = contamination),
    counts = matrix(as.integer(tab), nrow = Kx, ncol = Ky),
    robcat = list(
      mle    = extract(mle, Kx, Ky),
      robust = robust))

  write_json(payload, file.path(out_dir, paste0(id, ".json")),
             pretty = TRUE, auto_unbox = TRUE, digits = NA, null = "null")

  rob06 <- Find(function(r) isTRUE(all.equal(r$c, 0.6)), robust)
  cat(sprintf("wrote %-20s %dx%d n=%5d  dgp=%+.2f  mle=%+.4f  c0.6=%+.4f%s\n",
              id, Kx, Ky, case$n, case$rho, payload$robcat$mle$rho, rob06$rho,
              if (is.null(contamination)) "" else
                sprintf("  [contam %s eps=%.2f]",
                        contamination$kind, contamination$eps)))
}

args <- commandArgs(trailingOnly = TRUE)
ids <- if (length(args) == 0) names(cases) else args
for (id in ids) emit(id)
