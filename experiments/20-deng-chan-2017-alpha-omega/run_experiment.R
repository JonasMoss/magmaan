#!/usr/bin/env Rscript
# Deng & Chan (2017) alpha-vs-omega: a Wald contrast vs a Satorra-2000 LRT.
#
# Two asymptotically-valid tests of the same null, H0: alpha = omega (which,
# within a one-factor model, is exactly tau-equivalence / equal loadings):
#
#   * Deng & Chan (2017) Wald: a z-test on the reliability difference
#     (omega_hat - alpha_hat) with a delta-method standard error, in
#     normal-theory (nm) and sandwich (sw) flavours.
#   * Satorra-2000: a scaled likelihood-ratio test of the equal-loading
#     restriction (congeneric one-factor H1 vs tau-equivalent H0).
#
# magmaan supplies the fits and the empirical-Gamma nested test; the Deng-Chan
# tau^2 is assembled in R/alpha_omega.R. No magmaan C++ core is added.
#
# The run has three parts:
#   1. validation  -- alpha == omega(ULS-tau) identity; analytic sandwich SE
#                     vs nonparametric bootstrap SE.
#   2. demo        -- the two verdicts side by side on real Holzinger-Swineford
#                     subscales.
#   3. simulation  -- Type I error / power / divergence across tau-equivalent,
#                     congeneric, and misspecified populations x {normal,
#                     non-normal}.
#
# Usage:
#   Rscript run_experiment.R [--reps N] [--N CSV] [--cells FILTER]
#                            [--boot-B B] [--seed-base S]
#                            [--no-sim] [--smoke]
#
# Examples:
#   Rscript run_experiment.R --smoke
#   Rscript run_experiment.R --reps 2000 --N 250,500
#   Rscript run_experiment.R --cells pop=tau,dist=normal

.support_helpers <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg)) {
    script <- normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE)
  } else {
    ofile <- tryCatch(sys.frames()[[1L]]$ofile, error = function(e) NULL)
    script <- normalizePath(if (is.null(ofile)) "run_experiment.R" else ofile,
                            mustWork = FALSE)
  }
  file.path(dirname(dirname(script)), "_support", "R", "helpers.R")
}
source(.support_helpers())
rm(.support_helpers)
source(experiment_path("R", "alpha_omega.R"))
source(experiment_path("R", "populations.R"))
source(experiment_path("R", "tests.R"))

# ---- arguments ---------------------------------------------------------------

parse_args <- function(args) {
  out <- list(reps = 500L, N = c(250L, 500L), cells_filter = NULL,
              boot_B = 1000L, seed_base = 20260602L, do_sim = TRUE,
              smoke = FALSE)
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    if (a %in% c("-h", "--help")) {
      cat("Usage: Rscript run_experiment.R [--reps N] [--N CSV] [--cells FILTER]",
          "[--boot-B B] [--seed-base S] [--no-sim] [--smoke]\n")
      cat("  --reps      replications per simulation cell (default 500)\n")
      cat("  --N         comma-separated sample sizes (default 250,500)\n")
      cat("  --cells     filter sim grid, e.g. pop=tau,dist=normal,N=250\n")
      cat("  --boot-B    bootstrap draws for the SE validation (default 1000)\n")
      cat("  --seed-base base seed for the simulation (default 20260602)\n")
      cat("  --no-sim    run validation + demo only, skip the simulation\n")
      cat("  --smoke     fast path: tiny reps/bootstrap, one N, normal only\n")
      quit(save = "no", status = 0L)
    } else if (a == "--reps") { i <- i + 1L; out$reps <- as.integer(args[[i]])
    } else if (a == "--N") { i <- i + 1L; out$N <- as.integer(parse_csv_arg(args[[i]]))
    } else if (a == "--cells") { i <- i + 1L; out$cells_filter <- args[[i]]
    } else if (a == "--boot-B") { i <- i + 1L; out$boot_B <- as.integer(args[[i]])
    } else if (a == "--seed-base") { i <- i + 1L; out$seed_base <- as.integer(args[[i]])
    } else if (a == "--no-sim") { out$do_sim <- FALSE
    } else if (a == "--smoke") { out$smoke <- TRUE
    } else stop("unknown argument: ", a, call. = FALSE)
    i <- i + 1L
  }
  if (out$smoke) {
    out$reps <- 15L; out$N <- 250L; out$boot_B <- 200L
    out$cells_filter <- out$cells_filter %||% "dist=normal"
  }
  out
}

cfg <- parse_args(commandArgs(trailingOnly = TRUE))
suppressMessages({ library(magmaan); library(lavaan) })
set_single_threaded_math()
res_dir <- ensure_results_dir()

# Apply a "key=val,key=val" cell filter to the grid data frame.
apply_cells_filter <- function(grid, filter) {
  if (is.null(filter)) return(grid)
  for (kv in parse_csv_arg(filter)) {
    parts <- strsplit(kv, "=", fixed = TRUE)[[1L]]
    key <- parts[[1L]]; val <- parts[[2L]]
    if (!key %in% names(grid)) stop("unknown cell key: ", key, call. = FALSE)
    grid <- grid[as.character(grid[[key]]) == val, , drop = FALSE]
  }
  grid
}

# ---- part 1: validation ------------------------------------------------------

cat("[1/3] validation: alpha identity + sandwich-SE vs bootstrap\n")
hs6 <- lavaan::HolzingerSwineford1939[, paste0("x", 1:6)]

# (a) alpha(S) == omega of the ULS-fitted tau-equivalent model.
alpha_hs <- alpha_from_S(mle_cov(hs6))
omega_uls_hs <- omega_uls_tau(hs6)
identity_rows <- data.frame(
  dataset = "HolzingerSwineford x1-x6",
  alpha = alpha_hs, omega_uls_tau = omega_uls_hs,
  abs_diff = abs(alpha_hs - omega_uls_hs)
)

# (b) analytic sandwich SE of (omega - alpha) vs nonparametric bootstrap, on a
#     real scale and on a synthetic congeneric scale at non-trivial p.
se_validation_one <- function(d, label, B, seed) {
  dc <- deng_chan_test(d)
  boot <- bootstrap_diff_se(d, B = B, seed = seed)
  data.frame(dataset = label, n = nrow(d), diff = dc$diff,
             se_nm = dc$se_nm, se_sw = dc$se_sw, se_bootstrap = boot,
             ratio_sw_boot = dc$se_sw / boot)
}
set.seed(cfg$seed_base)
syn <- draw_sample(population("congeneric", 6L), 500L, "normal")
se_rows <- rbind(
  se_validation_one(hs6, "HolzingerSwineford x1-x6", cfg$boot_B, cfg$seed_base),
  se_validation_one(syn, "synthetic congeneric p=6, N=500", cfg$boot_B,
                    cfg$seed_base + 1L)
)
write_csv(identity_rows, file.path(res_dir, "validation_identity.csv"))
write_csv(se_rows, file.path(res_dir, "validation_se.csv"))
cat(sprintf("  identity |alpha - omega_ULS| = %.2e\n", identity_rows$abs_diff))
cat(sprintf("  sandwich/bootstrap SE ratio: %s\n",
            paste(sprintf("%.3f", se_rows$ratio_sw_boot), collapse = ", ")))

# ---- part 2: real-data demonstration -----------------------------------------

cat("[2/3] demo: two verdicts on Holzinger-Swineford subscales\n")
demo_scales <- list(
  `visual (x1-x3)`          = paste0("x", 1:3),
  `textual (x4-x6)`         = paste0("x", 4:6),
  `speed (x7-x9)`           = paste0("x", 7:9),
  `visual+textual (x1-x6)*` = paste0("x", 1:6)
)
demo_rows <- lapply(names(demo_scales), function(nm) {
  d <- lavaan::HolzingerSwineford1939[, demo_scales[[nm]]]
  dc <- deng_chan_test(d); st <- satorra_tau_test(d)
  data.frame(scale = nm, p = ncol(d),
             omega = dc$omega, alpha = dc$alpha, diff = dc$diff,
             se_sw = dc$se_sw, p_wald_sw = dc$p_sw,
             p_satorra_scaled = st$p_scaled)
})
demo_rows <- do.call(rbind, demo_rows)
write_csv(demo_rows, file.path(res_dir, "demo.csv"))
cat("  (* visual+textual is genuinely two-dimensional: a misspecified-scale illustration)\n")

# ---- part 3: simulation ------------------------------------------------------

if (cfg$do_sim) {
  grid <- expand.grid(
    population = c("tau", "congeneric", "misspecified"),
    dist = c("normal", "nonnormal"),
    N = cfg$N, p = 6L,
    stringsAsFactors = FALSE
  )
  grid <- apply_cells_filter(grid, cfg$cells_filter)
  cat(sprintf("[3/3] simulation: %d cells x %d reps\n", nrow(grid), cfg$reps))

  raw_list <- vector("list", nrow(grid))
  summ_list <- vector("list", nrow(grid))
  for (gi in seq_len(nrow(grid))) {
    cell <- grid[gi, ]
    pop <- population(cell$population, cell$p)
    reps <- lapply(seq_len(cfg$reps), function(r) {
      set.seed(cfg$seed_base + gi * 100000L + r)
      d <- draw_sample(pop, cell$N, cell$dist)
      cbind(rep = r, run_one_rep(d))
    })
    reps <- do.call(rbind, reps)
    reps <- cbind(population = cell$population, dist = cell$dist, N = cell$N,
                  p = cell$p, reps)
    raw_list[[gi]] <- reps

    tg <- population_targets(pop)
    summ_list[[gi]] <- data.frame(
      population = cell$population, dist = cell$dist, N = cell$N, p = cell$p,
      alpha_pop = tg["alpha_pop"], omega_1f = tg["omega_1f"], gap = tg["gap"],
      reject_wald_nm = rejection_rate(reps$p_wald_nm),
      reject_wald_sw = rejection_rate(reps$p_wald_sw),
      reject_satorra_unscaled = rejection_rate(reps$p_satorra_unscaled),
      reject_satorra_scaled = rejection_rate(reps$p_satorra_scaled),
      n_dc_ok = sum(reps$dc_converged), n_st_ok = sum(reps$st_converged),
      row.names = NULL
    )
    cat(sprintf("  %-12s %-9s N=%-4d  wald.sw=%.3f  satorra=%.3f\n",
                cell$population, cell$dist, cell$N,
                summ_list[[gi]]$reject_wald_sw,
                summ_list[[gi]]$reject_satorra_scaled))
  }
  sim_summary <- do.call(rbind, summ_list)
  write_csv(do.call(rbind, raw_list), file.path(res_dir, "simulation_raw.csv"))
  write_csv(sim_summary, file.path(res_dir, "simulation_summary.csv"))
}

# ---- metadata ----------------------------------------------------------------

write_metadata(
  file.path(res_dir, "metadata.csv"),
  values = list(
    reps = cfg$reps, N = cfg$N, p = 6L, boot_B = cfg$boot_B,
    seed_base = cfg$seed_base, do_sim = cfg$do_sim, smoke = cfg$smoke,
    cells_filter = cfg$cells_filter %||% "",
    nonnormal = "centred-scaled chi-square(5) components"
  ),
  packages = c("magmaan", "lavaan")
)

cat("\nwrote:\n")
cat(" ", file.path(res_dir, "validation_identity.csv"), "\n")
cat(" ", file.path(res_dir, "validation_se.csv"), "\n")
cat(" ", file.path(res_dir, "demo.csv"), "\n")
if (cfg$do_sim) {
  cat(" ", file.path(res_dir, "simulation_summary.csv"), "\n")
  cat(" ", file.path(res_dir, "simulation_raw.csv"), "\n")
}
cat(" ", file.path(res_dir, "metadata.csv"), "\n")
