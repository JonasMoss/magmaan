#!/usr/bin/env Rscript
# Deng & Chan (2017) alpha vs omega: a non-regular Wald contrast, an Imhof fix,
# and the regular trinity (LR, score).
#
# Testing H0: alpha = omega is, within a one-factor model, testing
# tau-equivalence (equal loadings). Because omega >= alpha always, the contrast
# omega - alpha sits at the floor of its range under H0, its gradient vanishes,
# and Deng & Chan's Wald z is NON-REGULAR exactly at the null -- it converges to
# a quadratic form, not a normal, and the z-test is mis-calibrated. The fix
# keeps the interpretable reliability gap as the statistic but references it
# against its true weighted-chi-square law via Imhof. We compare it to the
# regular (p-1)-df tests of the same null: the Satorra-2000 scaled LR and a
# standard score test.
#
# Estimation is traditional normal-theory ML throughout. magmaan supplies the
# fits and the Gamma helpers; the Imhof tail comes from CompQuadForm. No magmaan
# C++ core is added.
#
# Parts:
#   0. diagnostic  -- SD(omega-alpha) ~ 1/N (second-order) while SD(omega) ~
#                     1/sqrt(N), and corr(omega,alpha) -> 1 at the null.
#   1. validation  -- alpha == omega(ULS-tau) identity; sandwich SE vs bootstrap.
#   2. demo        -- every test side by side on real HS subscales.
#   3. simulation  -- Type I / power / divergence x {normal, non-normal}.
#
# Usage:
#   Rscript run_experiment.R [--reps N] [--N CSV] [--cells FILTER]
#                            [--boot-B B] [--seed-base S] [--no-sim] [--smoke]

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
source(experiment_path("R", "diagnostics.R"))

# ---- arguments ---------------------------------------------------------------

parse_args <- function(args) {
  out <- list(reps = 300L, N = c(250L, 500L), cells_filter = NULL,
              boot_B = 1000L, seed_base = 20260602L, do_sim = TRUE,
              smoke = FALSE)
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    if (a %in% c("-h", "--help")) {
      cat("Usage: Rscript run_experiment.R [--reps N] [--N CSV] [--cells FILTER]",
          "[--boot-B B] [--seed-base S] [--no-sim] [--smoke]\n")
      cat("  --reps      replications per simulation cell (default 300)\n")
      cat("  --N         comma-separated sample sizes (default 250,500)\n")
      cat("  --cells     filter sim grid, e.g. pop=tau,dist=normal,N=250\n")
      cat("  --boot-B    bootstrap draws for the SE validation (default 1000)\n")
      cat("  --seed-base base seed (default 20260602)\n")
      cat("  --no-sim    run diagnostic + validation + demo only\n")
      cat("  --smoke     fast path: tiny reps, one N, normal only\n")
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
    out$reps <- 8L; out$N <- 250L; out$boot_B <- 200L
    out$cells_filter <- out$cells_filter %||% "dist=normal"
  }
  out
}

cfg <- parse_args(commandArgs(trailingOnly = TRUE))
suppressMessages({ library(magmaan); library(lavaan) })
require_pkg("CompQuadForm", "install.packages('CompQuadForm')")
set_single_threaded_math()
res_dir <- ensure_results_dir()

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

# ---- part 0: non-regularity diagnostic ---------------------------------------

cat("[0/3] diagnostic: is omega-alpha a second-order statistic at the null?\n")
diag_reps <- if (cfg$smoke) 60L else 300L
nonreg <- nonregularity_scaling(reps = diag_reps, seed_base = cfg$seed_base)
write_csv(nonreg, file.path(res_dir, "nonregularity.csv"))
cat(sprintf("  SD(omega-alpha) ratio per Nx4: %s  (1/N ~0.25)\n",
            paste(sprintf("%.2f", nonreg$sd_diff_ratio[-1]), collapse = ", ")))
cat(sprintf("  SD(omega) ratio per Nx4:       %s  (1/sqrt(N) ~0.50)\n",
            paste(sprintf("%.2f", nonreg$sd_omega_ratio[-1]), collapse = ", ")))
cat(sprintf("  corr(omega,alpha) -> %.4f\n", nonreg$corr_omega_alpha[nrow(nonreg)]))

# ---- part 1: validation ------------------------------------------------------

cat("[1/3] validation: alpha identity + sandwich-SE vs bootstrap\n")
hs6 <- lavaan::HolzingerSwineford1939[, paste0("x", 1:6)]
alpha_hs <- alpha_from_S(mle_cov(hs6))
omega_uls_hs <- omega_uls_tau(hs6)
identity_rows <- data.frame(
  dataset = "HolzingerSwineford x1-x6",
  alpha = alpha_hs, omega_uls_tau = omega_uls_hs,
  abs_diff = abs(alpha_hs - omega_uls_hs)
)
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

# ---- part 2: real-data demonstration -----------------------------------------

cat("[2/3] demo: every test on Holzinger-Swineford subscales\n")
demo_scales <- list(
  `visual (x1-x3)`          = paste0("x", 1:3),
  `textual (x4-x6)`         = paste0("x", 4:6),
  `speed (x7-x9)`           = paste0("x", 7:9),
  `visual+textual (x1-x6)*` = paste0("x", 1:6)
)
demo_rows <- lapply(names(demo_scales), function(nm) {
  d <- lavaan::HolzingerSwineford1939[, demo_scales[[nm]]]
  rt <- reliability_tests(d); st <- satorra_tau_test(d); sc <- score_tau_test(d)
  data.frame(scale = nm, p = ncol(d),
             omega = rt$omega, alpha = rt$alpha, diff = rt$diff,
             p_wald_sw = rt$p_wald_sw, p_imhof_sw = rt$p_imhof_sw,
             p_satorra_scaled = st$p_scaled, p_score = sc$p_score)
})
demo_rows <- do.call(rbind, demo_rows)
write_csv(demo_rows, file.path(res_dir, "demo.csv"))

# ---- part 3: simulation ------------------------------------------------------

if (cfg$do_sim) {
  grid <- expand.grid(
    population = c("tau", "congeneric", "misspecified"),
    dist = c("normal", "nonnormal"),
    N = cfg$N, p = 6L, stringsAsFactors = FALSE
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
      cbind(rep = r, run_one_rep(draw_sample(pop, cell$N, cell$dist)))
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
      reject_imhof_nm = rejection_rate(reps$p_imhof_nm),
      reject_imhof_sw = rejection_rate(reps$p_imhof_sw),
      reject_satorra_scaled = rejection_rate(reps$p_satorra_scaled),
      reject_satorra_mixture = rejection_rate(reps$p_satorra_mixture),
      reject_score = rejection_rate(reps$p_score),
      n_rel_ok = sum(reps$rel_converged), n_st_ok = sum(reps$st_converged),
      row.names = NULL
    )
    cat(sprintf("  %-12s %-9s N=%-4d  wald.sw=%.3f  imhof.sw=%.3f  satorra=%.3f  score=%.3f\n",
                cell$population, cell$dist, cell$N,
                summ_list[[gi]]$reject_wald_sw, summ_list[[gi]]$reject_imhof_sw,
                summ_list[[gi]]$reject_satorra_scaled, summ_list[[gi]]$reject_score))
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
    estimation = "normal-theory ML", imhof = "CompQuadForm",
    cells_filter = cfg$cells_filter %||% "",
    nonnormal = "centred-scaled chi-square(5) components"
  ),
  packages = c("magmaan", "lavaan", "CompQuadForm")
)

cat("\nwrote results to:", res_dir, "\n")
