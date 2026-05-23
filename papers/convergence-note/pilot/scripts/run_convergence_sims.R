#!/usr/bin/env Rscript

args <- commandArgs(FALSE)
file_arg <- sub("^--file=", "", grep("^--file=", args, value = TRUE)[1L])
script_dir <- if (!is.na(file_arg)) {
  dirname(normalizePath(file_arg, mustWork = TRUE))
} else {
  file.path("papers", "convergence-note", "pilot", "scripts")
}
source(file.path(script_dir, "pilot-lib.R"))

pilot_ensure_dirs()
pilot_require_magmaan()

reps <- pilot_env_int("PILOT_REPS", 10L)
seed0 <- pilot_env_int("PILOT_SEED", 20260523L)
max_iter <- pilot_env_int("PILOT_MAX_ITER", 5000L)
optimizers <- pilot_env_list(
  "PILOT_OPTIMIZERS",
  c("lbfgs", "nlopt-lbfgs", "nlopt-var2", "port")
)
designs <- pilot_env_list(
  "PILOT_DESIGNS",
  c("dejonckere_simple_2025",
    "dejonckere_crossloading_2025",
    "dejonckere_shrinkage_2023_study1",
    "ludtke_cfa_2021")
)

catalog <- magmaan::convergence_sim_catalog()
unknown <- setdiff(designs, catalog$design)
if (length(unknown)) {
  stop("Unknown convergence-sim design(s): ", paste(unknown, collapse = ", "),
       call. = FALSE)
}

control <- list(max_iter = max_iter, ftol = 1e-10, gtol = 1e-7)
rows <- list()
k <- 0L

for (design in designs) {
  for (rep in seq_len(reps)) {
    sim_seed <- seed0 + match(design, catalog$design) * 100000L + rep
    sim <- magmaan::convergence_sim(design, seed = sim_seed)
    for (optimizer in optimizers) {
      message("sim design=", design, " rep=", rep, " optimizer=", optimizer)
      fit <- pilot_fit_once(sim$analysis_syntax, sim$data, optimizer, control)
      k <- k + 1L
      rows[[k]] <- pilot_result_row(
        data.frame(
          source = "convergence_sim",
          design = design,
          reference = sim$reference,
          replicate = rep,
          seed = sim_seed,
          n = sim$n,
          p = ncol(sim$data),
          estimator = "ML",
          start_policy = "magmaan_r_default",
          optimizer = optimizer,
          stringsAsFactors = FALSE
        ),
        fit
      )
    }
  }
}

raw <- pilot_bind(rows)
pilot_write_csv(raw, pilot_path("results", "convergence_sims_raw.csv"))
