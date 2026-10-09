#!/usr/bin/env Rscript
# Retained pricing failure: pilot-task145 cell_029, exact fixed target seed.
file <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])
leaf <- normalizePath(file.path(dirname(file), ".."))
root <- normalizePath(file.path(leaf, "../../.."))
source(file.path(root, "experiments/_support/R/helpers.R"))
source(support_path("R", "basis.R")); source(support_path("R", "missingness.R"))
for (name in c("config", "populations", "generators", "targets"))
  source(file.path(leaf, "R", paste0(name, ".R")))
set_single_threaded_math()
cell <- as.list(study_config("pilot")[29, ])
seed <- 82733967L
population <- build_population(cell)
calibration <- calibrate_generator(population, cell)
data <- apply_mar(draw_generator(calibration, 200000L, seed), cell, seed + 1000L)
rows <- lapply(c("h1", "baseline"), function(kind) {
  spec <- if (kind == "h1") population_spec(population$record, means = TRUE) else
    independence_spec(population, TRUE)
  fit <- magmaanlab::fit_model(spec, data, estimator = "FIML")
  audit <- fit$diagnostics$newton_accuracy
  data.frame(cell = cell$id, model = cell$model, seed = seed, n = 200000L, kind = kind,
    converged = fit$converged, optimizer_status = fit$optimizer_status,
    stationarity = fit$diagnostics$verdict$stationarity,
    newton_distance = audit$distance, budget = audit$budget, fmin = fit$fmin)
})
path <- file.path(leaf, "results/pricing/pilot-task145/target_failure_diagnostics.csv")
write_csv(do.call(rbind, rows), path)
print(do.call(rbind, rows)); cat("Wrote", path, "\n")
