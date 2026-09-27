#!/usr/bin/env Rscript
# Targeted follow-up to the pilot, selected after seeing its results. This does
# not change the frozen sphere reference or contribute to a default decision.
script <- normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
here <- dirname(dirname(script))
source(file.path(here, "..", "..", "_support", "R", "helpers.R"))
set_single_threaded_math()
source(file.path(here, "R", "designs.R")); source(file.path(here, "R", "fit.R"))
args <- commandArgs(trailingOnly = TRUE)
run <- if (length(args)) args[1] else "reference-pilot"
out <- file.path(here, "results", run)
cmp <- read.csv(file.path(out, "comparisons.csv"), stringsAsFactors = FALSE)
selected <- cmp[cmp$route == "lavaan" & cmp$comparison == "no_sphere_reference", ]
if (!nrow(selected)) stop("no qualifying lavaan witness in this run")
rows <- list(); spec <- magmaanlab::model_spec(model_syntax)
for (i in seq_len(nrow(selected))) {
  x <- selected[i, ]; scale <- switch(x$transform, native = 1, x0.01 = .01, x100 = 100)
  data <- draw_data(design_sigma(designs_all()[[x$design]]), x$n, x$seed) * scale
  sample <- sample_moments(data)
  fit <- suppressWarnings(lavaan::sem(model_syntax, data = data, meanstructure = FALSE))
  from <- magmaanlab::frontier_reidentify(lavaan::parTable(fit), spec)
  ev <- magmaanlab::magmaan_core$evaluate_at(spec$partable, sample, from$theta, estimator = "ML")
  stopifnot(abs(ev$fmin - x$objective) < 1e-8)
  for (backend in c("nlopt-lbfgs", "port")) {
    result <- run_fit(spec, data, sample, "ML", "sphere", backend, "lavaan_witness", from$theta)
    rows[[length(rows) + 1L]] <- cbind(x[c("design", "n", "rep", "seed", "transform")],
      backend = backend, witness_objective = x$objective, result$record)
  }
}
write.csv(do.call(rbind, rows), file.path(out, "witness_probe.csv"), row.names = FALSE)
print(do.call(rbind, rows)[c("design", "n", "rep", "backend", "label", "objective", "witness_objective")])
