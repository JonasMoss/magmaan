#!/usr/bin/env Rscript
script <- normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
here <- dirname(dirname(script))
source(file.path(here, "..", "..", "..", "_support", "R", "helpers.R"))
set_single_threaded_math()
for (f in c("designs.R", "fit.R", "start_design.R")) source(file.path(here, "R", f))
args <- commandArgs(trailingOnly = TRUE)
if ("--help" %in% args) {
  cat("Usage: Rscript scripts/tune_starts.R --phase witness|development|fresh [--run-id NAME]\n",
      "Witness: the named pilot miss, including sign/direction ablations.\n",
      "Development: the 60 original draws, four spectral sign recipes.\n",
      "Fresh: 60 new draws, canonical plus the same four fixed recipes.\n",
      "ML only; current screen and defaults unchanged. Raw fits/starts stay local.\n", sep = "")
  quit(save = "no")
}
opt <- function(k, default) if (k %in% args) args[match(k, args) + 1L] else default
phase <- opt("--phase", "witness"); stopifnot(phase %in% c("witness", "development", "fresh"))
run <- opt("--run-id", paste0("start-", phase)); stopifnot(grepl("^[a-zA-Z0-9_-]+$", run))
out <- file.path(here, "results", run)
if (file.exists(file.path(out, "fits.csv"))) stop("choose a new --run-id; existing run is retained")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
write_out <- function(x, name) write.csv(x, file.path(out, paste0(name, ".csv")), row.names = FALSE)
pilot <- file.path(here, "results", "reference-pilot")
refs <- read.csv(file.path(pilot, "references.csv"))
old <- if (phase != "fresh") read.csv(file.path(pilot, "fits.csv")) else NULL
if (phase == "witness") {
  tasks <- unique(old[old$design == "weak_marker" & old$n == 20 & old$rep == 7 & old$transform == "native", c("design", "n", "rep", "seed")])
  write_out(unresolved_inventory(old, refs), "unresolved_inventory")
} else if (phase == "development") {
  tasks <- unique(old[old$transform == "native", c("design", "n", "rep", "seed")])
} else {
  tasks <- expand.grid(design = names(designs_all()), n = c(20L, 100L), rep = 1:10, stringsAsFactors = FALSE)
  tasks$seed <- 902609281L + match(tasks$design, names(designs_all())) * 100000L + tasks$n * 100L + tasks$rep
}
spec <- magmaanlab::model_spec(model_syntax); rows <- starts <- parameters <- list()
t0 <- proc.time()[["elapsed"]]
for (i in seq_len(nrow(tasks))) {
  task <- tasks[i, ]; data <- draw_data(design_sigma(designs_all()[[task$design]]), task$n, task$seed)
  sample <- sample_moments(data)
  recipes <- c(list(canonical = NULL), start_recipes(spec, sample, ablations = phase == "witness"))
  for (name in names(recipes)) {
    theta <- recipes[[name]]
    if (!is.null(theta)) {
      ev <- magmaanlab::magmaan_core$evaluate_at(spec$partable, sample, theta, estimator = "ML")
      stopifnot(is.finite(ev$fmin), ev$diagnostics$admissibility$implied_sigma_pd)
      starts[[length(starts) + 1L]] <- cbind(task[rep(1L, length(theta)), , drop = FALSE],
        start_id = name, parameter = seq_along(theta), value = theta, start_objective = ev$fmin)
    }
    for (backend in c("nlopt-lbfgs", "port")) {
      result <- run_fit(spec, data, sample, "ML", "sphere", backend, name, theta)
      id <- length(rows) + 1L
      rows[[id]] <- cbind(fit_id = id, task, transform = "native", domain = "ML", route = "sphere",
                         backend = backend, start_id = name, result$record)
      if (!is.null(result$partable)) parameters[[length(parameters) + 1L]] <-
        cbind(fit_id = id, result$partable[c("lhs", "op", "rhs", "est")])
    }
  }
  cat(sprintf("[%d/%d] %s N=%d rep=%d; %.1fs\n", i, nrow(tasks), task$design, task$n, task$rep,
              proc.time()[["elapsed"]] - t0))
}
fits <- do.call(rbind, rows)
# Development reference includes the original sphere portfolio; witness and
# fresh phases use the current portfolio only. Original records are untouched.
pool <- if (phase == "development") rbind(fits[names(fits)], old[old$domain == "ML", names(fits)]) else fits
reference <- reference_rows(pool)
cmp <- compare_rows(fits, reference)
summary <- aggregate(list(fits = cmp$fit_id), cmp[c("design", "backend", "start_id", "label", "comparison")], length)
write_out(fits, "fits"); write_out(do.call(rbind, starts), "starts")
write_out(do.call(rbind, parameters), "parameters"); write_out(reference, "references")
write_out(cmp, "comparisons"); write_out(summary, "summary")
ref <- magmaan_cache_ref()
write_metadata(file.path(out, "metadata.csv"), values = list(phase = phase, tasks = nrow(tasks),
  designs = names(designs_all()), strength = .5, starts = names(recipes),
  source_pilot = "reference-pilot", seed_rule = if (phase == "fresh") "902609281 + design_index*100000 + N*100 + rep" else "original pilot seeds",
  git_head = ref$git_head, git_dirty = ref$git_dirty,
  magmaanlab_built = utils::packageDescription("magmaanlab")$Built), packages = "magmaanlab")
cat("Wrote ", out, "\n", sep = "")
