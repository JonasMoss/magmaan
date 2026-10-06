#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
if ("--ordinary" %in% args) {
  script <- normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
  here <- dirname(script)
  source(file.path(here, "../../../_support/R/helpers.R"))
  set_single_threaded_math()
  for (file in c("designs.R", "fit.R", "ordinary_program.R")) source(file.path(here, "R", file))
  if ("--fiml-audit" %in% args) {
    source(file.path(here,"R/fiml_audit.R"))
    run_fiml_audit(args,here)
  } else if ("--ordinal-audit" %in% args) {
    source(file.path(here,"R/ordinal_audit.R"))
    run_ordinal_audit(args,here)
  } else if ("--weighted-audit" %in% args) {
    source(file.path(here,"R/weighted_audit.R"))
    run_weighted_audit(args,here)
  } else if ("--uls-search" %in% args) {
    source(file.path(here,"R/uls_search.R"))
    run_uls_search(args,here)
  } else if ("--audit-terminal" %in% args) {
    source(file.path(here,"R/audit_terminal.R"))
    run_audit_terminal(args,here)
  } else if ("--audit-uncertainty" %in% args) {
    source(file.path(here, "R/audit_uncertainty.R"))
    run_audit_uncertainty(args, here)
  } else if (any(c("--uls-audit-guard","--uls-audit-curvature") %in% args)) {
    source(file.path(here, "R/uls_audit_guard.R"))
    run_uls_audit_guard(args, here)
  } else if ("--uls-audit-factor" %in% args) {
    source(file.path(here, "R/uls_audit_factor.R"))
    run_uls_audit_factor(args, here)
  } else if ("--uls-study" %in% args) {
    source(file.path(here, "R/uls_reliability_study.R"))
    run_uls_reliability_study(args, here)
  } else if ("--uls-unit-probe" %in% args) {
    source(file.path(here, "R/uls_unit_probe.R"))
    run_uls_unit_probe(args, here)
  } else if ("--witness-audit" %in% args) {
    source(file.path(here, "R/witness_audit.R"))
    run_witness_audit(args, here)
  } else run_ordinary_program(args, here)
  quit(save = "no")
}
usage <- paste(
  "Usage: Rscript run_experiment.R [--smoke|--pilot] [options]",
  "Exploratory sphere multistarts with separate accuracy, chart and extent labels.",
  "No default decision, global-optimum claim or nonexistence classification.",
  "--ordinary: minimal unrestricted ML/ULS/GLS programme; --ordinary --help for its grid.",
  "--smoke: 1 draw per design/N, 2 random starts. --pilot: 10 draws, 3 random starts.",
  "--reps N --random-starts N --ns 20,100 --designs ernst,weak_marker,high_r2",
  "--transforms native,x0.01,x100 --seed-base N --run-id NAME",
  "--no-lavaan: omit the ordinary lavaan comparator (otherwise required).",
  "Serial progress per draw. Outputs: results/NAME/{fits,parameters,starts,references,",
  "comparisons,summary,reference_summary,metadata,progress}.csv.", sep = "\n")
if (any(args %in% c("--help", "-h"))) { cat(usage, "\n"); quit(save = "no") }
value <- function(k, default) if (k %in% args) {
  i <- match(k, args); if (i == length(args)) stop("missing value for ", k)
  args[i + 1L]
} else default
script <- normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
here <- dirname(script)
source(file.path(here, "..", "..", "..", "_support", "R", "helpers.R"))
set_single_threaded_math()
require_pkg("magmaanlab")
for (file in c("designs.R", "fit.R")) source(file.path(here, "R", file))
profile <- if ("--pilot" %in% args) "pilot" else "smoke"
reps <- as.integer(value("--reps", if (profile == "pilot") 10 else 1))
random_starts <- as.integer(value("--random-starts", if (profile == "pilot") 3 else 2))
ns <- as.integer(strsplit(value("--ns", "20,100"), ",")[[1]])
designs <- strsplit(value("--designs", "ernst,weak_marker,high_r2"), ",")[[1]]
transforms <- strsplit(value("--transforms", "native"), ",")[[1]]
seed_base <- as.integer(value("--seed-base", if (profile == "smoke") 202609280 else 202609281))
run_id <- value("--run-id", paste0("reference-", profile))
known <- c("--smoke", "--pilot", "--no-lavaan", "--reps", "--random-starts", "--ns", "--designs", "--transforms", "--seed-base", "--run-id")
if (any(startsWith(args, "--") & !args %in% known)) stop("unknown option")
if (anyNA(c(reps, random_starts, ns, seed_base)) || reps < 1 || reps >= 100 || random_starts < 0 ||
    any(ns <= 6) || anyDuplicated(ns) || anyDuplicated(designs) || anyDuplicated(transforms) || seed_base < 1 || any(!designs %in% names(designs_all())) ||
    any(!transforms %in% c("native", "x0.01", "x100")) ||
    !grepl("^[A-Za-z0-9_-]+$", run_id)) stop("invalid options")
with_lavaan <- !"--no-lavaan" %in% args
if (with_lavaan) require_pkg("lavaan", "use --no-lavaan to omit this comparator explicitly")
out <- file.path(here, "results", run_id)
if (file.exists(file.path(out, "fits.csv"))) stop("run-id already has fits; choose a fresh --run-id")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
write_out <- function(x, name) write.csv(x, file.path(out, paste0(name, ".csv")), row.names = FALSE, na = "NA")
grid <- expand.grid(design = designs, n = ns, rep = seq_len(reps), stringsAsFactors = FALSE)
fit_rows <- parameter_rows <- start_rows <- list()
spec <- magmaanlab::model_spec(model_syntax)
t0 <- proc.time()[["elapsed"]]
for (i in seq_len(nrow(grid))) {
  task <- grid[i, ]; seed <- seed_base + match(task$design, names(designs_all())) * 100000L + task$n * 100L + task$rep
  original <- draw_data(design_sigma(designs_all()[[task$design]]), task$n, seed)
  for (tr in transforms) {
    scale <- switch(tr, native = 1, x0.01 = .01, x100 = 100)
    data <- original * scale; sample <- sample_moments(data)
    starts <- list(canonical = NULL, layered = NULL)
    if (random_starts > 0) for (j in seq_len(random_starts)) {
      sid <- paste0("random_", j)
      starts[sid] <- list(random_start(spec, sample, seed + 10000000L + j * 1000L))
      start_rows[[length(start_rows) + 1L]] <- cbind(task[rep(1L, length(starts[[sid]])), , drop = FALSE], transform = tr, seed = seed,
        start_id = sid, parameter = seq_along(starts[[sid]]), value = starts[[sid]])
    }
    arms <- rbind(
      expand.grid(domain = "ML", route = "sphere", backend = c("nlopt-lbfgs", "port"),
                  preconditioning = "none", start_id = names(starts), stringsAsFactors = FALSE),
      expand.grid(domain = "PSD", route = "sphere", backend = "nlopt-slsqp",
                  preconditioning = c("none", "diagonal"), start_id = names(starts), stringsAsFactors = FALSE),
      data.frame(domain = c("ML", "ML", "PSD"), route = "ordinary",
                 backend = c("nlopt-lbfgs", "port", "nlopt-slsqp"),
                 preconditioning = c("none", "none", "diagonal"), start_id = "default"))
    if (with_lavaan) arms <- rbind(arms, data.frame(domain = "ML", route = "lavaan",
      backend = "nlminb", preconditioning = "none", start_id = "default"))
    for (j in seq_len(nrow(arms))) {
      a <- arms[j, ]; result <- run_fit(spec, data, sample, a$domain, a$route,
        a$backend, a$start_id, starts[[a$start_id]], a$preconditioning)
      id <- length(fit_rows) + 1L
      fit_rows[[id]] <- cbind(fit_id = id, task, transform = tr, seed = seed, a, result$record)
      if (!is.null(result$partable)) parameter_rows[[length(parameter_rows) + 1L]] <-
        cbind(fit_id = id, result$partable[c("lhs", "op", "rhs", "free", "est")])
    }
  }
  elapsed <- proc.time()[["elapsed"]] - t0
  write_out(data.frame(draws = nrow(grid), completed = i, elapsed_s = elapsed,
                      eta_s = elapsed / i * (nrow(grid) - i)), "progress")
  cat(sprintf("[%d/%d] %s N=%d rep=%d; %.1fs elapsed, %.1fs ETA\n", i, nrow(grid),
              task$design, task$n, task$rep, elapsed, elapsed / i * (nrow(grid) - i)))
}
fits <- do.call(rbind, fit_rows); refs <- reference_rows(fits); cmp <- compare_rows(fits, refs)
summary <- aggregate(list(fits = cmp$fit_id), cmp[c("design", "domain", "transform", "route", "backend", "preconditioning", "start_id", "label", "comparison")], length)
ref_summary <- aggregate(list(draws = refs$rep), refs[c("design", "domain", "transform", "reference_label", "chart_status")], length)
write_out(fits, "fits"); write_out(do.call(rbind, parameter_rows), "parameters")
write_out(if (length(start_rows)) do.call(rbind, start_rows) else data.frame(start_id = character()), "starts")
write_out(refs, "references"); write_out(cmp, "comparisons")
write_out(summary, "summary"); write_out(ref_summary, "reference_summary")
ref <- magmaan_cache_ref()
write_metadata(file.path(out, "metadata.csv"), values = list(profile = profile,
  reps = reps, random_starts = random_starts, ns = ns, designs = designs,
  transforms = transforms, seed_base = seed_base, fit_count = nrow(fits),
  polish = FALSE, newton_budget = .01, extent_screen = 10,
  sphere_audit = "driven first-order plus strongest-indicator marker Newton cross-check; no sphere-native certificate",
  git_head = ref$git_head, git_dirty = ref$git_dirty,
  magmaanlab_path = find.package("magmaanlab"), magmaanlab_built = utils::packageDescription("magmaanlab")$Built),
  packages = c("magmaanlab", if (with_lavaan) "lavaan"))
cat("Wrote results to ", out, "\n", sep = "")
