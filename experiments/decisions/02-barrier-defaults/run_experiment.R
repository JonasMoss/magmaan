#!/usr/bin/env Rscript
# Barrier defaults: the start and optimizer of the complete-data ML barrier
# fitter (target = "determinacy"), judged by the library verdict on simulated
# populations. The criteria in criteria/barrier-ml.md were committed before
# the run.

args <- commandArgs(trailingOnly = TRUE)
usage <- "Usage: Rscript run_experiment.R [options]

Runs every arm of lane barrier-ml at penalty weights 0.25 and 1 on fresh draws
from the populations in R/families.R, under four unit transforms (native, x100,
x0.01, and separate units for invariant models), then writes the scored
summaries.

  --reps R          replications per population and N (default 100)
  --seed-base B     seed base (default 2026092701)
  --pops a,b        restrict to these population keys
  --roles r         test, control, or test,control (default)
  --workers W       parallel workers, one thread each (default 4)
  --batch B         tasks per checkpointed batch (default 40)
  --run-id ID       results/barrier-ml/<ID> (default 2026-09-27)
  --reproduces R:A  compare this run's `default` arm draw by draw with arm A of run R
                    (same seed base), writing reproduction.csv
  --arms a,b        run only these arms (the witness runs unless omitted from a list
                    that is given)
  --smoke           one replication of the first population per family, at the
                    small N, seed base + 1 (never the decision draws)
  --summarize       only rescore existing raw batches
  --help            this message

Raw per-fit rows go to results/barrier-ml/<ID>/raw/ (local); summaries and
metadata beside them are tracked evidence."
if ("--help" %in% args) { cat(usage, "\n"); quit(save = "no") }

opt <- function(name, default = NULL) if (name %in% args) args[match(name, args) + 1L] else default
script <- normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
here <- dirname(script)
source(file.path(here, "..", "..", "_support", "R", "helpers.R"))
set_single_threaded_math()
suppressPackageStartupMessages(library(magmaanlab))
for (f in c("families.R", "fits.R", "score.R")) source(file.path(here, "R", f))

lane <- "barrier-ml"
smoke <- "--smoke" %in% args
seed_base <- as.integer(opt("--seed-base", "2026092701")) + if (smoke) 1L else 0L
reps <- if (smoke) 1L else as.integer(opt("--reps", "100"))
workers <- as.integer(opt("--workers", "4"))
batch_size <- as.integer(opt("--batch", "40"))
run_id <- if (smoke) "smoke" else opt("--run-id", "2026-09-27")
out <- file.path(here, "results", lane, run_id)
raw_dir <- file.path(out, "raw")
dir.create(raw_dir, recursive = TRUE, showWarnings = FALSE)

arm_filter <- if (is.null(opt("--arms"))) NULL else strsplit(opt("--arms"), ",")[[1]]
pops <- all_populations()
roles <- strsplit(opt("--roles", "test,control"), ",")[[1]]
keep <- vapply(pops, function(p) p$role %in% roles, logical(1))
if (!is.null(opt("--pops"))) keep <- keep & names(pops) %in% strsplit(opt("--pops"), ",")[[1]]
if (smoke) keep <- keep & !duplicated(vapply(pops, `[[`, "", "family"))
pops_run <- pops[keep]

tasks <- do.call(rbind, lapply(pops_run, function(p) {
  ns <- population_ns(p)
  if (smoke) ns <- ns[1]
  g <- expand.grid(n = ns, rep = seq_len(reps))
  data.frame(pop = p$key, n = g$n, rep = g$rep,
             seed = mapply(draw_seed, seed_base, p$key, g$n, g$rep), stringsAsFactors = FALSE)
}))
batches <- split(seq_len(nrow(tasks)), ceiling(seq_len(nrow(tasks)) / batch_size))

if (!"--summarize" %in% args) {
  write_metadata(file.path(out, "metadata.csv"), values = list(
    lane = lane, seed_base = seed_base, reps = reps, smoke = smoke,
    populations = paste(names(pops_run), collapse = ","), tasks = nrow(tasks),
    arms = paste(arm_filter %||% c(names(lane_arms(lane)), "witness"), collapse = ","),
    candidates = paste(candidate_arms, collapse = ","), std_bound = 10, chart_bound = 1000,
    weights = paste(lane_estimators(lane), collapse = ","), target = "determinacy",
    transforms = "native,x100,x0.01,mixed (mixed only for unit-invariant models)",
    judge = "fit$converged (Newton check on the penalized objective); objective = penalty$penalized_fmin",
    git_head = git_scalar(c("rev-parse", "HEAD")), git_dirty = git_dirty(),
    magmaanlab_built = utils::packageDescription("magmaanlab")$Built,
    magmaanlab_path = find.package("magmaanlab")), packages = "magmaanlab")
  t_start <- proc.time()[["elapsed"]]
  done <- 0L
  for (b in seq_along(batches)) {
    file <- file.path(raw_dir, sprintf("batch_%04d.csv", b))
    if (!file.exists(file)) {
      idx <- batches[[b]]
      res <- parallel::mclapply(idx, function(i) {
        cache <- new.env()
        tryCatch(run_task(tasks[i, ], pops, lane, cache, arm_filter),
                 error = function(e) data.frame(lane = lane, pop = tasks$pop[i], n = tasks$n[i],
                   rep = tasks$rep[i], task_error = one_line(conditionMessage(e))))
      }, mc.cores = workers, mc.preschedule = TRUE)
      bad <- vapply(res, function(r) "task_error" %in% names(r), logical(1))
      if (any(bad)) {
        write_csv(do.call(rbind, res[bad]), file.path(raw_dir, sprintf("errors_%04d.csv", b)))
        res <- res[!bad]
      }
      write_csv(do.call(rbind, res), file)
    }
    done <- done + length(batches[[b]])
    el <- proc.time()[["elapsed"]] - t_start
    eta <- el / done * (nrow(tasks) - done)
    write_csv(data.frame(tasks = nrow(tasks), done = done, elapsed_s = round(el),
                         eta_s = round(eta)), file.path(out, "progress.csv"))
    cat(sprintf("[%s] batch %d/%d, tasks %d/%d, %.0f s elapsed, ETA %.0f s\n", lane, b,
                length(batches), done, nrow(tasks), el, eta))
  }
}

files <- list.files(raw_dir, pattern = "^batch_.*\\.csv$", full.names = TRUE)
raw <- do.call(rbind, lapply(files, utils::read.csv, stringsAsFactors = FALSE,
                              na.strings = "NA"))
raw$message[is.na(raw$message)] <- ""
# Family and role labels come from R/families.R; the population key is the
# stable identifier of the draws.
raw$family <- vapply(pops[raw$pop], `[[`, "", "family")
raw$role <- vapply(pops[raw$pop], `[[`, "", "role")
s <- score_rows(raw)
groups <- unit_groups(s)
pairs <- paired_table(s, lane)
inv <- invariance_table(s, groups)
written <- c(
  write_csv(rate_table(s, c("lane", "role", "family", "estimator", "transform", "arm")),
            file.path(out, "rates_by_family_transform.csv")),
  write_csv(rate_table(s, c("lane", "role", "family", "estimator", "arm")),
            file.path(out, "rates_by_family.csv")),
  write_csv(rate_table(s, c("lane", "role", "family", "pop", "model", "n", "estimator", "arm")),
            file.path(out, "rates_by_cell.csv")),
  write_csv(pairs, file.path(out, "paired.csv")),
  write_csv(loss_table(s, lane), file.path(out, "losses.csv")),
  write_csv(inv, file.path(out, "invariance.csv")),
  write_csv(inconsistency_table(s, lane, groups), file.path(out, "invariance_breaks.csv")),
  write_csv(failure_table(s), file.path(out, "failures.csv")),
  write_csv(do.call(rbind, lapply(candidate_arms, function(cd)
    cbind(candidate = cd, decision_table(s, lane, cd, pairs, inv)))), file.path(out, "decision.csv")),
  write_csv(choice_table(s, lane, candidate_arms, pairs, inv), file.path(out, "choice.csv")),
  write_csv(runaway_table(s), file.path(out, "runaways.csv")),
  write_csv(repair_table(s), file.path(out, "repairs.csv")))
if (!is.null(opt("--reproduces"))) {
  ref <- strsplit(opt("--reproduces"), ":", fixed = TRUE)[[1]]
  rf <- list.files(file.path(here, "results", lane, ref[1], "raw"), "^batch_.*\\.csv$", full.names = TRUE)
  b <- do.call(rbind, lapply(rf, utils::read.csv, stringsAsFactors = FALSE))
  k <- c("pop", "model", "n", "rep", "transform", "estimator")
  m <- merge(raw[raw$arm == "default", c(k, "certified", "fmin")], b[b$arm == ref[2], c(k, "certified", "fmin")],
             by = k, suffixes = c("_default", "_reference"))
  same_f <- (is.na(m$fmin_default) & is.na(m$fmin_reference)) | (m$fmin_default == m$fmin_reference)
  written <- c(written, write_csv(data.frame(reference_run = ref[1], reference_arm = ref[2],
    fits = sum(raw$arm == "default"), matched = nrow(m),
    same_certified = sum(m$certified_default == m$certified_reference),
    same_objective = sum(same_f %in% TRUE)), file.path(out, "reproduction.csv")))
}
errs <- list.files(raw_dir, pattern = "^errors_", full.names = TRUE)
if (length(errs)) cat("Task errors in:", errs, sep = "\n  ")
cat("Wrote:", written, sep = "\n  ")
