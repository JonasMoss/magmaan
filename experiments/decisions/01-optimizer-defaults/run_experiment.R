#!/usr/bin/env Rscript
# Optimizer defaults: one lane per estimation route, each judged by the
# library verdict on simulated populations the defaults were not developed
# on. The criteria in criteria/<lane>.md were committed before the run.

args <- commandArgs(trailingOnly = TRUE)
usage <- "Usage: Rscript run_experiment.R --lane ml-gls|psd-ml|ls-coordinates [options]

Runs every configuration of the lane on fresh draws from the populations in
R/families.R, under four unit transforms (native, x100, x0.01, and separate
units for invariant models), then writes the scored summaries.

  --lane L          ml-gls (ML and GLS: start x optimizer) or psd-ml
                    (direct PSD fits, preconditioning, two-stage route); ls-coordinates
                    (ULS, DWLS, WLS and MAR ML2S with those three weights)
  --reps R          replications per population and N (default 100 ml-gls, 50 psd-ml)
  --seed-base B     seed base (default 20260926, the freeze date)
  --pops a,b        restrict to these population keys
  --roles r         test, control, or test,control (default)
  --workers W       parallel workers, one thread each (default 4)
  --batch B         tasks per checkpointed batch (default 40)
  --run-id ID       results/<lane>/<ID> (default 2026-09-26; smoke default smoke)
  --candidate A[,B] the arm(s) the rules test against `default` (default layered_port, or objective_downward,objective for ls-coordinates);
                    with two, choice.csv applies the pre-registered choice
  --runaway-bound X certified fits with a standardized extent above X are runaways,
                    scored as failures (default Inf: no runaway rule)
  --reproduces R:A  compare this run's `default` arm draw by draw with arm A of run R
                    (same seed base), writing reproduction.csv
  --reproduces-ml R:A  optional ML reference when GLS and ML defaults differ
  --arms a,b        run only these arms (the witness runs unless omitted from a list
                    that is given)
  --smoke           one replication of the first population per family, at the
                    small N, seed base + 1 (never the decision draws)
  --summarize       only rescore existing raw batches
  --help            this message

Raw per-fit rows go to results/<lane>/<ID>/raw/ (local); summaries and
metadata beside them are tracked evidence."
if ("--help" %in% args) { cat(usage, "\n"); quit(save = "no") }

opt <- function(name, default = NULL) if (name %in% args) args[match(name, args) + 1L] else default
script <- normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
here <- dirname(script)
source(file.path(here, "..", "..", "_support", "R", "helpers.R"))
source(file.path(here, "..", "..", "_support", "R", "missingness.R"))
set_single_threaded_math()
suppressPackageStartupMessages(library(magmaanlab))
for (f in c("families.R", "fits.R", "score.R")) source(file.path(here, "R", f))

lane <- opt("--lane")
if (is.null(lane) || !lane %in% c("ml-gls", "psd-ml", "ls-coordinates")) stop("--lane ml-gls|psd-ml|ls-coordinates is required")
smoke <- "--smoke" %in% args
seed_base <- as.integer(opt("--seed-base", "20260926")) + if (smoke) 1L else 0L
reps <- if (smoke) 1L else as.integer(opt("--reps", if (lane != "psd-ml") "100" else "50"))
workers <- as.integer(opt("--workers", "4"))
batch_size <- as.integer(opt("--batch", "40"))
run_id <- opt("--run-id", if (smoke) "smoke" else "2026-09-26")
out <- file.path(here, "results", lane, run_id)
raw_dir <- file.path(out, "raw")
dir.create(raw_dir, recursive = TRUE, showWarnings = FALSE)

candidates <- strsplit(opt("--candidate", if (lane == "ls-coordinates") "objective_downward,objective" else "layered_port"), ",")[[1]]
candidate <- candidates[1]
runaway_bound <- as.numeric(opt("--runaway-bound", "Inf"))
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

# A resume must use the same design, seeds, source and installed package.
fingerprint <- c(lane, seed_base, reps, names(pops_run), arm_filter, candidates,
  runaway_bound, batch_size, as.character(tools::md5sum(c(script,
    file.path(here, "R", c("fits.R", "score.R", "families.R")),
    file.path(here, "criteria", paste0(lane, ".md")),
    file.path(find.package("magmaanlab"), "libs", "magmaanlab.so")))))
manifest <- file.path(raw_dir, "manifest.rds")
if (lane == "ls-coordinates" && file.exists(manifest) && !identical(readRDS(manifest), fingerprint))
  stop("Resume fingerprint differs; use a fresh --run-id")
if (lane == "ls-coordinates" && !file.exists(manifest) && length(list.files(raw_dir, "^batch_")))
  stop("Existing batches lack a resume fingerprint; use a fresh --run-id")
if (!"--summarize" %in% args) {
  if (lane == "ls-coordinates") saveRDS(fingerprint, manifest)
  write_metadata(file.path(out, "metadata.csv"), values = list(
    lane = lane, seed_base = seed_base, reps = reps, smoke = smoke,
    populations = paste(names(pops_run), collapse = ","), tasks = nrow(tasks),
    arms = paste(arm_filter %||% c(names(lane_arms(lane)), "witness"), collapse = ","),
    candidate = paste(candidates, collapse = ","), runaway_bound = runaway_bound,
    estimators = paste(lane_estimators(lane), collapse = ","),
    transforms = "native,x100,x0.01,mixed (mixed only for unit-invariant models)",
    judge = "fit$converged (Newton check); PSD fits also admissible",
    git_head = git_scalar(c("rev-parse", "HEAD")), git_dirty = git_dirty(),
    criteria_md5 = unname(tools::md5sum(file.path(here, "criteria", paste0(lane, ".md")))),
    runner_md5 = paste(tools::md5sum(c(script, file.path(here, "R", c("fits.R", "score.R", "families.R")))), collapse = ","),
    missingness = if (lane == "ls-coordinates") "single_cause_mar; cause first variable; targets last floor(p/2); rate .15" else "none",
    reference = if (lane == "ls-coordinates") "objective coordinates; max_iter 20000; arm witness" else "population-start witness",
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
s <- score_rows(raw, runaway_bound)
candidates <- intersect(candidates, unique(s$arm))
groups <- unit_groups(s)
written <- c(
  write_csv(rate_table(s, c("lane", "role", "family", "estimator", "transform", "arm")),
            file.path(out, "rates_by_family_transform.csv")),
  write_csv(rate_table(s, c("lane", "role", "family", "estimator", "arm")),
            file.path(out, "rates_by_family.csv")),
  write_csv(rate_table(s, c("lane", "role", "family", "pop", "model", "n", "estimator", "arm")),
            file.path(out, "rates_by_cell.csv")),
  write_csv(paired_table(s, lane, candidate = candidate), file.path(out, "paired.csv")),
  write_csv(loss_table(s, lane, candidates), file.path(out, "losses.csv")),
  write_csv(invariance_table(s, groups), file.path(out, "invariance.csv")),
  write_csv(inconsistency_table(s, lane, groups, candidates), file.path(out, "invariance_breaks.csv")),
  write_csv(failure_table(s), file.path(out, "failures.csv")),
  write_csv(escape_table(s), file.path(out, "escapes.csv")),
  if (length(candidates)) write_csv(do.call(rbind, lapply(candidates, function(cd)
    cbind(candidate = cd, decision_table(s, lane, cd)))), file.path(out, "decision.csv")),
  write_csv(runaway_table(s), file.path(out, "runaways.csv")))
if (!is.null(opt("--reproduces"))) {
  ref <- strsplit(opt("--reproduces"), ":", fixed = TRUE)[[1]]
  rf <- list.files(file.path(here, "results", lane, ref[1], "raw"), "^batch_.*\\.csv$", full.names = TRUE)
  b <- do.call(rbind, lapply(rf, utils::read.csv, stringsAsFactors = FALSE))
  k <- c("pop", "model", "n", "rep", "transform", "estimator")
  reproduction <- lapply(unique(raw$estimator), function(est) {
    est_ref <- if (est == "ML" && !is.null(opt("--reproduces-ml")))
      strsplit(opt("--reproduces-ml"), ":", fixed = TRUE)[[1]] else ref
    reference <- b
    if (est_ref[1] != ref[1]) stop("ML reference must use the same run")
    current <- raw[raw$arm == "default" & raw$estimator == est, c(k, "certified", "fmin")]
    m <- merge(current, reference[reference$arm == est_ref[2] & reference$estimator == est,
                                c(k, "certified", "fmin")],
               by = k, suffixes = c("_default", "_reference"))
    same_f <- (is.na(m$fmin_default) & is.na(m$fmin_reference)) |
      (m$fmin_default == m$fmin_reference)
    data.frame(estimator = est, reference_run = est_ref[1], reference_arm = est_ref[2],
      fits = nrow(current), matched = nrow(m),
      same_certified = sum(m$certified_default == m$certified_reference),
      same_objective = sum(same_f %in% TRUE))
  })
  written <- c(written, write_csv(do.call(rbind, reproduction), file.path(out, "reproduction.csv")))
}
if (length(candidates) == 2L) written <- c(written,
  write_csv(choice_table(s, lane, candidates), file.path(out, "choice.csv")))
if (lane == "psd-ml") written <- c(written,
  write_csv(tolerance_table(s), file.path(out, "tolerance.csv")),
  write_csv(stage_table(s), file.path(out, "twostage_stages.csv")))
errs <- list.files(raw_dir, pattern = "^errors_", full.names = TRUE)
if (length(errs)) cat("Task errors in:", errs, sep = "\n  ")
cat("Wrote:", written, sep = "\n  ")
