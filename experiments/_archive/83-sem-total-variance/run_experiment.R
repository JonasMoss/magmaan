#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
if ("--help" %in% args) {
  cat(paste0("Usage: Rscript run_experiment.R [options]\n",
    "Default: six structures, population/observed moments, three charts.\n",
    "  --smoke             Add one matched Gaussian draw per model\n",
    "  --reps N            Gaussian draws per model (default 0)\n",
    "  --cases LIST        hs_3factor_cfa,mediation,correlated_predictors,\n",
    "                      higher_order,correlated_disturbances,bollen_democracy_sem\n",
    "  --stress            Stress the four controlled models\n",
    "  --maxit N           BFGS iteration limit (default 200)\n",
    "  --budget-sec N      Stop between fits after N seconds (default 60)\n",
    "  --seed-base N       Default 20260919\n",
    "  --results-dir PATH  Default results beside this script\n",
    "  --dry-run-plan      Print exact fit count without fitting\n",
    "This R prototype tests chart geometry/work, not native magmaan speed.\n"))
  quit(status = 0)
}
script <- normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
here <- dirname(script); repo <- normalizePath(file.path(here, "../../.."))
o <- list(reps = 0L, cases = c("hs_3factor_cfa", "mediation", "correlated_predictors",
  "higher_order", "correlated_disturbances", "bollen_democracy_sem"), stress = FALSE,
  maxit = 200L, budget_sec = 60, seed_base = 20260919L,
  results_dir = file.path(here, "results"), plan = FALSE)
i <- 1L
while (i <= length(args)) {
  a <- args[i]
  if (a == "--smoke") o$reps <- 1L
  else if (a == "--stress") o$stress <- TRUE
  else if (a == "--dry-run-plan") o$plan <- TRUE
  else {
    if (i == length(args)) stop("Missing value for ", a)
    i <- i + 1L; v <- args[i]
    key <- sub("^--", "", gsub("-", "_", sub("^--", "", a)))
    if (!key %in% c("reps", "cases", "maxit", "budget_sec", "seed_base", "results_dir"))
      stop("Unknown option: ", a)
    o[[key]] <- if (key == "cases") strsplit(v, ",", fixed = TRUE)[[1]] else
      if (key == "results_dir") v else as.numeric(v)
  }
  i <- i + 1L
}
known <- c("hs_3factor_cfa", "mediation", "correlated_predictors", "higher_order",
           "correlated_disturbances", "bollen_democracy_sem")
stopifnot(length(o$cases) > 0, !anyDuplicated(o$cases), all(o$cases %in% known),
          is.finite(o$reps), o$reps >= 0, o$reps == as.integer(o$reps),
          is.finite(o$maxit), o$maxit > 0, o$maxit == as.integer(o$maxit), is.finite(o$budget_sec), o$budget_sec > 0,
          is.finite(o$seed_base), o$seed_base >= 0)
charts <- c("marker", "disturbance", "total")
planned <- length(o$cases) * (o$reps + 1L) * length(charts)
cat(sprintf("%d structures x %d moment sets x 3 charts = %d fits; %.0fs soft budget\n",
            length(o$cases), o$reps + 1, planned, o$budget_sec))
if (o$plan) quit(status = 0)
source(file.path(repo, "experiments/_support/R/helpers.R"))
set_single_threaded_math()
source(file.path(here, "R/charts.R")); source(file.path(here, "R/models.R"))
if (!requireNamespace("lavaan", quietly = TRUE)) stop("Install lavaan for the empirical sentinels")
dir.create(o$results_dir, recursive = TRUE, showWarnings = FALSE)
# A failed or truncated rerun must not leave an earlier run's completed tables.
unlink(file.path(o$results_dir, c("fits.csv", "validation.csv", "timing.csv", "metadata.csv")))
rows <- validations <- timings <- list(); t0 <- Sys.time(); exhausted <- FALSE
elapsed <- function() as.numeric(difftime(Sys.time(), t0, units = "secs"))
record_time <- function(id, stage, seconds) {
  timings[[length(timings) + 1L]] <<- data.frame(case = id, stage = stage, seconds = seconds)
}
for (id in o$cases) {
  if (elapsed() > o$budget_sec) { exhausted <- TRUE; break }
  t <- elapsed()
  case <- if (id %in% c("hs_3factor_cfa", "bollen_democracy_sem")) empirical_case(id, repo) else
    synthetic_case(id, o$stress)
  if (min_eigen(case$m$psi) <= 0 || min_eigen(case$m$theta) <= 0)
    stop("Reference outside common interior: ", id)
  if (is.finite(case$oracle)) {
    sigma <- implied(case$m)$Sigma
    f_reference <- as.numeric(determinant(sigma, logarithm = TRUE)$modulus) +
      sum(solve(sigma) * case$S) -
      as.numeric(determinant(case$S, logarithm = TRUE)$modulus) - nrow(sigma)
    if (abs(f_reference - case$oracle) > 1e-8) stop("Lavaan projection mismatch: ", id)
  }
  record_time(id, "setup_reference", elapsed() - t)
  seed <- o$seed_base + match(id, known) * 10000L
  start <- common_start(case, seed)
  maps <- lapply(charts, function(ch) chart_map(case$m, case$mask, ch)); names(maps) <- charts
  t <- elapsed()
  start_norms <- numeric()
  for (ch in charts) {
    v <- validate_chart(maps[[ch]], start)
    g <- geometry(maps[[ch]], maps[[ch]]$pack(case$m))
    if (g$rank != g$npar) stop("Rank-deficient selected model/chart: ", id, "/", ch)
    gs <- geometry(maps[[ch]], maps[[ch]]$pack(start))
    us <- maps[[ch]]$unpack(maps[[ch]]$pack(start))
    W <- solve(us$Sigma); G <- W - W %*% case$S %*% W
    grad <- vapply(maps[[ch]]$derivatives(us), function(d) sum(G * d), 0.0)
    start_norms[ch] <- sqrt(sum(grad * solve(gs$information, grad)))
    if (ch == "total" && max(abs(diag(us$C) - 1)) > 1e-10)
      stop("Total-variance normalization failed")
    if (ch == "disturbance" && max(abs(diag(us$m$psi) - 1)) > 1e-10)
      stop("Disturbance normalization failed")
    validations[[length(validations) + 1L]] <- data.frame(case = id, chart = ch,
      covariance_error = v[1], derivative_error = v[2], rank = g$rank,
      npar = g$npar, condition = g$condition, start_gradient_norm = start_norms[ch])
  }
  if (diff(range(start_norms)) > 1e-8 * max(1, max(start_norms)))
    stop("Information-metric gradient is not invariant across charts")
  record_time(id, "validation_geometry", elapsed() - t)
  for (r in 0:o$reps) {
    S <- case$S
    if (r > 0) {
      set.seed(seed + r)
      # Wishart covariance directly: no repeated raw-data/spec construction.
      S <- rWishart(1L, case$n - 1L, implied(case$m)$Sigma)[, , 1] / case$n
    }
    # Rotate execution order to reduce warmup/order bias.
    order <- charts[((seq_along(charts) + r + match(id, known) - 2L) %% 3L) + 1L]
    for (ch in order) {
      if (elapsed() > o$budget_sec) { exhausted <- TRUE; break }
      map <- maps[[ch]]
      t <- elapsed()
      fit <- tryCatch(fit_chart(map, map$pack(start), S, o$maxit), error = function(e) e)
      record_time(id, "fit", elapsed() - t)
      error <- inherits(fit, "error")
      audit_start <- elapsed()
      dual <- if (error) NA_real_ else tryCatch({
        g <- geometry(map, fit$fit$par)
        if (g$rank < g$npar) Inf else
          sqrt(max(0, sum(fit$gradient * solve(g$information, fit$gradient))))
      }, error = function(e) Inf)
      record_time(id, "terminal_audit", elapsed() - audit_start)
      row <- data.frame(case = id, origin = case$origin, stress = o$stress && case$origin == "controlled",
        replicate = r, moments = if (r > 0) "gaussian" else if (case$origin == "controlled") "population" else "observed",
        chart = ch, n = case$n, p = nrow(S), status = if (error) "error" else "returned",
        error = if (error) conditionMessage(fit) else "", optimizer_code = if (error) NA else fit$fit$convergence,
        objective = if (error) NA else fit$fit$value, fit_seconds = if (error) NA else fit$seconds,
        evaluations = if (error) NA else fit$evaluations, invalid_evaluations = if (error) NA else fit$invalid,
        min_psi = if (error) NA else min_eigen(fit$terminal$m$psi),
        min_theta = if (error) NA else min_eigen(fit$terminal$m$theta),
        min_sigma = if (error) NA else min_eigen(fit$terminal$Sigma),
        fisher_gradient_norm = dual, stationary = is.finite(dual) && dual < 1e-4,
        gradient_inf = if (error) NA else max(abs(fit$gradient)),
        reference_objective = if (r == 0L) if (case$origin == "controlled") 0 else case$oracle else NA_real_)
      row$admissible <- !error && row$min_psi > 0 && row$min_theta > 0 && row$min_sigma > 0
      rows[[length(rows) + 1L]] <- row
      write.csv(do.call(rbind, rows), file.path(o$results_dir, "fits.csv"), row.names = FALSE)
      cat(sprintf("%d/%d %-25s %-11s draw=%d %.2fs elapsed\n", length(rows), planned, id, ch, r, elapsed()))
      flush.console()
    }
    if (exhausted) break
  }
  if (exhausted) break
}
if (length(rows)) {
  d <- do.call(rbind, rows)
  key <- interaction(d$case, d$replicate, drop = TRUE)
  d$gap_from_best <- ave(d$objective, key, FUN = function(x)
    if (any(is.finite(x))) x - min(x[is.finite(x)]) else rep(NA_real_, length(x)))
  d$paired_complete <- ave(d$chart, key, FUN = function(x) as.character(length(unique(x)) == 3)) == "TRUE"
  write.csv(d, file.path(o$results_dir, "fits.csv"), row.names = FALSE)
}
if (length(validations)) write.csv(do.call(rbind, validations), file.path(o$results_dir, "validation.csv"), row.names = FALSE)
if (length(timings)) write.csv(do.call(rbind, timings), file.path(o$results_dir, "timing.csv"), row.names = FALSE)
write_metadata(file.path(o$results_dir, "metadata.csv"),
  values = c(o, list(planned_fits = planned, completed_fits = length(rows),
    elapsed_seconds = elapsed(), complete = !exhausted, engine = "R analytic-gradient BFGS prototype",
    git_head = git_scalar(c("rev-parse", "HEAD"), root = repo),
    source_md5 = paste(tools::md5sum(c(script, file.path(here, "R/charts.R"),
                                     file.path(here, "R/models.R"))), collapse = ","),
    corpus_available = corpus_available())), packages = c("lavaan"))
cat("Wrote metadata and available fit/validation/timing tables to ", o$results_dir, "\n", sep = "")
if (exhausted) { cat("Budget exhausted; incomplete pairs are flagged.\n"); quit(status = 2L) }
