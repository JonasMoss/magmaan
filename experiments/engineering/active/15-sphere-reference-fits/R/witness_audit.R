# Exact saved-point checks and separately labeled warm fits. Imported fitted
# values are diagnostic starts only; they never enter a sample-only policy.
run_witness_audit <- function(args, here) {
  if (any(args %in% c("--help", "-h"))) {
    cat("Usage: Rscript run_experiment.R --ordinary --witness-audit [--run-id NAME] [--cold-run NAME]\n",
        "Checks nine saved sphere endpoints and seven refined finite ML witnesses\n",
        "without optimization, using the current audit in a strong-marker chart.\n",
        "Then runs explicitly labeled warm NTML/ULS/GLS fits from the seven refined\n",
        "points, under tight L-BFGS/PORT in original marker and sphere charts (84 fits).\n",
        "Warm LS targets differ from ML; matching the ML objective is not required.\n",
        "Outputs exact_points.csv, warm_fits.csv, warm_summary.csv and metadata.csv;\n",
        "RDS endpoints stay local. No library default or audit threshold changes.\n", sep = "")
    return(invisible(NULL))
  }
  known <- c("--ordinary", "--witness-audit", "--run-id", "--cold-run")
  if (any(startsWith(args, "--") & !args %in% known)) stop("unknown witness-audit option")
  i <- match("--run-id", args)
  run <- if (is.na(i)) "ordinary-witness-audit" else args[i + 1L]
  if (is.na(run) || !grepl("^[A-Za-z0-9_-]+$", run)) stop("invalid witness-audit run-id")
  j <- match("--cold-run", args)
  cold_run <- if (is.na(j)) "ordinary-layered-witnesses" else args[j + 1L]
  if (is.na(cold_run) || !grepl("^[A-Za-z0-9_-]+$", cold_run)) stop("invalid cold run-id")
  cold_file <- file.path(here, "results", cold_run, "paired.csv")
  if (!file.exists(cold_file)) stop("missing cold results; run --ordinary --witnesses --run-id ", cold_run)
  out <- file.path(here, "results", run)
  if (dir.exists(out)) stop("choose a fresh witness-audit run-id")
  require_pkg("magmaanlab")
  dir.create(out, recursive = TRUE)
  write_out <- function(x, name) write_csv(x, file.path(out, paste0(name, ".csv")))
  files <- file.path(here, "results", c(
    "sphere-translations-inspected/source_endpoints.csv", "sphere-translations-inspected/sphere_parameters.csv",
    "open-case-conclusions/finite_witnesses.csv", "open-case-conclusions/parameters.csv"))
  saved <- read.csv(files[1]); saved_pt <- read.csv(files[2])
  finite <- read.csv(files[3]); finite_params <- read.csv(files[4])
  base <- magmaanlab::model_spec(model_syntax)
  key <- function(pt) paste(pt$lhs, pt$op, pt$rhs)
  point_rows <- warm_rows <- endpoints <- list()
  t0 <- proc.time()[["elapsed"]]
  evaluate_point <- function(pt, sample, task, source, expected_objective) {
    target <- strong_marker_model(pt, sample)
    tr <- magmaanlab::frontier_reidentify(pt, target, pole_tol = 0)
    ev <- magmaanlab::magmaan_core$evaluate_at(target$partable, sample, tr$theta, estimator = "ML")
    na <- ev$diagnostics$newton_accuracy
    gap <- abs(ev$fmin - expected_objective)
    if (!is.finite(gap) || gap > 1e-8 * (1 + abs(expected_objective))) stop("saved-point objective mismatch")
    available <- function(tol) !inherits(tryCatch(
      magmaanlab::frontier_reidentify(pt, base, pole_tol = tol), error = identity), "error")
    cbind(task, point_source = source, expected_objective = expected_objective,
      objective = ev$fmin, objective_gap = gap, audit_status = ev$verdict$status,
      newton_status = na$status, newton_distance = na$distance, condition = na$condition,
      original_chart_1e6 = available(1e-6), original_chart_1e4 = available(1e-4),
      primitive_admissible = ev$diagnostics$admissibility$admissible,
      assessment_chart = "strong_marker; same saved point, no optimization")
  }
  for (i in seq_len(nrow(saved))) {
    a <- saved[i, ]; sample <- sample_moments(draw_data(design_sigma(designs_all()[[a$design]]), a$n, a$seed))
    pt <- base$partable; point <- saved_pt[saved_pt$case_id == i, ]
    idx <- match(key(pt), key(point)); stopifnot(!anyNA(idx))
    pt$est <- point$est[idx]; pt$ustart <- pt$est; pt$free <- seq_len(nrow(pt))
    task <- data.frame(case_id = i, batch = a$batch, design = a$design, n = a$n, rep = a$rep, seed = a$seed)
    point_rows[[length(point_rows) + 1L]] <- evaluate_point(pt, sample, task, "saved_sphere", a$objective)
  }
  for (i in seq_len(nrow(finite))) {
    a <- finite[i, ]; v <- finite_params$value[finite_params$case_id == a$case_id]
    stopifnot(length(v) == 13)
    pt <- base$partable; values <- pt$ustart
    values[pt$op == "=~"] <- c(1, v[1:2], 1, v[3:4])
    values[pt$op == "~"] <- v[6] / v[5]
    values[pt$lhs == "X" & pt$op == "~~"] <- v[5]
    values[pt$lhs == "Y" & pt$op == "~~"] <- v[7] - v[6]^2 / v[5]
    for (j in 1:6) values[pt$lhs == ov_names[j] & pt$op == "~~"] <- v[7 + j]
    theta <- numeric(max(pt$free)); theta[pt$free[pt$free > 0]] <- values[pt$free > 0]
    pt$est <- values; pt$ustart <- values
    seed <- (if (a$batch == "development") 202609281L else 902609281L) +
      match(a$design, names(designs_all())) * 100000L + a$n * 100L + a$rep
    sample <- sample_moments(draw_data(design_sigma(designs_all()[[a$design]]), a$n, seed))
    task <- data.frame(case_id = a$case_id, batch = a$batch, design = a$design, n = a$n, rep = a$rep, seed = seed)
    point_rows[[length(point_rows) + 1L]] <- evaluate_point(pt, sample, task, "refined_finite", a$objective)
    # Also audit the original marker chart: a change of chart can improve the
    # conditioning guard while preserving the model-implied covariance.
    ev_original <- magmaanlab::magmaan_core$evaluate_at(base$partable, sample, theta, estimator = "ML")
    point_rows[[length(point_rows)]]$original_marker_audit <- ev_original$verdict$status
    point_rows[[length(point_rows)]]$original_marker_newton <- ev_original$diagnostics$newton_accuracy$status
    for (estimator in c("ML", "ULS", "GLS")) for (chart in c("marker", "sphere"))
      for (backend in c("nlopt-lbfgs", "port")) {
        ctl <- if (backend == "port") list(port = list(rel_f_tol = 1e-12, x_tol = 1e-10)) else
          list(nlopt = list(ftol_rel = 1e-12, xtol_rel = 1e-10))
        target <- base; free <- target$partable$free
        target$partable$ustart[free > 0] <- theta[free[free > 0]]
        fit_t0 <- proc.time()[["elapsed"]]
        fit <- tryCatch(suppressWarnings(if (chart == "sphere")
          magmaanlab::frontier_fit_sphere(target, sample, estimator = estimator,
            optimizer = backend, control = ctl, polish = FALSE, start = "user") else
          magmaanlab::fit_model(base, sample, estimator = estimator, optimizer = backend,
            control = modifyList(ctl, list(start = theta)))), error = identity)
        z <- ordinary_endpoint(fit, base, sample, estimator, chart)
        id <- length(warm_rows) + 1L
        warm_rows[[id]] <- cbind(task, estimator = estimator, chart = chart, backend = backend,
          start_source = "refined_ML_endpoint; diagnostic only", seconds = proc.time()[["elapsed"]] - fit_t0,
          known_ml_objective = a$objective, z$record)
        endpoints[[id]] <- fit
      }
    cat(sprintf("Refined witness %d/%d; %.1fs elapsed\n", i, nrow(finite), proc.time()[["elapsed"]] - t0))
  }
  # Fill optional columns before binding saved and refined records.
  for (i in seq_along(point_rows)) for (name in c("original_marker_audit", "original_marker_newton"))
    if (is.null(point_rows[[i]][[name]])) point_rows[[i]][[name]] <- "not_checked"
  points <- do.call(rbind, point_rows); warm <- do.call(rbind, warm_rows)
  warm$ml_witness_match <- warm$estimator == "ML" & warm$accepted &
    is.finite(warm$objective) & abs(warm$objective - warm$known_ml_objective) <= 1e-6 * (1 + abs(warm$known_ml_objective))
  write_out(points, "exact_points"); write_out(warm, "warm_fits")
  write_out(aggregate(warm[c("returned", "accepted", "ml_witness_match")],
    warm[c("estimator", "chart", "backend")], sum), "warm_summary")
  # Qualified historical minima are a different judge from the best returned
  # by this cold portfolio. Only same-estimator ML targets are imported.
  cold <- read.csv(cold_file); cold_ml <- cold[cold$estimator == "ML", ]
  targets <- points[points$point_source == "refined_finite" & points$audit_status == "passed", ]
  case_key <- function(x) paste(x$batch, x$design, x$n, x$rep, x$seed)
  idx <- match(case_key(cold_ml), case_key(targets))
  cold_ml$known_local_objective <- targets$objective[idx]
  cold_ml$target_available <- is.finite(cold_ml$known_local_objective)
  cold_ml$known_local_hit <- cold_ml$accepted & cold_ml$target_available &
    is.finite(cold_ml$objective) & abs(cold_ml$objective - cold_ml$known_local_objective) <=
      1e-6 * (1 + abs(cold_ml$known_local_objective))
  write_out(aggregate(cold_ml[c("accepted", "target_available", "known_local_hit")],
    cold_ml[c("chart", "start_id", "profile")], sum), "known_target_summary")
  write_out(cold_ml[c("batch", "design", "n", "rep", "seed", "chart", "start_id", "profile",
    "accepted", "audit_status", "newton_status", "objective", "known_local_objective",
    "target_available", "known_local_hit")], "known_target_checks")
  # LS has its own objective. Preserve how each warm endpoint compares with
  # locally accepted cold/warm candidates of the same estimator and dataset.
  warm_ls <- warm[warm$estimator != "ML", ]; cold_ls <- cold[cold$estimator != "ML", ]
  warm_ls$best_observed_objective <- vapply(seq_len(nrow(warm_ls)), function(i) {
    a <- warm_ls[i, ]; same <- function(x) case_key(x) == case_key(a) & x$estimator == a$estimator & x$accepted
    values <- c(cold_ls$objective[same(cold_ls)], warm_ls$objective[same(warm_ls)])
    values <- values[is.finite(values)]
    if (length(values)) min(values) else NA_real_
  }, 0.0)
  warm_ls$best_observed_hit <- warm_ls$accepted & is.finite(warm_ls$best_observed_objective) &
    is.finite(warm_ls$objective) & abs(warm_ls$objective - warm_ls$best_observed_objective) <=
      1e-6 * (1 + abs(warm_ls$best_observed_objective))
  write_out(warm_ls[c("case_id", "batch", "design", "n", "rep", "seed", "estimator", "chart", "backend",
    "accepted", "audit_status", "newton_status", "newton_distance", "objective",
    "best_observed_objective", "best_observed_hit")], "ls_warm_checks")
  saveRDS(endpoints, file.path(out, "endpoints.rds"))
  ref <- magmaan_cache_ref()
  code <- file.path(here, c("run_experiment.R", "R/witness_audit.R", "R/ordinary_program.R", "R/designs.R", "R/fit.R"))
  package_files <- list.files(find.package("magmaanlab"), recursive = TRUE, full.names = TRUE)
  write_metadata(file.path(out, "metadata.csv"), values = list(
    input_md5 = paste(names(tools::md5sum(files)), tools::md5sum(files), collapse = ";"),
    cold_run = cold_run, cold_md5 = tools::md5sum(cold_file),
    source_md5 = paste(names(tools::md5sum(code)), tools::md5sum(code), collapse = ";"),
    package_md5 = paste(tools::md5sum(package_files[grepl("\\.(so|rdb|rdx)$", package_files)]), collapse = ";"),
    git_head = ref$git_head, git_dirty = ref$git_dirty, magmaanlab_path = find.package("magmaanlab"),
    magmaanlab_built = utils::packageDescription("magmaanlab")$Built,
    saved_points = nrow(saved), refined_points = nrow(finite), warm_fits = nrow(warm),
    exact_method = "current evaluate_at audit in strongest-marker chart; no optimization",
    warm_method = "known refined ML endpoint as start; tight L-BFGS/PORT; no sphere polish",
    audit_budget = .01, max_condition = 1e12, chart_tolerances = "1e-6 library; 1e-4 study only",
    interpretation = "development witnesses; warm restarts are not sample-only starts or global-optimum evidence",
    elapsed_s = proc.time()[["elapsed"]] - t0), packages = "magmaanlab")
  cat("Wrote witness checks to ", out, "\n", sep = "")
}
