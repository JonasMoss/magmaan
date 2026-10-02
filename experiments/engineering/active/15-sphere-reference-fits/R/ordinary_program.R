# A bounded engineering lane: identical moments and objective within estimator,
# sample-only starts, and endpoint audits independent of backend stop labels.
run_ordinary_program <- function(args, here) {
  usage <- paste(
    "Usage: Rscript run_experiment.R --ordinary [--smoke|--pilot] [options]",
    "Complete-data unrestricted NTML (ML), ULS and GLS; marker and sphere charts.",
    "Axes: route-native vs shared FABIN3 starts; default/tight L-BFGS and PORT.",
    "--smoke: regular N=100, one draw (48 fits).",
    "--pilot: ernst,weak_marker,high_r2 at N=20,100, two draws (576 fits).",
    "--reps N --ns 20,100 --designs ernst,weak_marker,high_r2",
    "--estimators ML,ULS,GLS --seed-base N --run-id NAME",
    "--transforms native,mixed (mixed units .01,100,2,.3,10,.1).",
    "No PSD, barriers, bounds, automatic retries, or default changes.",
    "All arms share ML-scaled sample covariance (divisor N); GLS weight is frozen",
    "from that sample. ULS changes target under heterogeneous unit changes.",
    "Writes fit/start/pair, summary/audit/change/reference/metadata CSVs and",
    "endpoint/sample RDSs under results/NAME; fresh run IDs required.", sep = "\n")
  if (any(args %in% c("--help", "-h"))) { cat(usage, "\n"); return(invisible(NULL)) }
  known <- c("--ordinary", "--smoke", "--pilot", "--reps", "--ns", "--designs",
             "--estimators", "--seed-base", "--run-id", "--transforms")
  if (any(startsWith(args, "--") & !args %in% known)) stop("unknown ordinary option")
  value <- function(k, default) {
    i <- match(k, args)
    if (is.na(i)) return(default)
    if (i == length(args) || startsWith(args[i + 1L], "--")) stop("missing value for ", k)
    args[i + 1L]
  }
  smoke <- !"--pilot" %in% args
  reps <- as.integer(value("--reps", if (smoke) "1" else "2"))
  ns <- as.integer(parse_csv_arg(value("--ns", if (smoke) "100" else "20,100")))
  designs <- parse_csv_arg(value("--designs", if (smoke) "ernst" else "ernst,weak_marker,high_r2"))
  estimators <- parse_csv_arg(value("--estimators", "ML,ULS,GLS"))
  transforms <- parse_csv_arg(value("--transforms", "native"))
  seed_base <- as.integer(value("--seed-base", if (smoke) "620260001" else "620260002"))
  run_id <- value("--run-id", if (smoke) "ordinary-smoke" else "ordinary-pilot")
  if (anyNA(c(reps, ns, seed_base)) || reps < 1 || reps >= 100 || any(ns <= 6) ||
      !length(ns) || !length(designs) || !length(estimators) || !length(transforms) ||
      seed_base < 1 || seed_base > 2e9 || any(ns > 10000) ||
      anyDuplicated(ns) || anyDuplicated(designs) || anyDuplicated(estimators) || anyDuplicated(transforms) ||
      any(!designs %in% names(designs_all())) || any(!estimators %in% c("ML", "ULS", "GLS")) ||
      any(!transforms %in% c("native", "mixed")) || !grepl("^[A-Za-z0-9_-]+$", run_id))
    stop("invalid ordinary options")
  require_pkg("magmaanlab", "install the sphere-native-audit branch, with gauge$native_audit")
  out <- file.path(here, "results", run_id)
  if (dir.exists(out)) stop("run directory exists; choose a fresh --run-id")
  dir.create(out, recursive = TRUE)
  write_out <- function(x, name) write_csv(x, file.path(out, paste0(name, ".csv")))
  source_files <- c(file.path(here, "run_experiment.R"),
                    file.path(here, "R", c("ordinary_program.R", "designs.R", "fit.R")),
                    file.path(here, "../../../_support/R/helpers.R"))
  source_hash <- paste(names(tools::md5sum(source_files)), tools::md5sum(source_files), collapse = ";")
  package_files <- list.files(find.package("magmaanlab"), recursive = TRUE, full.names = TRUE)
  package_hash <- paste(tools::md5sum(package_files[grepl("\\.(so|rdb|rdx)$", package_files)]), collapse = ";")
  # Tight profiles change stopping thresholds only, retaining stock budgets,
  # L-BFGS memory and internal gradient tolerance for attribution.
  profiles <- list(
    default = list(optimizer = NULL, control = NULL),
    lbfgs_tight = list(optimizer = "nlopt-lbfgs", control = list(nlopt = list(ftol_rel = 1e-12, xtol_rel = 1e-10))),
    port = list(optimizer = "port", control = NULL),
    port_tight = list(optimizer = "port", control = list(port = list(rel_f_tol = 1e-12, x_tol = 1e-10))))
  grid <- expand.grid(design = designs, n = ns, rep = seq_len(reps), transform = transforms,
                      stringsAsFactors = FALSE)
  arms <- expand.grid(estimator = estimators, chart = c("marker", "sphere"),
                      start_id = c("native", "shared_fabin3"), profile = names(profiles),
                      stringsAsFactors = FALSE)
  spec <- magmaanlab::model_spec(model_syntax)
  ref <- magmaan_cache_ref()
  meta <- list(lane = "ordinary", profile = if (smoke) "smoke" else "pilot", seed_base = seed_base,
    seed_rule = "base + stable design index*100000 + N*100 + rep; unit transforms share draws",
    reps = reps, ns = ns, designs = designs, estimators = estimators, transforms = transforms,
    planned_fits = nrow(grid) * nrow(arms), source_md5 = source_hash, package_md5 = package_hash,
    git_head = ref$git_head, git_dirty = ref$git_dirty, magmaanlab_path = find.package("magmaanlab"),
    magmaanlab_built = utils::packageDescription("magmaanlab")$Built,
    sample_scaling = "N for every estimator and chart", domain = "unrestricted; no PSD/barriers/bounds",
    sphere_polish = FALSE, sphere_metric = "unit_free", sphere_pin_weight = 1, chart_tolerance = 1e-6,
    audit_budget = .01, audit_max_condition = 1e12,
    audit_judge = "library native endpoint verdict, plus independent same-point objective consistency",
    first_order = "telemetry only; sphere product-Euclidean and marker model-Frobenius norms differ",
    start_design = "route-native; shared auto-transported FABIN3 vector constructed once before fitting",
    sphere_ls_native = "existing canonical path includes an auxiliary ML fit; time includes that work",
    optimizer_profiles = "default no options; tight f=1e-12 x=1e-10; PORT stock/tight; budgets unchanged",
    reference = "lowest locally audited objective; repeated only across both backends with implied covariance agreement",
    evidence_role = "exploratory development, not held-out default confirmation")
  write_metadata(file.path(out, "metadata.csv"), values = meta, packages = "magmaanlab")
  rows <- list(); t0 <- proc.time()[["elapsed"]]
  cat(sprintf("%d draws, %d fits; output %s\n", nrow(grid), nrow(grid) * nrow(arms), out))
  for (i in seq_len(nrow(grid))) {
    task <- grid[i, ]
    seed <- seed_base + match(task$design, names(designs_all())) * 100000L + task$n * 100L + task$rep
    data <- draw_data(design_sigma(designs_all()[[task$design]]), task$n, seed)
    scale <- if (task$transform == "native") rep(1, 6) else c(.01, 100, 2, .3, 10, .1)
    data[] <- sweep(as.matrix(data), 2, scale, "*")
    sample <- magmaanlab::df_to_data(data, spec, scaling = "n")
    saveRDS(list(task = task, seed = seed, sample = sample), file.path(out, sprintf("sample_%03d.rds", i)))
    shared <- tryCatch(magmaanlab::magmaan_core$estimate_start_values(
      spec$partable, sample, start = "fabin3", transport = "auto"), error = identity)
    if (!inherits(shared, "error")) append_csv(data.frame(draw_id = i, start_id = "shared_fabin3",
      parameter = seq_along(shared), value = as.numeric(shared), method = attr(shared, "start_method"),
      transport = attr(shared, "start_transport"), fallback = attr(shared, "start_fallback_reason")),
      file.path(out, "starts.csv"))
    draw_rows <- list(); endpoints <- list()
    for (j in seq_len(nrow(arms))) {
      arm <- arms[j, ]; p <- profiles[[arm$profile]]
      ctl <- p$control; target <- spec
      if (arm$start_id == "shared_fabin3" && !inherits(shared, "error")) {
        if (arm$chart == "marker") ctl <- modifyList(ctl %||% list(), list(start = as.numeric(shared)))
        else {
          free <- target$partable$free
          target$partable$ustart[free > 0] <- shared[free[free > 0]]
        }
      }
      warnings <- character(); fit_t0 <- proc.time()[["elapsed"]]
      fit <- tryCatch(withCallingHandlers({
        if (arm$start_id == "shared_fabin3" && inherits(shared, "error")) stop(shared)
        if (arm$chart == "marker") magmaanlab::fit_model(target, sample,
          estimator = arm$estimator, optimizer = p$optimizer, control = ctl)
        else magmaanlab::frontier_fit_sphere(target, sample, estimator = arm$estimator,
          optimizer = p$optimizer, control = ctl, polish = FALSE,
          start = if (arm$start_id == "native") "canonical" else "user")
      }, warning = function(w) { warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning") }),
        error = identity)
      elapsed <- proc.time()[["elapsed"]] - fit_t0
      result <- ordinary_endpoint(fit, spec, sample, arm$estimator, arm$chart)
      if (arm$chart == "sphere" && result$record$returned && is.null(fit$gauge$native_audit))
        stop("installed magmaanlab lacks the sphere-native audit; rebuild the audit branch")
      id <- length(rows) + 1L
      record <- cbind(fit_id = id, draw_id = i, task, seed = seed, arm,
        backend = if (arm$profile %in% c("port", "port_tight")) "port" else "nlopt-lbfgs",
        seconds = elapsed, result$record, warnings = paste(unique(warnings), collapse = " | "))
      rows[[id]] <- draw_rows[[j]] <- record
      endpoints[[j]] <- list(fit_id = id, fit = fit, sigma = result$sigma)
      if (arm$chart == "marker" && !inherits(fit, "error") && length(fit$start$theta))
        append_csv(data.frame(draw_id = i, start_id = paste(arm$estimator, arm$start_id, arm$profile, sep = ":"),
          parameter = seq_along(fit$start$theta), value = fit$start$theta,
          method = fit$start$method %||% "", transport = fit$start$transport %||% "",
          fallback = fit$start$fallback_reason %||% ""), file.path(out, "starts.csv"))
      if (j %% 12L == 0L) cat(sprintf("draw %d/%d fits %d/%d; %.1fs\n", i, nrow(grid), j,
        nrow(arms), proc.time()[["elapsed"]] - t0))
    }
    append_csv(do.call(rbind, draw_rows), file.path(out, "fits.csv"))
    saveRDS(endpoints, file.path(out, sprintf("endpoints_%03d.rds", i)))
    elapsed <- proc.time()[["elapsed"]] - t0
    write_out(data.frame(completed_draws = i, draws = nrow(grid), elapsed_s = elapsed,
      eta_s = elapsed / i * (nrow(grid) - i)), "progress")
  }
  fits <- do.call(rbind, rows)
  comparisons <- ordinary_comparisons(fits, out)
  write_out(comparisons$paired, "paired"); write_out(comparisons$references, "references")
  keys <- c("design", "n", "transform", "estimator", "chart", "start_id", "profile")
  summary <- do.call(rbind, lapply(split(comparisons$paired, interaction(comparisons$paired[keys], drop = TRUE)), function(d)
    cbind(d[1, keys], data.frame(fits = nrow(d), returned = sum(d$returned),
      audited = sum(d$accepted), user_chart = sum(d$user_chart %in% TRUE),
      reference_available = sum(is.finite(d$best_objective)), best_observed_hits = sum(d$best_hit),
      repeated_reference_hits = sum(d$best_hit & d$reference_repeated),
      median_seconds = median(d$seconds), max_seconds = max(d$seconds)))))
  write_out(summary, "summary")
  audit_keys <- c("design", "estimator", "chart", "audit_status", "newton_status", "accepted")
  write_out(aggregate(list(fits = rep(1L, nrow(fits))), fits[audit_keys], sum), "audit_summary")
  write_out(aggregate(list(fits = rep(1L, nrow(fits))),
    fits[c("optimizer_status", "call_status", "accepted")], sum), "stop_summary")
  paired <- comparisons$paired
  changes <- paired[paired$acceptance_gain | paired$acceptance_loss | paired$recovery_gain |
    paired$recovery_loss | (paired$accepted & !paired$best_hit), ]
  write_out(changes[c("draw_id", "design", "n", "rep", "transform", "seed", "estimator", "chart",
    "start_id", "profile", "audit_status", "newton_status", "newton_distance", "objective_gap",
    "acceptance_gain", "acceptance_loss", "recovery_gain", "recovery_loss", "reference_repeated")], "changes")
  meta$elapsed_s <- proc.time()[["elapsed"]] - t0; meta$completed_fits <- nrow(fits)
  write_metadata(file.path(out, "metadata.csv"), values = meta, packages = "magmaanlab")
  cat(sprintf("Wrote %d fits in %.1fs to %s\n", nrow(fits), meta$elapsed_s, out))
  invisible(fits)
}

ordinary_endpoint <- function(fit, spec, sample, estimator, chart) {
  rec <- data.frame(returned = FALSE, call_status = "error", audit_status = "unavailable",
    accepted = FALSE, final_verdict = "unavailable", user_chart = NA, optimizer_status = "",
    objective = NA_real_, recomputed_objective = NA_real_, objective_consistent = NA,
    newton_status = "unavailable", newton_distance = NA_real_, condition = NA_real_,
    curvature = "", accuracy_metric = "", first_order_residual = NA_real_, first_order_metric = "",
    newton_budget = NA_real_, implied_sigma_pd = NA, primitive_admissible = NA,
    chart_level = NA_real_, pin_residual = NA_real_, start_used = "", iterations = NA_integer_,
    f_evals = NA_integer_, g_evals = NA_integer_, detail = "", message = "", stringsAsFactors = FALSE)
  sigma <- NULL
  endpoint <- !inherits(fit, "error") || inherits(fit, "magmaan_sphere_condition")
  if (!endpoint) { rec$message <- conditionMessage(fit); return(list(record = rec, sigma = sigma)) }
  rec$returned <- TRUE
  rec$call_status <- if (inherits(fit, "error")) class(fit)[1] else "returned"
  rec$final_verdict <- fit$verdict$status %||% "unavailable"
  if (chart == "sphere") {
    g <- fit$gauge; a <- g$native_audit; na <- a$newton_accuracy
    rec$audit_status <- a$status %||% "unavailable"
    rec$objective <- a$fmin %||% NA_real_
    rec$user_chart <- isTRUE(g$user_chart)
    rec$optimizer_status <- g$optimizer_status
    rec$first_order_residual <- a$first_order_residual %||% NA_real_
    rec$first_order_metric <- a$first_order_metric %||% ""
    rec$chart_level <- if (nrow(g$units)) min(abs(g$units$direction_level)) else NA_real_
    rec$pin_residual <- g$pin_residual
    rec$start_used <- g$start; rec$iterations <- g$iterations
    rec$detail <- a$detail %||% ""
    # Diagnostic translation only, with no refit and no change to the requested
    # chart verdict. A strong marker lets us recompute covariance at near-poles.
    ev <- tryCatch({
      target <- strong_marker_model(g$sphere_partable, sample)
      translated <- magmaanlab::frontier_reidentify(g$sphere_partable, target, pole_tol = 0)
      magmaanlab::magmaan_core$evaluate_at(target$partable, sample, translated$theta, estimator = estimator)
    }, error = identity)
  } else {
    na <- fit$diagnostics$newton_accuracy
    rec$audit_status <- fit$verdict$status %||% "unavailable"
    rec$user_chart <- TRUE; rec$objective <- fit$fmin
    rec$optimizer_status <- fit$optimizer_status
    rec$first_order_residual <- fit$diagnostics$geometric_stationarity$ambient_residual_l2 %||% NA_real_
    rec$first_order_metric <- "model_frobenius"
    rec$start_used <- fit$start$method %||% ""; rec$iterations <- fit$iterations
    rec$f_evals <- fit$f_evals; rec$g_evals <- fit$g_evals
    ev <- tryCatch(magmaanlab::magmaan_core$evaluate_at(spec$partable, sample, fit$theta,
      estimator = estimator), error = identity)
  }
  rec$newton_status <- na$status %||% "unavailable"
  rec$newton_distance <- na$distance %||% NA_real_; rec$condition <- na$condition %||% NA_real_
  rec$curvature <- na$curvature %||% ""; rec$accuracy_metric <- na$metric %||% ""
  rec$newton_budget <- na$budget %||% NA_real_
  if (inherits(ev, "error")) rec$message <- conditionMessage(ev)
  else {
    rec$recomputed_objective <- ev$fmin
    rec$objective_consistent <- is.finite(rec$objective) && is.finite(ev$fmin) &&
      abs(rec$objective - ev$fmin) <= 1e-6 * (1 + abs(rec$objective))
    rec$implied_sigma_pd <- ev$diagnostics$sigma_pd_all
    rec$primitive_admissible <- ev$diagnostics$admissibility$admissible
    sigma <- magmaanlab::magmaan_core$model_implied(ev)$sigma[[1]]
  }
  rec$accepted <- rec$audit_status == "passed" && isTRUE(rec$objective_consistent)
  list(record = rec, sigma = sigma)
}

ordinary_comparisons <- function(fits, out) {
  keys <- c("draw_id", "estimator")
  groups <- split(fits, interaction(fits[keys], drop = TRUE))
  refs <- paired <- list()
  for (k in seq_along(groups)) {
    d <- groups[[k]]; good <- d[d$accepted & is.finite(d$objective), ]
    best <- if (nrow(good)) min(good$objective) else NA_real_
    near <- if (nrow(good)) good[abs(good$objective - best) <= 1e-6 * (1 + abs(best)), ] else good
    sigma_gap <- NA_real_
    if (nrow(near) > 1) {
      endpoints <- readRDS(file.path(out, sprintf("endpoints_%03d.rds", d$draw_id[1])))
      sigmas <- lapply(near$fit_id, function(id) endpoints[[match(id, vapply(endpoints, `[[`, 0L, "fit_id"))]]$sigma)
      sample <- readRDS(file.path(out, sprintf("sample_%03d.rds", d$draw_id[1])))$sample
      sd <- sqrt(diag(sample$S[[1]])); anchor <- sigmas[[which.min(near$objective)]]
      sigma_gap <- max(vapply(sigmas, function(s) max(abs((s - anchor) / outer(sd, sd))), 0.0))
    }
    repeated <- length(unique(near$backend)) >= 2 && is.finite(sigma_gap) && sigma_gap <= 1e-5
    refs[[k]] <- cbind(d[1, c("draw_id", "design", "n", "rep", "transform", "seed", "estimator")],
      best_objective = best, audited_candidates = nrow(good), best_backends = length(unique(near$backend)),
      max_standardized_sigma_gap = sigma_gap, reference_repeated = repeated)
    d$best_objective <- best; d$objective_gap <- d$objective - best
    d$best_hit <- d$accepted & is.finite(best) & is.finite(d$objective) &
      abs(d$objective_gap) <= 1e-6 * (1 + abs(best))
    d$reference_repeated <- repeated
    # Paired baseline is the actual no-options fit in the same chart. Shared
    # starts and tight controls are explicit ablations, never the baseline.
    b <- d[d$start_id == "native" & d$profile == "default", ]
    idx <- match(d$chart, b$chart)
    d$baseline_accepted <- b$accepted[idx]; d$baseline_best_hit <- d$best_hit[match(b$fit_id[idx], d$fit_id)]
    d$acceptance_gain <- d$accepted & !d$baseline_accepted
    d$acceptance_loss <- !d$accepted & d$baseline_accepted
    d$recovery_gain <- d$best_hit & !d$baseline_best_hit
    d$recovery_loss <- !d$best_hit & d$baseline_best_hit
    paired[[k]] <- d
  }
  list(paired = do.call(rbind, paired), references = do.call(rbind, refs))
}
