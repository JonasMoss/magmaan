# Positive scalar weights preserve the ULS minimizers. Every endpoint is
# re-evaluated under the original ULS target, irrespective of its driven scale.
run_uls_unit_probe <- function(args, here) {
  if (any(args %in% c("--help", "-h"))) {
    cat(paste("Usage: Rscript run_experiment.R --ordinary --uls-unit-probe [options]",
      "--run-id NAME --source-run ordinary-scaling-units-v4",
      "--fresh-run ordinary-scaling-fresh-units-v2 (omit with --smoke)",
      "--reference-digits 0|60|90 --python PATH (requires mpmath for references)",
      "Two shared sample-only starts; marker/sphere; L-BFGS, PORT and PORT-NLS",
      "stock/long-tight controls; scalar weights 1,1e-4,1e-8 on the same ULS target.",
      "Uses this leaf's saved mixed-unit samples; missing inputs need regeneration",
      "with the ordinary unit-change runner. Fresh run IDs required.", sep = "\n"), "\n")
    return(invisible(NULL))
  }
  known <- c("--ordinary", "--uls-unit-probe", "--smoke", "--run-id", "--source-run",
             "--fresh-run", "--reference-digits", "--python")
  if (any(startsWith(args, "--") & !args %in% known)) stop("unknown ULS probe option")
  value <- function(k, default) {
    i <- match(k, args)
    if (is.na(i)) return(default)
    if (i == length(args) || startsWith(args[i + 1L], "--")) stop("missing value for ", k)
    args[i + 1L]
  }
  run <- value("--run-id", "uls-unit-probe")
  sources <- value("--source-run", "ordinary-scaling-units-v4")
  if (!"--smoke" %in% args) sources <- c(sources, value("--fresh-run", "ordinary-scaling-fresh-units-v2"))
  digits <- as.integer(value("--reference-digits", "0"))
  if (any(!grepl("^[A-Za-z0-9_-]+$", c(run, sources))) || is.na(digits) || !digits %in% c(0, 60, 90))
    stop("invalid ULS probe options")
  require_pkg("magmaanlab")
  out <- file.path(here, "results", run)
  if (dir.exists(out)) stop("choose a fresh ULS probe run-id")
  inputs <- unlist(lapply(sources, function(name) {
    files <- list.files(file.path(here, "results", name), pattern = "^sample_[0-9]+\\.rds$", full.names = TRUE)
    if (!length(files)) stop("missing saved samples for ", name,
      "; regenerate using run_experiment.R --ordinary --smoke --transforms native,mixed (fresh draws: --reps 4 --seed-base 620260004)")
    files[vapply(files, function(file) readRDS(file)$task$transform == "mixed", FALSE)]
  }))
  if (!length(inputs)) stop("no mixed-unit samples")
  if (digits > 0 && any(vapply(inputs, function(file) readRDS(file)$sample$nobs != 100L, FALSE)))
    stop("the independent reference audit currently requires N=100")
  dir.create(out, recursive = TRUE)
  write_out <- function(d, name) write_csv(d, file.path(out, paste0(name, ".csv")))
  profiles <- list(
    lbfgs = list(optimizer = "nlopt-lbfgs", control = NULL),
    lbfgs_more = list(optimizer = "nlopt-lbfgs", control = list(max_iter = 10000,
      nlopt = list(ftol_rel = 1e-14, xtol_rel = 1e-12))),
    port = list(optimizer = "port", control = NULL),
    port_more = list(optimizer = "port", control = list(max_iter = 10000,
      port = list(rel_f_tol = 1e-14, x_tol = 1e-12))),
    nls = list(optimizer = "port-nls", control = NULL),
    nls_more = list(optimizer = "port-nls", control = list(max_iter = 10000,
      port = list(rel_f_tol = 1e-14, x_tol = 1e-12))))
  arms <- expand.grid(chart = c("marker", "sphere"), start_id = c("fabin3", "layered"),
    profile = names(profiles), objective_scale = c(1, 1e-4, 1e-8), stringsAsFactors = FALSE)
  spec <- magmaanlab::model_spec(model_syntax)
  source_files <- c(file.path(here, "run_experiment.R"), file.path(here, "R",
    c("uls_unit_probe.R", "fit.R", "ordinary_program.R", "designs.R")),
    file.path(here, "scripts/uls_unit_reference.py"), file.path(here, "../../../_support/R/helpers.R"))
  package_files <- list.files(find.package("magmaanlab"), recursive = TRUE, full.names = TRUE)
  ref <- magmaan_cache_ref()
  meta <- list(lane = "uls_unit_probe", input_md5 = paste(tools::md5sum(inputs), collapse = ";"),
    source_md5 = paste(tools::md5sum(source_files), collapse = ";"),
    package_md5 = paste(tools::md5sum(package_files[grepl("\\.(so|rdb|rdx)$", package_files)]), collapse = ";"),
    git_head = ref$git_head, git_dirty = ref$git_dirty, sources = sources,
    judge = "original ULS same-point strong-marker audit; no threshold changes",
    objective_scaling = "positive scalar W=c*I preserves minimizers; absolute backend stops and pin/objective ratio may change",
    starts = "shared FABIN3(auto) and layered(native), computed before fits",
    reference_digits = digits, planned_fits = length(inputs) * nrow(arms))
  write_metadata(file.path(out, "metadata.csv"), values = meta, packages = "magmaanlab")
  rows <- covariance <- seeds <- list(); t0 <- proc.time()[["elapsed"]]
  for (case_id in seq_along(inputs)) {
    saved <- readRDS(inputs[case_id]); sample <- saved$sample
    saveRDS(saved, file.path(out, sprintf("sample_%03d.rds", case_id)))
    starts <- lapply(c("fabin3", "layered"), function(method)
      magmaanlab::magmaan_core$estimate_start_values(spec$partable, sample, start = method,
        transport = if (method == "layered") "native" else "auto"))
    names(starts) <- c("fabin3", "layered")
    points <- list(); best <- NULL; best_value <- Inf
    for (j in seq_len(nrow(arms))) {
      arm <- arms[j, ]; p <- profiles[[arm$profile]]; target <- spec; ctl <- p$control
      theta <- starts[[arm$start_id]]
      if (arm$chart == "marker") ctl <- modifyList(ctl %||% list(), list(start = as.numeric(theta)))
      else { free <- target$partable$free; target$partable$ustart[free > 0] <- theta[free[free > 0]] }
      warnings <- character(); start_time <- proc.time()[["elapsed"]]
      fit <- tryCatch(withCallingHandlers({
        W <- diag(21) * arm$objective_scale
        if (arm$chart == "marker") magmaanlab::fit_model(target, sample, estimator = "WLS", W = W,
          optimizer = p$optimizer, control = ctl)
        else magmaanlab::frontier_fit_sphere(target, sample, estimator = "WLS", W = W,
          optimizer = p$optimizer, control = ctl, start = "user", polish = FALSE)
      }, warning = function(w) { warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning") }), error = identity)
      returned <- !inherits(fit, "error") || inherits(fit, "magmaan_sphere_condition")
      pt <- if (!returned) NULL else if (arm$chart == "sphere") fit$gauge$sphere_partable else fit$partable
      checked <- if (is.null(pt)) NULL else tryCatch({
        strong <- strong_marker_model(pt, sample)
        tr <- magmaanlab::frontier_reidentify(pt, strong, pole_tol = 0)
        magmaanlab::magmaan_core$evaluate_at(strong$partable, sample, tr$theta, estimator = "ULS")
      }, error = identity)
      available <- !is.null(checked) && !inherits(checked, "error")
      na <- if (available) checked$diagnostics$newton_accuracy else NULL
      native <- if (returned && arm$chart == "sphere") fit$gauge$native_audit else NULL
      reported <- if (!returned) NA_real_ else if (arm$chart == "sphere") fit$gauge$fmin_sphere else fit$fmin
      objective <- if (available) checked$fmin else NA_real_
      consistent <- available && is.finite(reported) &&
        abs(objective - reported / arm$objective_scale) <= 1e-6 * (1 + abs(objective))
      record <- cbind(case_id = case_id, seed = saved$seed, source_sample = inputs[case_id], arm,
        returned = returned, objective = objective, objective_consistent = consistent,
        original_audit = if (available) checked$verdict$status else "unavailable",
        original_newton_status = na$status %||% "unavailable", original_distance = na$distance %||% NA_real_,
        original_condition = na$condition %||% NA_real_, native_audit = native$status %||% "unavailable",
        curvature_status = native$curvature_system$status %||% "unavailable",
        curvature_condition = native$curvature_system$condition %||% NA_real_,
        metric_status = native$accuracy_metric_system$status %||% "unavailable",
        metric_condition = native$accuracy_metric_system$condition %||% NA_real_,
        seconds = proc.time()[["elapsed"]] - start_time,
        message = if (inherits(fit, "error")) conditionMessage(fit) else "",
        warnings = paste(unique(warnings), collapse = " | "))
      rows[[length(rows) + 1L]] <- record; points[[j]] <- list(fit = fit, checked = checked)
      if (available && consistent && arm$profile %in% c("nls", "nls_more") &&
          arm$objective_scale == 1 && objective < best_value) {
        best <- pt; best_value <- objective
      }
      if (j %% 12 == 0) cat(sprintf("ULS case %d/%d: %d/%d fits; %.1fs\n", case_id, length(inputs), j, nrow(arms), proc.time()[["elapsed"]] - t0))
    }
    if (is.null(best)) stop("no finite unscaled least-squares candidate for reference refinement")
    point_value <- function(lhs, op, rhs) best$est[best$lhs == lhs & best$op == op & best$rhs == rhs][1]
    scales <- c(.01, 100, 2, .3, 10, .1)
    lx <- vapply(ov_names[1:3], function(name) point_value("X", "=~", name), 0.0) / scales[1:3]
    ly <- vapply(ov_names[4:6], function(name) point_value("Y", "=~", name), 0.0) / scales[4:6]
    vx <- point_value("X", "~~", "X"); beta <- point_value("Y", "~", "X"); psi <- point_value("Y", "~~", "Y")
    start <- c(lx[c(1, 3)] / lx[2], ly[c(1, 3)] / ly[2], vx * lx[2]^2,
               beta * vx * lx[2] * ly[2], (beta^2 * vx + psi) * ly[2]^2)
    seeds[[case_id]] <- data.frame(case_id = case_id, parameter = seq_along(start), value = start)
    covariance[[case_id]] <- expand.grid(case_id = case_id, row = 1:6, col = 1:6)
    covariance[[case_id]]$value <- as.vector(sample$S[[1]])
    saveRDS(points, file.path(out, sprintf("endpoints_%03d.rds", case_id)))
    write_out(do.call(rbind, rows), "fits")
  }
  fits <- do.call(rbind, rows)
  write_out(do.call(rbind, seeds), "reference_starts"); write_out(do.call(rbind, covariance), "covariances")
  write_out(aggregate(list(fits = rep(1L, nrow(fits)), returned = fits$returned,
    original_passes = fits$original_audit == "passed", consistent = fits$objective_consistent),
    fits[c("chart", "profile", "objective_scale")], sum), "summary")
  write_out(aggregate(list(fits = rep(1L, nrow(fits))),
    fits[c("chart", "profile", "original_newton_status", "curvature_status", "metric_status")], sum), "audit_summary")
  meta$elapsed_s <- proc.time()[["elapsed"]] - t0
  write_metadata(file.path(out, "metadata.csv"), values = meta, packages = "magmaanlab")
  if (digits > 0) {
    python <- value("--python", "python3")
    status <- system2(python, c(shQuote(file.path(here, "scripts/uls_unit_reference.py")),
      "--run-dir", shQuote(normalizePath(out)), "--digits", digits))
    if (status != 0) stop("high-precision reference failed; retained fit outputs remain in ", out)
    parameters <- read.csv(file.path(out, "reference_parameters.csv"))
    references <- read.csv(file.path(out, "reference_summary.csv"))
    qualified <- references$case_id[tolower(references$finite_local_minimum) == "true"]
    cold_checks <- warm_rows <- list()
    checks <- lapply(qualified, function(case_id) {
      pt <- spec$partable; p <- parameters[parameters$case_id == case_id, ]
      key <- function(d) paste(d$lhs, d$op, d$rhs)
      pt$est <- p$est[match(key(pt), key(p))]
      theta <- pt$est[pt$free > 0][order(pt$free[pt$free > 0])]
      sample <- readRDS(inputs[case_id])$sample
      ev <- magmaanlab::magmaan_core$evaluate_at(pt, sample, theta, estimator = "ULS")
      core_gap <- abs(ev$fmin - references$objective[references$case_id == case_id])
      if (core_gap > 1e-8 * (1 + abs(ev$fmin))) stop("independent ULS reference objective disagrees with core")
      sigma <- magmaanlab::magmaan_core$model_implied(ev)$sigma[[1]]
      sd <- sqrt(diag(sample$S[[1]])); units <- outer(sd, sd)
      points <- readRDS(file.path(out, sprintf("endpoints_%03d.rds", case_id)))
      d <- fits[fits$case_id == case_id, ]; d$reference_objective <- ev$fmin
      d$objective_gap <- d$objective - ev$fmin
      d$standardized_sigma_gap <- vapply(points, function(point) {
        checked <- point$checked
        if (is.null(checked) || inherits(checked, "error")) return(NA_real_)
        max(abs((magmaanlab::magmaan_core$model_implied(checked)$sigma[[1]] - sigma) / units))
      }, 0.0)
      d$reference_match <- d$objective_consistent & is.finite(d$objective_gap) &
        abs(d$objective_gap) <= 1e-6 * (1 + abs(ev$fmin)) &
        is.finite(d$standardized_sigma_gap) & d$standardized_sigma_gap <= 1e-5
      cold_checks[[length(cold_checks) + 1L]] <<- d
      # Imported fitted values are diagnostic warm starts only. The cold grid
      # above was completed and checkpointed before constructing references.
      for (chart in c("marker", "sphere")) {
        target <- spec; free <- target$partable$free
        target$partable$ustart[free > 0] <- theta[free[free > 0]]
        warm <- tryCatch(suppressWarnings(if (chart == "marker")
          magmaanlab::fit_model(target, sample, estimator = "ULS", optimizer = "port-nls",
            control = list(start = as.numeric(theta)))
          else magmaanlab::frontier_fit_sphere(target, sample, estimator = "ULS", optimizer = "port-nls",
            start = "user", polish = FALSE)), error = identity)
        endpoint <- ordinary_endpoint(warm, spec, sample, "ULS", chart)
        gap <- endpoint$record$objective - ev$fmin
        covariance_gap <- if (is.null(endpoint$sigma)) NA_real_ else max(abs((endpoint$sigma - sigma) / units))
        warm_rows[[length(warm_rows) + 1L]] <<- cbind(case_id = case_id, chart = chart,
          endpoint$record[c("returned", "accepted", "objective", "newton_status", "condition")],
          objective_gap = gap, standardized_sigma_gap = covariance_gap,
          reference_match = is.finite(gap) && abs(gap) <= 1e-6 * (1 + abs(ev$fmin)) &&
            is.finite(covariance_gap) && covariance_gap <= 1e-5)
      }
      data.frame(case_id = case_id, objective = ev$fmin, objective_gap = core_gap,
        audit_status = ev$verdict$status, newton_status = ev$diagnostics$newton_accuracy$status,
        condition = ev$diagnostics$newton_accuracy$condition, distance = ev$diagnostics$newton_accuracy$distance)
    })
    write_out(do.call(rbind, checks), "reference_core_checks")
    cold <- do.call(rbind, cold_checks)
    write_out(cold, "reference_checks")
    write_out(aggregate(list(fits = rep(1L, nrow(cold)), reference_matches = cold$reference_match),
      cold[c("chart", "profile", "objective_scale")], sum), "reference_hits")
    warm <- do.call(rbind, warm_rows)
    write_out(warm, "warm_checks")
    write_out(aggregate(list(fits = rep(1L, nrow(warm)), accepted = warm$accepted,
      reference_matches = warm$reference_match), warm["chart"], sum), "warm_summary")
  }
  cat("Wrote ULS probe to", out, "\n")
}
