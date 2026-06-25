#!/usr/bin/env Rscript

`%||%` <- function(x, y) if (is.null(x) || length(x) == 0 || is.na(x)) y else x

script_path <- {
  cmd <- commandArgs(FALSE)
  hit <- grep("^--file=", cmd, value = TRUE)
  if (length(hit)) sub("^--file=", "", hit[[1]]) else NA_character_
}
script_dir <- if (!is.na(script_path)) dirname(normalizePath(script_path)) else getwd()

usage <- function() {
  cat(
    "Usage: Rscript run_experiment.R [options]\n",
    "\n",
    "Reproduce focused summaries from Jamil, Rosseel, Kemp, and Kosmidis' SEM\n",
    "bias-reduction paper using the authors' published OSF result objects.\n",
    "\n",
    "Options:\n",
    "  --download           Download required OSF result files into resources/osf/.\n",
    "  --full               Also require the two-factor split files needed for the\n",
    "                       non-normal/all-condition summaries.\n",
    "  --smoke              Limit summaries to the first 25 simulations per cell.\n",
    "  --max-sims N         Limit summaries to simulations with sim <= N.\n",
    "  --source-dir PATH    Directory containing OSF .RData files.\n",
    "                       Default: resources/osf\n",
    "  --results-dir PATH   Output directory. Default: results\n",
    "  --help               Show this help.\n",
    "\n",
    "Default output reproduces the paper's two main ML/eRBM/iRBM examples:\n",
    "  * two-factor SEM, normal data, reliability 0.8 (paper Figure 4)\n",
    "  * growth curve model, normal data, reliability 0.5 (paper Figure 6)\n",
    sep = ""
  )
}

parse_args <- function(args) {
  opts <- list(
    download = FALSE,
    full = FALSE,
    max_sims = NA_integer_,
    source_dir = file.path(script_dir, "resources", "osf"),
    results_dir = file.path(script_dir, "results")
  )

  i <- 1L
  while (i <= length(args)) {
    arg <- args[[i]]
    if (arg == "--help") {
      usage()
      quit(status = 0)
    } else if (arg == "--download") {
      opts$download <- TRUE
    } else if (arg == "--full") {
      opts$full <- TRUE
    } else if (arg == "--smoke") {
      opts$max_sims <- 25L
    } else if (arg == "--max-sims") {
      i <- i + 1L
      if (i > length(args)) stop("--max-sims needs a value", call. = FALSE)
      opts$max_sims <- as.integer(args[[i]])
    } else if (arg == "--source-dir") {
      i <- i + 1L
      if (i > length(args)) stop("--source-dir needs a value", call. = FALSE)
      opts$source_dir <- args[[i]]
    } else if (arg == "--results-dir") {
      i <- i + 1L
      if (i > length(args)) stop("--results-dir needs a value", call. = FALSE)
      opts$results_dir <- args[[i]]
    } else {
      stop("Unknown option: ", arg, call. = FALSE)
    }
    i <- i + 1L
  }
  opts
}

opts <- parse_args(commandArgs(TRUE))
dir.create(opts$source_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(opts$results_dir, recursive = TRUE, showWarnings = FALSE)

osf_files <- list(
  twofac_mp1 = list(
    file = "simu_res_twofac_mp1.RData",
    url = "https://osf.io/download/67e89df2f2b09a00f1829de3/"
  ),
  twofac_mp2 = list(
    file = "simu_res_twofac_mp2.RData",
    url = "https://osf.io/download/d3cbh/"
  ),
  twofac_mp3 = list(
    file = "simu_res_twofac_mp3.RData",
    url = "https://osf.io/download/63buh/"
  ),
  growth = list(
    file = "simu_res_growth.RData",
    url = "https://osf.io/download/wexcu/"
  ),
  serobust_twofac = list(
    file = "simu_res_serobust_twofac.RData",
    url = "https://osf.io/download/68664a0f31bfcde29949f1a7/"
  ),
  serobust_growth = list(
    file = "simu_res_serobust_growth.RData",
    url = "https://osf.io/download/686649e6d0f0d2d59849f0ea/"
  )
)

needed <- c("twofac_mp3", "growth", "serobust_twofac", "serobust_growth")
if (isTRUE(opts$full)) needed <- unique(c("twofac_mp1", "twofac_mp2", needed))

local_path <- function(key) file.path(opts$source_dir, osf_files[[key]]$file)

if (isTRUE(opts$download)) {
  for (key in needed) {
    path <- local_path(key)
    if (file.exists(path)) next
    message("Downloading ", osf_files[[key]]$file)
    download.file(osf_files[[key]]$url, destfile = path, mode = "wb", quiet = FALSE)
  }
}

missing <- vapply(needed, function(key) !file.exists(local_path(key)), logical(1))
if (any(missing)) {
  stop(
    "Missing OSF result files in ", opts$source_dir, ": ",
    paste(vapply(needed[missing], function(key) osf_files[[key]]$file, character(1)), collapse = ", "),
    "\nRun with --download or copy the files there first.",
    call. = FALSE
  )
}

load_object <- function(path, name) {
  env <- new.env(parent = emptyenv())
  load(path, envir = env)
  if (!exists(name, envir = env, inherits = FALSE)) {
    stop("Expected object ", name, " in ", path, call. = FALSE)
  }
  get(name, envir = env, inherits = FALSE)
}

load_twofac <- function(full) {
  twofac <- load_object(local_path("twofac_mp3"), "simu_res_twofac")
  if (isTRUE(full)) {
    mp1 <- load_object(local_path("twofac_mp1"), "simu_res_twofac")
    mp2 <- load_object(local_path("twofac_mp2"), "simu_res_twofac")
    twofac[11:19] <- mp2[11:19]
    twofac[20:30] <- mp1[20:30]
  }
  twofac
}

main_results <- list(
  twofac = load_twofac(opts$full),
  growth = load_object(local_path("growth"), "simu_res_growth")
)
robust_results <- list(
  twofac = load_object(local_path("serobust_twofac"), "simu_res_serobust_twofac"),
  growth = load_object(local_path("serobust_growth"), "simu_res_serobust_growth")
)

focal_params <- list(
  twofac = c("y1~~y1", "fx~~fx", "fy~~fy", "fy~fx", "fx=~x2"),
  growth = c("v", "i~~i", "s~~s", "i~~s")
)
param_labels <- c(
  "y1~~y1" = "theta_11",
  "fx~~fx" = "psi_11",
  "fy~~fy" = "psi_22",
  "fy~fx" = "beta",
  "fx=~x2" = "lambda_21",
  "v" = "theta_1",
  "i~~i" = "psi_11",
  "s~~s" = "psi_22",
  "i~~s" = "psi_12"
)
methods_main <- c("ML", "eRBM", "iRBM")

cell_key <- function(x) {
  if (is.null(x) || !nrow(x)) return(NA_character_)
  paste(x$model[[1]], x$dist[[1]], x$rel[[1]], x$n[[1]], sep = "\r")
}

row_key <- function(x) {
  paste(x$seed, x$sim, x$model, x$dist, x$rel, x$n, x$method, sep = "\r")
}

robust_map <- function(cells) {
  keys <- vapply(cells, cell_key, character(1))
  out <- cells[!is.na(keys)]
  names(out) <- keys[!is.na(keys)]
  out
}

value_at <- function(x, idx) {
  if (is.null(x) || length(x) < idx) return(NA_real_)
  as.numeric(x[[idx]])
}

param_index <- function(truth, param) {
  hit <- which(names(truth) == param)
  if (length(hit)) hit[[1]] else NA_integer_
}

scale_value <- function(model, value) {
  if (model == "growth") value * 100 else value
}

selected_cells <- function(cells, model, full) {
  keep <- vapply(cells, function(x) {
    if (is.null(x) || !nrow(x)) return(FALSE)
    if (!isTRUE(full)) {
      if (model == "twofac") return(x$dist[[1]] == "Normal" && x$rel[[1]] == "0.8")
      if (model == "growth") return(x$dist[[1]] == "Normal" && x$rel[[1]] == "0.5")
    }
    x$dist[[1]] %in% c("Normal", "Non-normal")
  }, logical(1))
  cells[keep]
}

long_for_model <- function(model, full, max_sims) {
  cells <- selected_cells(main_results[[model]], model, full)
  rob <- robust_map(robust_results[[model]])
  threshold <- if (model == "twofac") 5 else 500
  rows <- vector("list", 1024)
  k <- 0L

  for (cell in cells) {
    if (is.null(cell) || !nrow(cell)) next
    if (!is.na(max_sims)) cell <- cell[cell$sim <= max_sims, , drop = FALSE]
    cell <- cell[cell$method %in% methods_main, , drop = FALSE]
    rcell <- rob[[cell_key(cell)]]
    rse <- vector("list", nrow(cell))
    if (!is.null(rcell) && nrow(rcell)) {
      matched <- match(row_key(cell), row_key(rcell))
      ok <- !is.na(matched)
      rse[ok] <- rcell$se[matched[ok]]
    }

    for (i in seq_len(nrow(cell))) {
      truth <- cell$truth[[i]]
      est <- cell$est[[i]]
      se <- cell$se[[i]]
      serob <- rse[[i]]

      for (param in focal_params[[model]]) {
        idx <- param_index(truth, param)
        if (is.na(idx)) next

        est_v <- scale_value(model, value_at(est, idx))
        truth_v <- scale_value(model, value_at(truth, idx))
        se_v <- scale_value(model, value_at(se, idx))
        serob_v <- scale_value(model, value_at(serob, idx))
        bias_v <- est_v - truth_v
        included <- isTRUE(cell$converged[[i]]) &&
          is.finite(se_v) && abs(se_v) <= threshold
        covered <- is.finite(serob_v) &&
          truth_v >= est_v - qnorm(0.975) * serob_v &&
          truth_v <= est_v + qnorm(0.975) * serob_v

        k <- k + 1L
        if (k > length(rows)) length(rows) <- length(rows) * 2L
        rows[[k]] <- data.frame(
          model = model,
          dist = cell$dist[[i]],
          reliability = as.numeric(cell$rel[[i]]),
          n = as.integer(cell$n[[i]]),
          seed = as.integer(cell$seed[[i]]),
          sim = as.integer(cell$sim[[i]]),
          method = cell$method[[i]],
          param = param,
          param_label = unname(param_labels[[param]]),
          estimate = est_v,
          truth = truth_v,
          bias = bias_v,
          se = se_v,
          robust_se = serob_v,
          converged = isTRUE(cell$converged[[i]]),
          sigma_ok = isTRUE(cell$Sigma_OK[[i]]),
          included = included,
          covered = covered,
          stringsAsFactors = FALSE
        )
      }
    }
  }

  do.call(rbind, rows[seq_len(k)])
}

summarise_metrics <- function(long) {
  groups <- unique(long[c("model", "dist", "reliability", "n", "method", "param", "param_label")])
  out <- vector("list", nrow(groups))
  for (i in seq_len(nrow(groups))) {
    g <- groups[i, , drop = FALSE]
    ix <- long$model == g$model &
      long$dist == g$dist &
      long$reliability == g$reliability &
      long$n == g$n &
      long$method == g$method &
      long$param == g$param
    d <- long[ix & long$included, , drop = FALSE]
    out[[i]] <- data.frame(
      g,
      replications = nrow(d),
      bias = mean(d$bias, trim = 0.05, na.rm = TRUE),
      rmse = sqrt(mean(d$bias ^ 2, trim = 0.05, na.rm = TRUE)),
      pu = mean(d$bias < 0, na.rm = TRUE),
      coverage = mean(d$covered, na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, out)
}

summarise_by_method <- function(metrics) {
  groups <- unique(metrics[c("model", "dist", "reliability", "n", "method")])
  out <- vector("list", nrow(groups))
  for (i in seq_len(nrow(groups))) {
    g <- groups[i, , drop = FALSE]
    ix <- metrics$model == g$model &
      metrics$dist == g$dist &
      metrics$reliability == g$reliability &
      metrics$n == g$n &
      metrics$method == g$method
    d <- metrics[ix, , drop = FALSE]
    out[[i]] <- data.frame(
      g,
      mean_abs_bias = mean(abs(d$bias), na.rm = TRUE),
      mean_rmse = mean(d$rmse, na.rm = TRUE),
      mean_abs_pu_error = mean(abs(d$pu - 0.5), na.rm = TRUE),
      mean_abs_coverage_error = mean(abs(d$coverage - 0.95), na.rm = TRUE),
      min_param_replications = min(d$replications, na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, out)
}

acceptance_for_model <- function(model, full, max_sims) {
  cells <- selected_cells(main_results[[model]], model, full)
  rob <- robust_map(robust_results[[model]])
  threshold <- if (model == "twofac") 5 else 500
  rows <- list()
  k <- 0L

  for (cell in cells) {
    if (is.null(cell) || !nrow(cell)) next
    if (!is.na(max_sims)) cell <- cell[cell$sim <= max_sims, , drop = FALSE]
    cell <- cell[cell$method %in% methods_main, , drop = FALSE]
    rcell <- rob[[cell_key(cell)]]
    rse <- vector("list", nrow(cell))
    if (!is.null(rcell) && nrow(rcell)) {
      matched <- match(row_key(cell), row_key(rcell))
      ok <- !is.na(matched)
      rse[ok] <- rcell$se[matched[ok]]
    }

    for (method in methods_main) {
      ix <- which(cell$method == method)
      if (!length(ix)) next
      ok <- vapply(ix, function(i) {
        serob <- rse[[i]]
        se_ok <- is.numeric(serob) && length(serob) && all(is.finite(serob)) &&
          all(abs(scale_value(model, serob)) <= threshold)
        isTRUE(cell$converged[[i]]) && isTRUE(cell$Sigma_OK[[i]]) && se_ok
      }, logical(1))

      k <- k + 1L
      rows[[k]] <- data.frame(
        model = model,
        dist = cell$dist[[1]],
        reliability = as.numeric(cell$rel[[1]]),
        n = as.integer(cell$n[[1]]),
        method = method,
        replications = length(ix),
        acceptable = sum(ok, na.rm = TRUE),
        acceptance_rate = mean(ok, na.rm = TRUE),
        stringsAsFactors = FALSE
      )
    }
  }

  do.call(rbind, rows)
}

long <- rbind(
  long_for_model("twofac", opts$full, opts$max_sims),
  long_for_model("growth", opts$full, opts$max_sims)
)
metrics <- summarise_metrics(long)
method_summary <- summarise_by_method(metrics)
acceptance <- rbind(
  acceptance_for_model("twofac", opts$full, opts$max_sims),
  acceptance_for_model("growth", opts$full, opts$max_sims)
)

write.csv(metrics, file.path(opts$results_dir, "paper_main_metrics.csv"), row.names = FALSE)
write.csv(method_summary, file.path(opts$results_dir, "paper_main_method_summary.csv"), row.names = FALSE)
write.csv(acceptance, file.path(opts$results_dir, "paper_acceptance_rates.csv"), row.names = FALSE)

metadata <- data.frame(
  run_time = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
  source_dir = normalizePath(opts$source_dir, mustWork = FALSE),
  results_dir = normalizePath(opts$results_dir, mustWork = FALSE),
  full = opts$full,
  max_sims = ifelse(is.na(opts$max_sims), NA_integer_, opts$max_sims),
  r_version = paste(R.version$major, R.version$minor, sep = "."),
  stringsAsFactors = FALSE
)
write.csv(metadata, file.path(opts$results_dir, "metadata.csv"), row.names = FALSE)

cat("Wrote:\n")
cat("  ", file.path(opts$results_dir, "paper_main_metrics.csv"), "\n", sep = "")
cat("  ", file.path(opts$results_dir, "paper_main_method_summary.csv"), "\n", sep = "")
cat("  ", file.path(opts$results_dir, "paper_acceptance_rates.csv"), "\n", sep = "")
cat("  ", file.path(opts$results_dir, "metadata.csv"), "\n", sep = "")
