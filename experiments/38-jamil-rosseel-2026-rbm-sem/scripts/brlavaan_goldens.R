#!/usr/bin/env Rscript

script_path <- {
  cmd <- commandArgs(FALSE)
  hit <- grep("^--file=", cmd, value = TRUE)
  if (length(hit)) sub("^--file=", "", hit[[1]]) else NA_character_
}
script_dir <- if (!is.na(script_path)) dirname(normalizePath(script_path)) else getwd()
experiment_dir <- normalizePath(file.path(script_dir, ".."), mustWork = TRUE)

usage <- function() {
  cat(
    "Usage: Rscript scripts/brlavaan_goldens.R [options]\n",
    "\n",
    "Optional brlavaan-vs-magmaan smoke oracle for the Jamil/Rosseel SEM paper.\n",
    "The script skips with exit status 0 if the GitHub-only brlavaan package is\n",
    "not installed, unless --require-brlavaan is supplied.\n",
    "\n",
    "Options:\n",
    "  --n N                 Sample size. Default: 100\n",
    "  --rel VALUE           Reliability setting. Default: 0.8\n",
    "  --seed VALUE          Data-generation seed. Default: 1235\n",
    "  --bounds VALUE        Bounds preset, usually 'standard' or 'none'. Default: standard\n",
    "  --results-dir PATH    Output directory. Default: results\n",
    "  --require-brlavaan    Error instead of skip when brlavaan is absent.\n",
    "  --help                Show this help.\n",
    sep = ""
  )
}

parse_args <- function(args) {
  opts <- list(
    n = 100L,
    rel = 0.8,
    seed = 1235L,
    bounds = "standard",
    results_dir = file.path(experiment_dir, "results"),
    require_brlavaan = FALSE
  )
  i <- 1L
  while (i <= length(args)) {
    arg <- args[[i]]
    if (arg == "--help") {
      usage()
      quit(status = 0)
    } else if (arg == "--n") {
      i <- i + 1L
      if (i > length(args)) stop("--n needs a value", call. = FALSE)
      opts$n <- as.integer(args[[i]])
    } else if (arg == "--rel") {
      i <- i + 1L
      if (i > length(args)) stop("--rel needs a value", call. = FALSE)
      opts$rel <- as.numeric(args[[i]])
    } else if (arg == "--seed") {
      i <- i + 1L
      if (i > length(args)) stop("--seed needs a value", call. = FALSE)
      opts$seed <- as.integer(args[[i]])
    } else if (arg == "--bounds") {
      i <- i + 1L
      if (i > length(args)) stop("--bounds needs a value", call. = FALSE)
      opts$bounds <- args[[i]]
    } else if (arg == "--results-dir") {
      i <- i + 1L
      if (i > length(args)) stop("--results-dir needs a value", call. = FALSE)
      opts$results_dir <- args[[i]]
    } else if (arg == "--require-brlavaan") {
      opts$require_brlavaan <- TRUE
    } else {
      stop("Unknown option: ", arg, call. = FALSE)
    }
    i <- i + 1L
  }
  opts
}

opts <- parse_args(commandArgs(TRUE))
dir.create(opts$results_dir, recursive = TRUE, showWarnings = FALSE)

if (!requireNamespace("brlavaan", quietly = TRUE)) {
  msg <- paste0(
    "brlavaan is not installed; skipping optional oracle check.\n",
    "Install with remotes::install_github('haziqj/brlavaan') to run it."
  )
  if (isTRUE(opts$require_brlavaan)) stop(msg, call. = FALSE)
  message(msg)
  quit(status = 0)
}
if (!requireNamespace("magmaan", quietly = TRUE)) {
  stop("magmaan is not installed; run just r-dev first.", call. = FALSE)
}

br <- function(name) get(name, envir = asNamespace("brlavaan"), inherits = FALSE)

param_name <- function(partable) {
  paste(partable$lhs, partable$op, partable$rhs)
}

free_estimates <- function(fit) {
  pt <- fit$partable
  free <- which(pt$free > 0L)
  theta <- as.numeric(fit$theta)
  names(theta) <- param_name(pt[free, , drop = FALSE])
  theta
}

fit_brlavaan <- function(model, data, truth, method, bounds) {
  args <- list(
    model = model,
    data = data,
    rbm = switch(method, ML = "none", eRBM = "explicit", iRBM = "implicit"),
    plugin_pen = NULL,
    debug = FALSE,
    lavfun = "sem",
    maxgrad = FALSE,
    fn.scale = 1,
    bounds = bounds,
    start = truth,
    information = "observed",
    se = "standard"
  )
  out <- do.call(br("fit_sem"), args)
  est <- as.numeric(out$coefficients)
  if (!is.null(names(out$coefficients))) {
    names(est) <- names(out$coefficients)
  } else {
    names(est) <- names(truth)[seq_along(est)]
  }
  est
}

fit_magmaan <- function(model, data, method, bounds) {
  spec <- magmaan::model_spec(model, meanstructure = FALSE)
  d <- magmaan::df_to_data(data, spec)
  b <- NULL
  if (!identical(tolower(bounds), "none")) {
    if (!identical(tolower(bounds), "standard")) {
      stop("magmaan oracle check currently supports bounds = 'standard' or 'none'", call. = FALSE)
    }
    b <- magmaan::magmaan_core$estimate_bounds_standard(spec, d)
  }
  fit <- magmaan::magmaan_core$fit_ml(spec, d, bounds = b)
  if (identical(method, "ML")) return(free_estimates(fit))
  rb <- magmaan::magmaan_core$frontier_rbm(
    fit,
    raw_data = d,
    method = switch(method, eRBM = "explicit", iRBM = "implicit"),
    bounds = b
  )
  free_estimates(rb)
}

data <- br("gen_data_twofac")(
  n = opts$n,
  rel = opts$rel,
  dist = "Normal",
  lavsim = FALSE,
  scale = 1,
  seed = opts$seed
)
model <- br("txt_mod_twofac")(opts$rel)
truth <- br("truth")(data)

methods <- c("ML", "eRBM", "iRBM")
rows <- list()
k <- 0L
for (method in methods) {
  br_est <- fit_brlavaan(model, data, truth, method, opts$bounds)
  mg_est <- fit_magmaan(model, data, method, opts$bounds)
  common <- intersect(names(br_est), names(mg_est))
  for (param in common) {
    k <- k + 1L
    rows[[k]] <- data.frame(
      model = "twofac",
      dist = "Normal",
      reliability = opts$rel,
      n = opts$n,
      seed = opts$seed,
      bounds = opts$bounds,
      method = method,
      param = param,
      brlavaan = br_est[[param]],
      magmaan = mg_est[[param]],
      diff = mg_est[[param]] - br_est[[param]],
      abs_diff = abs(mg_est[[param]] - br_est[[param]]),
      stringsAsFactors = FALSE
    )
  }
}

out <- do.call(rbind, rows)
golden_path <- file.path(opts$results_dir, "brlavaan_magmaan_goldens.csv")
write.csv(out, golden_path, row.names = FALSE)

metadata <- data.frame(
  run_time = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
  brlavaan_version = as.character(utils::packageVersion("brlavaan")),
  magmaan_version = as.character(utils::packageVersion("magmaan")),
  n = opts$n,
  rel = opts$rel,
  seed = opts$seed,
  bounds = opts$bounds,
  max_abs_diff = max(out$abs_diff, na.rm = TRUE),
  stringsAsFactors = FALSE
)
metadata_path <- file.path(opts$results_dir, "brlavaan_magmaan_goldens_metadata.csv")
write.csv(metadata, metadata_path, row.names = FALSE)

cat("Wrote:\n")
cat("  ", golden_path, "\n", sep = "")
cat("  ", metadata_path, "\n", sep = "")
cat("Max absolute difference: ", format(metadata$max_abs_diff, digits = 6), "\n", sep = "")
