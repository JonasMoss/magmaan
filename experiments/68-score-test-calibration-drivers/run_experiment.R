#!/usr/bin/env Rscript

# What governs the calibration of the mean-scaled (Satorra-Bentler) non-normal
# score test: degrees of freedom, structural information geometry, or the
# dispersion of the test's eigenvalue spectrum? Two small studies:
#   df        - vary df = q_tested * (G-1) with clean homogeneous Phi=Theta=1.
#   structure - fix df=28, vary Phi/Theta concentration (homogeneous/geometry/
#               concentrated) crossed with normal vs Vale-Maurelli kurtosis.

.support_helpers <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  script <- if (length(file_arg))
    normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE)
  else normalizePath("run_experiment.R", mustWork = FALSE)
  file.path(dirname(dirname(script)), "_support", "R", "helpers.R")
}
source(.support_helpers()); rm(.support_helpers)
suppressWarnings(suppressMessages(library(magmaan)))
set_single_threaded_math()
source(experiment_path("R", "engine.R"))

usage <- function() cat(
  "Usage: Rscript run_experiment.R [--full] [options]\n\n",
  "Score-test calibration drivers (df vs structure vs eigenvalue spectrum).\n\n",
  "Profiles:\n",
  "  (default)   smoke: reps=15, flips=49, n_avg capped at 300.\n",
  "  --full      reps=300, flips=199, df study n_avg=500, structure n_avg=800.\n\n",
  "Overrides:\n",
  "  --reps N --flips N --cores N --seed-base N --results-dir P\n",
  "  --study df|structure|both   Restrict to one sub-study (default both).\n",
  "  --max-cells N               Run only the first N cells (smoke aid).\n", sep = "")

parse_args <- function(args) {
  out <- list(full = FALSE, reps = NULL, flips = NULL,
              cores = max(1L, parallel::detectCores() - 1L),
              seed_base = 20260714L, results_dir = NULL, study = "both",
              max_cells = NULL)
  i <- 1L
  take <- function() { i <<- i + 1L
    if (i > length(args)) stop("missing value after ", args[[i - 1L]], call. = FALSE); args[[i]] }
  while (i <= length(args)) {
    a <- args[[i]]
    if (a %in% c("-h", "--help")) { usage(); quit(save = "no", status = 0L) }
    else if (a == "--full") out$full <- TRUE
    else if (a == "--reps") out$reps <- as.integer(take())
    else if (a == "--flips") out$flips <- as.integer(take())
    else if (a == "--cores") out$cores <- as.integer(take())
    else if (a == "--seed-base") out$seed_base <- as.integer(take())
    else if (a == "--results-dir") out$results_dir <- take()
    else if (a == "--study") out$study <- match.arg(take(), c("df", "structure", "both"))
    else if (a == "--max-cells") out$max_cells <- as.integer(take())
    else stop("unknown argument: ", a, call. = FALSE)
    i <- i + 1L
  }
  out
}

# ---- design: a flat list of cells with a stable global id ----------------------
build_design <- function(full) {
  cell <- function(study, structure, G, q, p, dist, n_avg)
    list(study = study, structure = structure, G = G, q = q, p = p,
         dist = dist, n_avg = n_avg, df_expected = q * (G - 1))
  df_navg <- if (full) 500L else 300L
  st_navg <- if (full) 800L else 300L
  cells <- list()
  # df study (p=8, homogeneous Phi=Theta=1). Low df via G=2 varying q; high df via G=8.
  for (q in 1:7) cells <- c(cells, list(cell("df", "homogeneous", 2L, q, 8L, "vm", df_navg)))
  for (q in c(1L, 2L, 4L)) cells <- c(cells, list(cell("df", "homogeneous", 8L, q, 8L, "vm", df_navg)))
  cells <- c(cells, list(cell("df", "homogeneous", 2L, 1L, 8L, "normal", df_navg)))  # control df=1
  cells <- c(cells, list(cell("df", "homogeneous", 8L, 4L, 8L, "normal", df_navg)))  # control df=28
  # structure study (p=20, df=28). Vary information geometry x kurtosis.
  cells <- c(cells, list(cell("structure", "homogeneous",  8L, 4L, 20L, "vm", st_navg)))
  cells <- c(cells, list(cell("structure", "geometry",     8L, 4L, 20L, "vm", st_navg)))
  cells <- c(cells, list(cell("structure", "concentrated", 8L, 4L, 20L, "vm", st_navg)))
  cells <- c(cells, list(cell("structure", "concentrated", 8L, 4L, 20L, "normal", st_navg)))  # control
  for (k in seq_along(cells)) cells[[k]]$id <- k
  cells
}

run_cell <- function(cell, reps, flips, cores, seed_base, n_cap) {
  n_avg <- if (is.na(n_cap)) cell$n_avg else min(cell$n_avg, n_cap)
  pop <- sbd_structure(cell$structure, cell$G, cell$p)
  sampler <- sbd_calibrate_sampler(pop$Sigma, cell$dist, rep(3, cell$p), rep(21, cell$p))
  specs <- sbd_specs(cell$G, cell$p, cell$q)
  gs <- rep(as.integer(n_avg), cell$G)
  base <- as.integer(seed_base + cell$id * 100000L)          # cell-stable, id-keyed
  rows <- parallel::mclapply(seq_len(reps), function(r)
    sbd_one_rep(specs, sampler, gs, base + r, base + 500000L + r, flips),
    mc.cores = cores, mc.preschedule = TRUE)
  df <- do.call(rbind, Filter(is.data.frame, rows)); n <- nrow(df)
  rate <- function(m) if (n) mean(df[[m]] <= .05, na.rm = TRUE) else NA_real_
  summ <- data.frame(
    id = cell$id, study = cell$study, structure = cell$structure,
    G = cell$G, q_tested = cell$q, p = cell$p, distribution = cell$dist,
    n_avg = n_avg, df_expected = cell$df_expected,
    df_obs = if (n) as.integer(stats::median(df$df_obs)) else NA_integer_,
    valid = n, reps = reps, min_pd = round(pop$min_pd, 4),
    eigen_cv = if (n) round(stats::median(df$eigen_cv), 4) else NA_real_,
    eigen_ratio = if (n) round(stats::median(df$eigen_ratio), 1) else NA_real_,
    stringsAsFactors = FALSE)
  for (m in sbd_methods) summ[[m]] <- round(rate(m), 4)
  summ
}

# ---- run ------------------------------------------------------------------------
opt <- parse_args(commandArgs(trailingOnly = TRUE))
reps  <- if (!is.null(opt$reps))  opt$reps  else if (opt$full) 300L else 15L
flips <- if (!is.null(opt$flips)) opt$flips else if (opt$full) 199L else 49L
n_cap <- if (opt$full) NA_integer_ else 300L
results <- if (!is.null(opt$results_dir)) opt$results_dir else ensure_results_dir()
dir.create(results, showWarnings = FALSE, recursive = TRUE)

cells <- build_design(opt$full)
if (opt$study != "both") cells <- Filter(function(c) c$study == opt$study, cells)
if (!is.null(opt$max_cells)) cells <- cells[seq_len(min(opt$max_cells, length(cells)))]

cat(sprintf("cells=%d reps=%d flips=%d cores=%d n_cap=%s\n",
            length(cells), reps, flips, opt$cores, ifelse(is.na(n_cap), "none", n_cap)))
t0 <- proc.time()[["elapsed"]]
summaries <- lapply(seq_along(cells), function(k) {
  cell <- cells[[k]]
  s <- run_cell(cell, reps, flips, opt$cores, opt$seed_base, n_cap)
  cat(sprintf("[%d/%d] %-9s %-12s G=%d q=%d %-6s df=%2d valid=%3d CV=%.2f | SB=%.3f pEBA4=%.3f flipEff=%.3f chisq=%.3f\n",
      k, length(cells), s$study, s$structure, s$G, s$q_tested, s$distribution,
      s$df_obs, s$valid, s$eigen_cv, s$score_sb, s$score_peba4, s$flip_effective, s$score_chisq))
  s
})
allsum <- do.call(rbind, summaries)

df_rows <- allsum[allsum$study == "df", , drop = FALSE]
st_rows <- allsum[allsum$study == "structure", , drop = FALSE]
if (nrow(df_rows)) write_csv(df_rows[order(df_rows$df_obs, df_rows$G), ], file.path(results, "df_sweep.csv"))
if (nrow(st_rows)) write_csv(st_rows, file.path(results, "structure.csv"))
write_csv(allsum, file.path(results, "all_cells.csv"))
write_metadata(file.path(results, "metadata.csv"), list(
  profile = if (opt$full) "full" else "smoke", cells = length(cells),
  reps = reps, flips = flips, seed_base = opt$seed_base,
  elapsed_seconds = round(proc.time()[["elapsed"]] - t0, 1),
  alpha = "p <= 0.05", note = "loadings equal across groups => equality null true"),
  packages = c("magmaan"))

cat(sprintf("\nwrote %s\n", results))
for (f in c("df_sweep.csv", "structure.csv", "all_cells.csv", "metadata.csv"))
  if (file.exists(file.path(results, f))) cat("  ", file.path(results, f), "\n")
