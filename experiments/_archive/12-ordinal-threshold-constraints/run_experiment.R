#!/usr/bin/env Rscript
# Ordinal threshold-constraint experiment.
#
# Usage:
#   Rscript experiments/_archive/12-ordinal-threshold-constraints/run_experiment.R [options]

.support_helpers <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg)) {
    script <- normalizePath(sub("^--file=", "", file_arg[[1L]]),
                            mustWork = TRUE)
  } else {
    ofile <- tryCatch(sys.frames()[[1L]]$ofile, error = function(e) NULL)
    script <- normalizePath(if (is.null(ofile)) "run_experiment.R" else ofile,
                            mustWork = FALSE)
  }
  file.path(dirname(dirname(script)), "_support", "R", "helpers.R")
}
source(.support_helpers())
rm(.support_helpers)

usage <- function() {
  cat(
    "Usage: Rscript run_experiment.R [options]\n\n",
    "Options:\n",
    "  --preset NAME     CMake preset to use (default: opt)\n",
    "  --reps N          repeated timings per fit row (default: 3)\n",
    "  --max-iter N      optimizer max_iter (default: 300)\n",
    "  --seed-base N     deterministic seed base (default: 20260601)\n",
    "  --smoke           run the small validation design\n",
    "  --skip-build      use an already-built benchmark executable\n",
    "  --help            print this help\n",
    sep = ""
  )
}

parse_args <- function(args) {
  out <- list(
    preset = "opt",
    reps = 3L,
    max_iter = 300L,
    seed_base = 20260601L,
    smoke = FALSE,
    skip_build = FALSE
  )
  i <- 1L
  while (i <= length(args)) {
    arg <- args[[i]]
    if (arg %in% c("-h", "--help")) {
      usage()
      quit(save = "no", status = 0L)
    } else if (arg == "--preset") {
      i <- i + 1L
      out$preset <- args[[i]]
    } else if (startsWith(arg, "--preset=")) {
      out$preset <- sub("^--preset=", "", arg)
    } else if (arg == "--reps") {
      i <- i + 1L
      out$reps <- as.integer(args[[i]])
    } else if (startsWith(arg, "--reps=")) {
      out$reps <- as.integer(sub("^--reps=", "", arg))
    } else if (arg == "--max-iter") {
      i <- i + 1L
      out$max_iter <- as.integer(args[[i]])
    } else if (startsWith(arg, "--max-iter=")) {
      out$max_iter <- as.integer(sub("^--max-iter=", "", arg))
    } else if (arg == "--seed-base") {
      i <- i + 1L
      out$seed_base <- as.integer(args[[i]])
    } else if (startsWith(arg, "--seed-base=")) {
      out$seed_base <- as.integer(sub("^--seed-base=", "", arg))
    } else if (arg == "--smoke") {
      out$smoke <- TRUE
    } else if (arg == "--skip-build") {
      out$skip_build <- TRUE
    } else {
      stop("unknown argument: ", arg, call. = FALSE)
    }
    i <- i + 1L
  }

  if (!nzchar(out$preset)) stop("--preset must not be empty", call. = FALSE)
  for (name in c("reps", "max_iter", "seed_base")) {
    if (!is.finite(out[[name]]) || out[[name]] < 1L) {
      stop("--", gsub("_", "-", name), " must be a positive integer",
           call. = FALSE)
    }
    out[[name]] <- as.integer(out[[name]])
  }
  out
}

run_cmd <- function(command, args, root) {
  message("$ ", paste(c(command, args), collapse = " "))
  old <- setwd(root)
  on.exit(setwd(old), add = TRUE)
  status <- system2(command, args)
  if (!identical(status, 0L)) {
    stop("command failed with status ", status, ": ",
         paste(c(command, args), collapse = " "), call. = FALSE)
  }
  invisible(TRUE)
}

git_head <- function(root) {
  old <- setwd(root)
  on.exit(setwd(old), add = TRUE)
  out <- tryCatch(system2("git", c("rev-parse", "HEAD"), stdout = TRUE,
                         stderr = FALSE),
                  error = function(e) "")
  if (!length(out)) "" else out[[1L]]
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
project_root <- repo_root()
results <- ensure_results_dir()
set_single_threaded_math()

binary <- file.path(project_root, "build", args$preset, "benchmarks",
                    "magmaan_ordinal_threshold_constraint_bench")
constraints_csv <- file.path(results, "threshold_constraints.csv")
metadata_csv <- file.path(results, "metadata.csv")

if (!isTRUE(args$skip_build)) {
  run_cmd("cmake", c("--preset", args$preset, "-DMAGMAAN_BUILD_BENCH=ON"),
          project_root)
  run_cmd("cmake", c("--build", "--preset", args$preset,
                     "--target", "magmaan_ordinal_threshold_constraint_bench"),
          project_root)
}

if (!file.exists(binary)) {
  stop("benchmark executable not found: ", binary,
       "\nRun without --skip-build or choose a configured --preset.",
       call. = FALSE)
}

bench_args <- c(
  "--out", constraints_csv,
  "--reps", as.character(args$reps),
  "--max-iter", as.character(args$max_iter),
  "--seed-base", as.character(args$seed_base)
)
if (isTRUE(args$smoke)) bench_args <- c(bench_args, "--smoke")

run_cmd(binary, bench_args, project_root)

write_metadata(
  metadata_csv,
  values = list(
    experiment = "12-ordinal-threshold-constraints",
    question = paste(
      "Probe ordinal threshold equality constraints under full bounded,",
      "threshold-profiled bounded, full-threshold SNLLS, and",
      "threshold-profiled SNLLS fits in a single-group ordinal CFA."
    ),
    preset = args$preset,
    smoke = args$smoke,
    reps = args$reps,
    max_iter = args$max_iter,
    seed_base = args$seed_base,
    git_head = git_head(project_root),
    benchmark_binary = binary,
    threshold_constraints_csv = constraints_csv
  )
)

cat("Wrote:\n")
cat("  ", constraints_csv, "\n", sep = "")
cat("  ", metadata_csv, "\n", sep = "")
