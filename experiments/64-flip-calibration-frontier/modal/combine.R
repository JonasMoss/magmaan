#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  i <- match(flag, args)
  if (is.na(i) || i == length(args)) default else args[[i + 1L]]
}
out_dir <- get_arg("--out-dir", ".")
profile <- get_arg("--profile")
expected_cells <- as.integer(get_arg("--expected-cells", NA_character_))
if (is.null(profile) || !profile %in% c("screen", "focus", "smoke"))
  stop("--profile must be screen, focus, or smoke", call. = FALSE)
slices <- file.path(out_dir, "slices")
if (!dir.exists(slices)) stop("no slices/ under ", out_dir, call. = FALSE)

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- dirname(normalizePath(sub("^--file=", "", script_arg[[1L]])))
exp_dir <- normalizePath(file.path(script_dir, ".."))
support <- normalizePath(file.path(exp_dir, "..", "_support", "R", "helpers.R"))
source(support)
source(file.path(exp_dir, "R", "engine.R"))
source(file.path(exp_dir, "R", "summaries.R"))

bind_file <- function(name) {
  files <- list.files(slices, pattern = paste0("^", name, "\\.csv$"),
                      recursive = TRUE, full.names = TRUE)
  if (!length(files)) return(NULL)
  parts <- lapply(files, function(path)
    read.csv(path, stringsAsFactors = FALSE, check.names = FALSE))
  parts <- parts[vapply(parts, ncol, integer(1)) > 0L]
  if (!length(parts)) return(NULL)
  do.call(rbind, parts)
}

design <- bind_file("design")
pvalues <- bind_file("pvalues")
if (is.null(design) || is.null(pvalues))
  stop("each completed slice must provide design.csv and pvalues.csv", call. = FALSE)
design <- unique(design)
design <- design[order(design$cell_id), ]
pvalues <- pvalues[order(pvalues$cell_id, pvalues$rep), ]
if (anyDuplicated(design$cell_id) || anyDuplicated(pvalues[c("cell_id", "rep")]))
  stop("overlapping Modal slices detected", call. = FALSE)
if (is.finite(expected_cells) && nrow(design) != expected_cells)
  stop("combined ", nrow(design), " cells; expected ", expected_cells,
       call. = FALSE)
if (!setequal(unique(pvalues$cell_id), design$cell_id))
  stop("pvalue cells do not match combined design", call. = FALSE)

write_csv(design, file.path(out_dir, "design.csv"))
write_csv(pvalues, file.path(out_dir, "pvalues.csv"))
method_rows <- frontier_method_rows(pvalues)
method_summary <- frontier_summarize_methods(method_rows)
write_csv(method_summary, file.path(out_dir, "method_summary.csv"))
write_csv(frontier_availability_summary(method_rows),
          file.path(out_dir, "availability_summary.csv"))
write_csv(frontier_calibration_summary(method_summary),
          file.path(out_dir, "calibration_summary.csv"))
write_csv(frontier_summarize_matched_power(method_rows),
          file.path(out_dir, "matched_power.csv"))

for (name in c("paired_summary", "timing_summary", "failures")) {
  value <- bind_file(name)
  if (!is.null(value)) write_csv(value, file.path(out_dir, paste0(name, ".csv")))
}
methods <- bind_file("methods")
if (!is.null(methods)) write_csv(unique(methods), file.path(out_dir, "methods.csv"))

reps <- table(pvalues$cell_id)
if (length(unique(as.integer(reps))) != 1L)
  stop("Modal slices have unequal replication counts", call. = FALSE)
write_metadata(file.path(out_dir, "metadata.csv"), list(
  profile = profile, cells = nrow(design), replications = nrow(pvalues),
  reps_min = min(reps), reps_max = max(reps),
  modal_slices = length(unique(dirname(list.files(
    slices, pattern = "^design\\.csv$", recursive = TRUE, full.names = TRUE)))),
  alpha_convention = "p <= 0.05", strict_sensitivity = "p < 0.05"))
cat(sprintf("combined %d cells and %d replications into %s\n",
            nrow(design), nrow(pvalues), out_dir))
