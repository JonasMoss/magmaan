#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  index <- match(flag, args)
  if (is.na(index) || index == length(args)) default else args[[index + 1L]]
}
out_dir <- get_arg("--out-dir")
profile <- get_arg("--profile")
expected_cells <- as.integer(get_arg("--expected-cells", NA_character_))
if (is.null(out_dir) || is.null(profile) ||
    !profile %in% c("focus", "stress")) {
  stop("provide --out-dir and --profile focus|stress", call. = FALSE)
}

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- dirname(normalizePath(sub("^--file=", "", script_arg[[1L]])))
exp_path <- normalizePath(file.path(script_dir, ".."))
source(file.path(exp_path, "..", "_support", "R", "helpers.R"))
source(file.path(exp_path, "R", "sem_summaries.R"))

slices <- file.path(out_dir, "slices")
bind_file <- function(name) {
  files <- list.files(
    slices, pattern = paste0("^", name, "\\.csv$"),
    recursive = TRUE, full.names = TRUE)
  if (!length(files)) return(NULL)
  do.call(rbind, lapply(files, function(path) {
    read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  }))
}

design <- bind_file("design")
replications <- bind_file("replications")
if (is.null(design) || is.null(replications)) {
  stop("completed slices must contain design.csv and replications.csv",
       call. = FALSE)
}
design <- unique(design)
design <- design[order(design$cell_id, design$estimator), ]
replications <- replications[order(
  replications$cell_id, replications$rep, replications$estimator), ]
if (anyDuplicated(design[c("cell_id", "estimator")]) ||
    anyDuplicated(replications[c("cell_id", "estimator", "rep")])) {
  stop("overlapping Modal slices detected", call. = FALSE)
}
if (is.finite(expected_cells) && length(unique(design$cell_id)) != expected_cells) {
  stop("combined ", length(unique(design$cell_id)), " cells; expected ",
       expected_cells, call. = FALSE)
}
counts <- table(replications$cell_id)
if (length(unique(as.integer(counts))) != 1L) {
  stop("Modal slices have unequal replication counts", call. = FALSE)
}

write_csv(design, file.path(out_dir, "design.csv"))
write_csv(replications, file.path(out_dir, "replications.csv"))
sem_write_method_summaries(replications, out_dir)
for (name in c("generator_calibration", "timing_summary")) {
  value <- bind_file(name)
  if (!is.null(value)) write_csv(value, file.path(out_dir, paste0(name, ".csv")))
}
write_metadata(file.path(out_dir, "metadata.csv"), list(
  profile = profile,
  cells = length(unique(design$cell_id)),
  estimator_rows = nrow(replications),
  reps_per_cell = unique(as.integer(counts)),
  estimators = unique(replications$estimator),
  truth = unique(replications$truth),
  regions = unique(replications$analysis_region),
  alpha_convention = "p <= 0.05; p < 0.05 retained as sensitivity",
  modal_slices = length(unique(dirname(list.files(
    slices, pattern = "^design\\.csv$", recursive = TRUE,
    full.names = TRUE))))))
cat(sprintf("combined %d cells and %d estimator rows into %s\n",
            length(unique(design$cell_id)), nrow(replications), out_dir))
