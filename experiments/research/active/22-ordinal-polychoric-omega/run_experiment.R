#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
root <- dirname(normalizePath(sub("^--file=", "", file_arg[[1L]])))
lane <- "coverage"
lane_arg <- match("--lane", args)
if (!is.na(lane_arg)) {
  if (lane_arg == length(args)) stop("--lane needs coverage or stress")
  lane <- args[[lane_arg + 1L]]
  args <- args[-c(lane_arg, lane_arg + 1L)]
}
if (!lane %in% c("coverage", "stress")) stop("Unknown lane: ", lane)
if ("--help" %in% args) {
  cat("Polychoric omega delta intervals: --lane coverage|stress (default coverage).\n",
      "Coverage uses the Gaussian latent-response target; stress uses a\n",
      "large-draw polychoric pseudo-target. Each lane keeps its own results.\n\n",
      sep = "")
}
if (!"--results-dir" %in% args) {
  args <- c(args, "--results-dir", file.path(root, "results", lane))
}
status <- system2(file.path(R.home("bin"), "Rscript"),
                  shQuote(c(file.path(root, "R", paste0(lane, ".R")), args)))
quit(save = "no", status = status)
