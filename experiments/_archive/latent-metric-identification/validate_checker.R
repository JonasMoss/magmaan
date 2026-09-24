#!/usr/bin/env Rscript

# Validate the structural admissibility checker against the empirical
# same_implied_moments signal from results/pairs.csv. Does NOT refit anything;
# rebuilds marker specs only to run the checker.

.support_helpers <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg)) {
    script <- normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE)
  } else {
    ofile <- tryCatch(sys.frames()[[1L]]$ofile, error = function(e) NULL)
    script <- normalizePath(if (is.null(ofile)) "validate_checker.R" else ofile,
                            mustWork = FALSE)
  }
  file.path(dirname(dirname(script)), "_support", "R", "helpers.R")
}
source(.support_helpers())
rm(.support_helpers)

experiment_dir_path <- experiment_dir()
repo_dir <- repo_root()
fixtures_dir <- file.path(repo_dir, "tests", "fixtures")
results_dir_path <- results_dir()
pairs_path <- file.path(results_dir_path, "pairs.csv")

if (!file.exists(pairs_path))
  stop("Run run_experiment.R first; missing ", pairs_path, call. = FALSE)

# Reuse case-loading + spec-build helpers from run_experiment.R without
# triggering the runner's main loop. We strip out the entry-point block.
# The admissibility checker now lives in the magmaan package
# (magmaan::magmaan_core$frontier_is_std_lv_admissible); the local
# admissibility.R kept for historical reference.
is_std_lv_admissible <- function(marker_spec, std_lv_spec = NULL) {
  magmaan::magmaan_core$frontier_is_std_lv_admissible(marker_spec, std_lv_spec)
}

# Pull only the helper definitions out of run_experiment.R (everything before
# `args <- parse_args(...)`).
runner_src <- readLines(file.path(experiment_dir_path, "run_experiment.R"))
entry_line <- grep("^args <- parse_args", runner_src)[[1L]]
helpers_env <- new.env(parent = globalenv())
eval(parse(text = runner_src[seq_len(entry_line - 1L)]), envir = helpers_env)

require_pkg("jsonlite")
require_pkg("magmaan")

pairs <- utils::read.csv(pairs_path, check.names = FALSE)
cases <- helpers_env$load_experiment_cases(fixtures_dir)

cat(sprintf("validating checker over %d cases against %d pair rows\n",
            length(cases), nrow(pairs)))

verdict_rows <- list()
for (case in cases) {
  spec <- tryCatch(helpers_env$build_spec(case, "marker"), error = identity)
  if (inherits(spec, "error")) {
    verdict_rows[[length(verdict_rows) + 1L]] <- data.frame(
      case_id = case$id,
      checker_admissible = NA,
      checker_reason = paste("spec build failed:", conditionMessage(spec)),
      stringsAsFactors = FALSE
    )
    next
  }
  std_spec <- tryCatch(helpers_env$build_spec(case, "std_lv"), error = identity)
  if (inherits(std_spec, "error")) std_spec <- NULL
  v <- is_std_lv_admissible(spec, std_spec)
  verdict_rows[[length(verdict_rows) + 1L]] <- data.frame(
    case_id = case$id,
    checker_admissible = isTRUE(v$admissible),
    checker_reason = v$reason,
    stringsAsFactors = FALSE
  )
}
verdicts <- do.call(rbind, verdict_rows)

# Empirical per-case: take pairs with status=="OK" and reduce to "all same".
ok_pairs <- pairs[pairs$status == "OK" &
                   is.finite(pairs$elapsed_ratio_std_lv_over_marker), ]
emp <- if (nrow(ok_pairs)) {
  split_emp <- split(ok_pairs$same_implied_moments, ok_pairs$case_id)
  data.frame(
    case_id = names(split_emp),
    empirical_n_ok = vapply(split_emp, length, integer(1)),
    empirical_n_same = vapply(split_emp,
                              function(x) sum(x %in% TRUE),
                              integer(1)),
    stringsAsFactors = FALSE
  )
} else data.frame(case_id = character(), empirical_n_ok = integer(),
                  empirical_n_same = integer())
emp$empirical_admissible <- emp$empirical_n_same == emp$empirical_n_ok &
                             emp$empirical_n_ok > 0
emp$empirical_any_same <- emp$empirical_n_same > 0

joined <- merge(verdicts, emp, by = "case_id", all = TRUE)
# Skip cases that never produced OK pairs at all (npar mismatch / errors)
classifiable <- joined[!is.na(joined$empirical_admissible) &
                        !is.na(joined$checker_admissible), ]

cat("\n=== confusion matrix (rows: checker, cols: empirical) ===\n")
cm <- table(
  checker = ifelse(classifiable$checker_admissible, "ADMIT", "REJECT"),
  empirical = ifelse(classifiable$empirical_admissible, "SAME", "DIFFER")
)
print(cm)

fp <- classifiable[classifiable$checker_admissible &
                    !classifiable$empirical_admissible, ]
fn <- classifiable[!classifiable$checker_admissible &
                    classifiable$empirical_admissible, ]
tp <- classifiable[classifiable$checker_admissible &
                    classifiable$empirical_admissible, ]
tn <- classifiable[!classifiable$checker_admissible &
                    !classifiable$empirical_admissible, ]

cat(sprintf("\nTP (admit + same):       %d\n", nrow(tp)))
cat(sprintf("TN (reject + differ):    %d\n", nrow(tn)))
cat(sprintf("FP (admit + DIFFER!):    %d  <-- must be 0\n", nrow(fp)))
cat(sprintf("FN (reject + would-have-been-fine): %d\n", nrow(fn)))

if (nrow(fp)) {
  cat("\n=== FALSE POSITIVES — checker said safe but fits disagreed ===\n")
  print(fp[, c("case_id", "checker_reason", "empirical_n_ok",
               "empirical_n_same")], row.names = FALSE)
}
if (nrow(fn)) {
  cat("\n=== false negatives — checker rejected but fits agreed ===\n")
  print(fn[, c("case_id", "checker_reason", "empirical_n_ok")],
        row.names = FALSE)
}

reason_brief <- function(r) {
  if (!nzchar(r)) return("")
  r <- sub("^([^:]*: )?", "", r)
  r <- sub("\\(.*$", "", r)
  trimws(r)
}
cat("\n=== rejection reasons across TN cases ===\n")
if (nrow(tn)) {
  tab <- table(vapply(tn$checker_reason, reason_brief, character(1)))
  for (nm in names(tab)) cat(sprintf("  %3d  %s\n", tab[[nm]], nm))
}

skipped <- joined[is.na(joined$empirical_admissible), ]
if (nrow(skipped)) {
  cat(sprintf("\n=== %d case(s) had no OK pairs to classify (npar SKIP, fit errors) ===\n",
              nrow(skipped)))
  skipped_admit <- skipped[!is.na(skipped$checker_admissible) &
                            skipped$checker_admissible, ]
  if (nrow(skipped_admit)) {
    cat("  of those, checker would have ADMITTED (worth investigating):\n")
    print(skipped_admit[, c("case_id", "checker_reason")], row.names = FALSE)
  } else {
    cat("  all of those are also rejected by the checker (consistent).\n")
  }
}

invisible(NULL)
