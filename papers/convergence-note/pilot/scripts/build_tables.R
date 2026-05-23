#!/usr/bin/env Rscript

args <- commandArgs(FALSE)
file_arg <- sub("^--file=", "", grep("^--file=", args, value = TRUE)[1L])
script_dir <- if (!is.na(file_arg)) {
  dirname(normalizePath(file_arg, mustWork = TRUE))
} else {
  file.path("papers", "convergence-note", "pilot", "scripts")
}
source(file.path(script_dir, "pilot-lib.R"))

pilot_ensure_dirs()

summarise_results <- function(x, group_cols) {
  if (!nrow(x)) return(data.frame())
  key <- interaction(x[group_cols], drop = TRUE, lex.order = TRUE)
  groups <- split(seq_len(nrow(x)), key)
  rows <- lapply(groups, function(idx) {
    g <- x[idx, , drop = FALSE]
    first <- g[1L, group_cols, drop = FALSE]
    data.frame(
      first,
      n_cells = nrow(g),
      converged = sum(g$converged %in% TRUE, na.rm = TRUE),
      convergence_rate = mean(g$converged %in% TRUE, na.rm = TRUE),
      error_rate = mean(!(g$ok %in% TRUE), na.rm = TRUE),
      median_elapsed_sec = stats::median(g$elapsed_sec, na.rm = TRUE),
      median_iterations = stats::median(g$iterations, na.rm = TRUE),
      median_fmin_abs_diff_lavaan =
        if ("fmin_abs_diff_lavaan" %in% names(g)) {
          stats::median(g$fmin_abs_diff_lavaan, na.rm = TRUE)
        } else {
          NA_real_
        },
      top_error_kind = top_error_kind(g$error_kind),
      stringsAsFactors = FALSE
    )
  })
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out[do.call(order, out[group_cols]), , drop = FALSE]
}

top_error_kind <- function(x) {
  x <- x[!is.na(x) & nzchar(x)]
  if (!length(x)) return(NA_character_)
  tab <- sort(table(x), decreasing = TRUE)
  paste0(names(tab)[1L], " (", unname(tab[[1L]]), ")")
}

fmt_num <- function(x, digits = 3) {
  ifelse(is.na(x), "", formatC(x, digits = digits, format = "fg", flag = "#"))
}

tex_escape <- function(x) {
  x <- as.character(x)
  x <- gsub("\\\\", "\\\\textbackslash{}", x)
  x <- gsub("([_%&#$])", "\\\\\\1", x, perl = TRUE)
  x
}

write_tex_table <- function(x, path, caption) {
  if (!nrow(x)) return(invisible(NULL))
  y <- x
  for (nm in names(y)) {
    if (is.numeric(y[[nm]])) y[[nm]] <- fmt_num(y[[nm]])
    y[[nm]][is.na(y[[nm]])] <- ""
    y[[nm]] <- tex_escape(y[[nm]])
  }
  lines <- c(
    "\\begin{table}[ht]",
    "\\centering",
    paste0("\\caption{", tex_escape(caption), "}"),
    paste0("\\begin{tabular}{", paste(rep("l", ncol(y)), collapse = ""), "}"),
    "\\hline",
    paste(tex_escape(names(y)), collapse = " & "),
    "\\\\ \\hline"
  )
  body <- apply(y, 1L, function(row) paste(row, collapse = " & "))
  lines <- c(lines, paste0(body, " \\\\"), "\\hline", "\\end{tabular}", "\\end{table}")
  writeLines(lines, path)
  invisible(path)
}

read_if_exists <- function(path) {
  if (!file.exists(path)) return(data.frame())
  utils::read.csv(path, stringsAsFactors = FALSE)
}

sim_raw <- read_if_exists(pilot_path("results", "convergence_sims_raw.csv"))
textbook_raw <- read_if_exists(pilot_path("results", "textbook_subset_raw.csv"))

sim_summary <- summarise_results(sim_raw, c("design", "optimizer"))
textbook_summary <- summarise_results(textbook_raw, c("source", "optimizer"))

pilot_write_csv(sim_summary, pilot_path("tables", "convergence_sims_summary.csv"))
pilot_write_csv(textbook_summary, pilot_path("tables", "textbook_subset_summary.csv"))

write_tex_table(
  sim_summary,
  pilot_path("tables", "convergence_sims_summary.tex"),
  "Generated hard-case simulation pilot summary."
)
write_tex_table(
  textbook_summary,
  pilot_path("tables", "textbook_subset_summary.tex"),
  "Strict Little/Newsom textbook-corpus pilot summary."
)

inline_tex <- function(path, fallback) {
  if (!file.exists(path)) return(fallback)
  readLines(path, warn = FALSE)
}

report <- c(
  "\\documentclass[11pt]{article}",
  "\\usepackage[margin=1in]{geometry}",
  "\\usepackage[T1]{fontenc}",
  "\\usepackage{booktabs}",
  "\\title{magmaan convergence pilot}",
  "\\date{}",
  "\\begin{document}",
  "\\maketitle",
  "\\section*{Scope}",
  paste(
    "This pilot is deliberately separate from the convergence-note manuscript.",
    "It records boring baseline runs from the root magmaan R package, using",
    "the package's current default start path and varying optimizer backend."
  ),
  "\\section*{Outputs}",
  "\\begin{itemize}",
  "\\item \\texttt{results/convergence\\_sims\\_raw.csv}",
  "\\item \\texttt{results/textbook\\_subset\\_raw.csv}",
  "\\item \\texttt{tables/convergence\\_sims\\_summary.csv}",
  "\\item \\texttt{tables/textbook\\_subset\\_summary.csv}",
  "\\end{itemize}",
  "\\section*{Simulation Summary}",
  if (nrow(sim_summary)) {
    inline_tex(pilot_path("tables", "convergence_sims_summary.tex"),
               "No simulation CSV was found.")
  } else {
    "No simulation CSV was found."
  },
  "\\section*{Textbook Summary}",
  if (nrow(textbook_summary)) {
    inline_tex(pilot_path("tables", "textbook_subset_summary.tex"),
               "No textbook CSV was found.")
  } else {
    "No textbook CSV was found."
  },
  "\\section*{Reading Notes}",
  paste(
    "The \\texttt{start\\_policy} column is currently",
    "\\texttt{magmaan\\_r\\_default}. The R package does not yet expose",
    "individual deterministic start producers or a custom theta-zero fit path,",
    "so this pilot is an optimizer/backend baseline rather than a start-policy",
    "comparison. The table shape is ready for those columns once exposed."
  ),
  "\\end{document}"
)
writeLines(report, pilot_path("report", "pilot-report.tex"))
message("Wrote ", pilot_path("report", "pilot-report.tex"))
