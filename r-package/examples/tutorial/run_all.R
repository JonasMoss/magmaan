## Run every lavaan-tutorial parity script in order, reporting pass/fail.
##     Rscript r-package/examples/tutorial/run_all.R

here <- dirname(sub("^--file=", "",
                    grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
if (is.na(here) || !nzchar(here)) here <- "r-package/examples/tutorial"

scripts <- sort(list.files(here, pattern = "^[0-9].*\\.R$", full.names = TRUE))
fail <- character()
for (s in scripts) {
  cat("\n========== ", basename(s), " ==========\n", sep = "")
  ok <- tryCatch({ source(s, local = new.env()); TRUE },
                 error = function(e) { cat("ERROR: ", conditionMessage(e), "\n"); FALSE })
  if (!ok) fail <- c(fail, basename(s))
}

cat("\n=================================================\n")
if (length(fail)) {
  cat("FAILED:", paste(fail, collapse = ", "), "\n")
  quit(status = 1L)
}
cat("all", length(scripts), "tutorial parity scripts passed\n")
