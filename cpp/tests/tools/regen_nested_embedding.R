#!/usr/bin/env Rscript
# Frozen component reference for inference-side parameter-key embedding.
# Run from the repository root. Public HS1939 rows are an allowlisted dataset.
suppressPackageStartupMessages({library(lavaan); library(jsonlite)})
fixtures <- "cpp/tests/fixtures"
pinned <- readLines(file.path(fixtures, "lavaan_version.txt"), n = 1L)
stopifnot(gsub("-", ".", pinned, fixed = TRUE) == as.character(packageVersion("lavaan")))
three <- paste("visual =~ x1 + x2 + x3", "textual =~ x4 + x5 + x6",
               "speed =~ x7 + x8 + x9", sep = "\n")
syn1 <- paste(three, "visual =~ x9", sep = "\n")
syn0 <- paste(three, "visual =~ a*x9\na == 0", sep = "\n")
d <- HolzingerSwineford1939
h1 <- cfa(syn1, d, estimator = "MLR")
h0 <- cfa(paste(three, "visual =~ 0*x9", sep = "\n"), d, estimator = "MLR")
score0 <- cfa(syn0, d)
lr <- lavTestLRT(h1, h0, method = "satorra.2000", A.method = "exact")
expected1 <- cfa(syn1, d, estimator = "MLR", information = "expected")
expected0 <- cfa(paste(three, "visual =~ 0*x9", sep = "\n"), d,
                 estimator = "MLR", information = "expected")
lr_expected <- lavTestLRT(expected1, expected0, method = "satorra.2000", A.method = "exact")
rows <- function(fit) {
  p <- parTable(fit)
  p <- p[p$op %in% c("=~", "~~", "~", "~1"), c("lhs", "op", "rhs", "group", "est")]
  lapply(seq_len(nrow(p)), function(i) as.list(p[i, ]))
}
out <- list(provenance = list(generator = "cpp/tests/tools/regen_nested_embedding.R",
  lavaan_version = as.character(packageVersion("lavaan")), data = "HolzingerSwineford1939",
  lr = "lavTestLRT MLR, satorra.2000, A.method=exact; H0 uses fixed-zero loading",
  score = "lavTestScore release=1; H0 uses affine equality a==0"),
  model_H1 = syn1, model_H0 = three,
  X = unname(as.matrix(d[paste0("x", 1:9)])),
  rows_H1 = rows(h1), rows_H0 = rows(score0),
  fmin_H1 = unname(fitMeasures(h1, "chisq"))/(2*nrow(d)),
  fmin_H0 = unname(fitMeasures(score0, "chisq"))/(2*nrow(d)),
  lr = unname(fitMeasures(score0, "chisq")-fitMeasures(h1, "chisq")),
  score = unname(lavTestScore(score0, release = 1)$test$X2),
  lr_scaled_lavaan = lr[["Chisq diff"]][2L],
  lr_scaled_expected = lr_expected[["Chisq diff"]][2L])
write_json(out, file.path(fixtures, "nested_embedding.json"), auto_unbox = TRUE,
           pretty = TRUE, digits = 17)
