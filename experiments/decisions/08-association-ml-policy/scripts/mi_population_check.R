#!/usr/bin/env Rscript
# Derived evidence from production frozen at b37ebefb; no fitting or simulation.
args <- commandArgs(TRUE)
if (length(args) != 1 || args[1] %in% c("-h", "--help")) {
  cat("Usage: Rscript scripts/mi_population_check.R RESULTS_DIR\n",
      "Reads cells.csv, population_targets.csv and raw.rds; writes mi_population_check.csv.\n")
  quit(status = if (length(args) == 1) 0 else 1)
}
dir <- args[1]
cells <- read.csv(file.path(dir, "cells.csv"))
pop <- read.csv(file.path(dir, "population_targets.csv"))
raw <- readRDS(file.path(dir, "raw.rds"))
mi_ids <- sort(unique(raw$cell_id[raw$arm == "association_mi_size" & raw$target == "mi"]))
cells <- cells[cells$groups == 1 & cells$cell_id %in% mi_ids, ]
stopifnot(nrow(cells) > 0, setequal(cells$cell_id, mi_ids),
          !anyDuplicated(cells$cell_id))
cells <- cells[order(cells$cell_id), ]
pop <- pop[pop$estimator == "ML" & pop$target == "loading", ]
stopifnot(!anyDuplicated(pop[c("key", "draw")]))
loading <- raw[raw$arm == "association_ij" & raw$target == "loading" & raw$success, ]
stopifnot(!anyDuplicated(loading[c("cell_id", "replicate")]))
result <- do.call(rbind, lapply(seq_len(nrow(cells)), function(i) {
  cell <- cells[i, ]
  key <- with(cell, paste(factors, categories, imbalance, generator, misspecified, sep = "_"))
  lambda1 <- pop$estimate[pop$key == key & pop$draw == 1]
  lambda2 <- pop$estimate[pop$key == key & pop$draw == 2]
  se <- loading$se[loading$cell_id == cell$cell_id]
  stopifnot(length(lambda1) == 1, length(lambda2) == 1,
            all(is.finite(c(lambda1, lambda2))), length(se) > 0,
            all(is.finite(se) & se > 0))
  Delta <- abs(lambda1 - lambda2)
  se_median <- median(se)
  ncp <- (Delta / se_median)^2
  implied_size <- pchisq(qchisq(.95, 1), df = 1, ncp = ncp, lower.tail = FALSE)
  data.frame(cell_id = cell$cell_id, key, n = cell$n, lambda1, lambda2,
             Delta, se_median, ncp, implied_size)
}))
path <- file.path(dir, "mi_population_check.csv")
write.csv(result, path, row.names = FALSE)
worst <- result[which.max(result$implied_size), ]
cat(sprintf("Wrote %s (%d cells). Maximum implied size %.6f%%, cell %d: %s. %s\n",
            path, nrow(result), 100 * worst$implied_size, worst$cell_id, worst$key,
            if (worst$implied_size <= .055) "PASS (<=5.5%)." else "FAIL (>5.5%)."))
