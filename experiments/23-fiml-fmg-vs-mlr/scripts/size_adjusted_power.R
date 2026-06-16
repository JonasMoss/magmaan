#!/usr/bin/env Rscript
# Size-adjusted (empirical-size-corrected) power from raw per-replicate p-values.
#
# Ordinary power rewards a test for over-rejecting: a statistic that rejects 27%
# of TRUE models will also reject more false ones, but that "power" is borrowed
# from a broken Type-I rate. Size-adjusted power removes the loan. For each
# (method, cell) we read the empirical alpha-quantile of the H0 p-values as the
# critical threshold c -- the cutoff that WOULD give exactly 5% under the true
# model in that cell -- then score power as P(p < c) under the alternative.
# A well-calibrated test has c ~ alpha and ordinary ~ size-adjusted; a liberal
# test has c < alpha and its size-adjusted power drops to a fair footing.
#
# Reads results/raw/{gof,nested}_fits.csv, writes results/power_adjusted.csv.

suppressPackageStartupMessages(library(data.table))
alpha <- 0.05
res_dir <- "results"
cols <- c("truth", "dist", "mech", "rate", "outcome", "method", "p_value")

G <- fread(file.path(res_dir, "raw/gof_fits.csv"), select = cols)
N <- fread(file.path(res_dir, "raw/nested_fits.csv"), select = cols)
DT <- rbind(G, N)
keys <- c("outcome", "method", "dist", "mech", "rate")

# Critical threshold: alpha-quantile of the H0 p-value distribution per cell.
h0 <- DT[truth == "h0",
         .(c_thresh = as.numeric(quantile(p_value, alpha, names = FALSE,
                                           type = 8, na.rm = TRUE))),
         by = keys]

# Power cells: gof outcome vs gof_power, nested outcome vs nested_power.
pw <- DT[(outcome == "gof"    & truth == "gof_power") |
         (outcome == "nested" & truth == "nested_power")]
pw <- merge(pw, h0, by = keys, all.x = TRUE)

out <- pw[, .(n_rep      = .N,
              power       = mean(p_value < alpha,    na.rm = TRUE),
              power_adj   = mean(p_value < c_thresh, na.rm = TRUE),
              c_thresh    = c_thresh[1]),
          by = keys]
setorderv(out, keys)
fwrite(out, file.path(res_dir, "power_adjusted.csv"))
cat("Wrote ", file.path(res_dir, "power_adjusted.csv"), " (",
    nrow(out), " cells)\n", sep = "")
