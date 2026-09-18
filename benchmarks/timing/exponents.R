#!/usr/bin/env Rscript
# Fit measured complexity exponents to magmaan_timing_bench output.
#
#   Rscript benchmarks/timing/exponents.R <p-ladder csv> [<p x n grid csv>] [out.json]
#
# The p-ladder (one n, several p) gives the exponent in p per stage:
#     log t = a + b * log p                 -> stage cost ~ p^b
# A crossed p x n grid gives both exponents jointly:
#     log t = a + b * log p + c * log n     -> stage cost ~ p^b * n^c
#
# R^2 is reported alongside every exponent and is not decoration: a stage that
# crosses a cache boundary inside the measured range is not a single power law,
# and its low R^2 is the signal to read the raw ladder instead of the exponent.
#
# Input must come from an --isolate run. Exponents are within-stage so a rotated
# run would not bias them, but the absolute values printed beside them would be
# composition-dependent (see timing/README.md).

args <- commandArgs(trailingOnly = TRUE)
if (!length(args)) stop("usage: exponents.R <p-ladder csv> [<pxn grid csv>] [out.json]")

pladder_path <- args[1]
grid_path    <- if (length(args) >= 2 && nzchar(args[2])) args[2] else NA_character_
out_json     <- if (length(args) >= 3) args[3] else NA_character_

read_bench <- function(path) {
  d <- utils::read.csv(path, stringsAsFactors = FALSE)
  d$us <- d$median_ns / 1000
  d
}

# Derived rows carry no batch/rep information and are sums, not measurements.
DERIVED <- c("staged_fit_ml", "pipeline_total")

fit_p <- function(d, model) {
  s <- d[d$model == model & d$parameterization == "marker" &
           !(d$stage %in% DERIVED), ]
  if (!nrow(s)) return(NULL)
  s <- s[s$n == min(s$n), ]
  out <- lapply(split(s, s$stage), function(g) {
    g <- g[order(g$p), ]
    ok <- is.finite(g$us) & g$us > 0
    if (sum(ok) < 3L) return(NULL)
    m <- stats::lm(log(g$us[ok]) ~ log(g$p[ok]))
    data.frame(stage = g$stage[1],
               b_p   = unname(stats::coef(m)[2]),
               r2    = summary(m)$r.squared,
               us_lo = g$us[ok][1],
               us_hi = g$us[ok][sum(ok)],
               p_lo  = g$p[ok][1],
               p_hi  = g$p[ok][sum(ok)],
               stringsAsFactors = FALSE)
  })
  out <- do.call(rbind, Filter(Negate(is.null), out))
  out[order(-out$b_p), ]
}

fit_pn <- function(d) {
  s <- d[!(d$stage %in% DERIVED), ]
  out <- lapply(split(s, s$stage), function(g) {
    ok <- is.finite(g$us) & g$us > 0
    if (sum(ok) < 4L || length(unique(g$p[ok])) < 2L ||
        length(unique(g$n[ok])) < 2L) return(NULL)
    m <- stats::lm(log(g$us[ok]) ~ log(g$p[ok]) + log(g$n[ok]))
    cf <- stats::coef(m)
    data.frame(stage = g$stage[1],
               b_p = unname(cf[2]), c_n = unname(cf[3]),
               r2 = summary(m)$r.squared,
               stringsAsFactors = FALSE)
  })
  out <- do.call(rbind, Filter(Negate(is.null), out))
  out[order(-out$c_n, -out$b_p), ]
}

pl <- read_bench(pladder_path)
cat("=== exponent in p, cfa_3f, n =", min(pl$n), "(isolated) ===\n")
tab_p <- fit_p(pl, "cfa_3f")
print(data.frame(stage = tab_p$stage,
                 p_exp = round(tab_p$b_p, 2),
                 R2    = round(tab_p$r2, 3),
                 us_at_p6  = round(tab_p$us_lo, 2),
                 us_at_p96 = round(tab_p$us_hi, 1)),
      row.names = FALSE)

tab_pn <- NULL
if (!is.na(grid_path)) {
  gr <- read_bench(grid_path)
  cat("\n=== joint exponents, cfa_3f, p x n grid (isolated) ===\n")
  cat("    cost ~ p^b * n^c\n")
  tab_pn <- fit_pn(gr)
  print(data.frame(stage = tab_pn$stage,
                   p_exp = round(tab_pn$b_p, 2),
                   n_exp = round(tab_pn$c_n, 2),
                   R2    = round(tab_pn$r2, 3)),
        row.names = FALSE)
}

# --- JSON emit (hand-rolled; no jsonlite dependency) ----------------------
if (!is.na(out_json)) {
  q <- function(x) paste0('"', x, '"')
  num <- function(x) ifelse(is.finite(x), formatC(x, format = "g", digits = 6), "null")
  rows_json <- function(df, cols) {
    paste0("[", paste(apply(df, 1, function(r) {
      paste0("{", paste(vapply(cols, function(cn) {
        v <- r[[cn]]
        paste0(q(cn), ":", if (cn == "stage") q(v) else num(as.numeric(v)))
      }, character(1)), collapse = ","), "}")
    }), collapse = ","), "]")
  }

  # Raw ladder, long form, for the scaling chart.
  pls <- pl[pl$model == "cfa_3f" & pl$parameterization == "marker" &
              !(pl$stage %in% DERIVED) & pl$n == min(pl$n),
            c("stage", "p", "us")]
  parts <- c(
    paste0(q("p_exponents"), ":",
           rows_json(tab_p, c("stage", "b_p", "r2", "us_lo", "us_hi"))),
    paste0(q("ladder"), ":", rows_json(pls, c("stage", "p", "us")))
  )
  if (!is.null(tab_pn))
    parts <- c(parts, paste0(q("pn_exponents"), ":",
                             rows_json(tab_pn, c("stage", "b_p", "c_n", "r2"))))
  writeLines(paste0("{", paste(parts, collapse = ","), "}"), out_json)
  cat("\nwrote", out_json, "\n")
}
