#!/usr/bin/env Rscript
# Preliminary cross-MAR/MCAR summary tables for coauthors.
# Reads results/summary_rejection.csv (no fitting), writes results/prelim_tables.md
# and prints to the console. Rough method-comparison view, not the final report.

res_dir  <- "results"
sum_path <- file.path(res_dir, "summary_rejection.csv")
stopifnot(file.exists(sum_path))
S <- read.csv(sum_path, stringsAsFactors = FALSE)

alpha   <- 0.05
nn_fams <- c("vm1", "vm2", "ig1", "ig2", "pl1", "pl2")  # the 6 non-normal families

# --- condition label from (mech,rate) ------------------------------------
cond_of <- function(mech, rate) {
  ifelse(mech == "complete", "Complete",
    paste0(mech, " ", formatC(round(100 * rate), format = "d"), "%"))
}
S$cond <- cond_of(S$mech, S$rate)
cond_order <- c("Complete", "MCAR 15%", "MCAR 30%", "MAR 15%", "MAR 30%")

pa_path <- file.path(res_dir, "power_adjusted.csv")
PA <- if (file.exists(pa_path)) {
  z <- read.csv(pa_path, stringsAsFactors = FALSE); z$cond <- cond_of(z$mech, z$rate); z
} else NULL

# headline method subsets (drop the redundant YB_mplus==MLR and SB==YB_exact dupes)
gof_methods    <- c("naive", "MLR", "YB_exact", "SS", "SF",
                    "EBA4", "pEBA4", "pall", "pOLS", "all")
nested_methods <- c("naive", "SB", "adjusted", "mixture")

fmt2 <- function(x) ifelse(is.na(x), "--", formatC(x, format = "f", digits = 3))

# method x condition matrix, aggregated over a set of distributions (mean[,max])
mat <- function(outcome, truth, methods, dists, show_max = FALSE) {
  d <- S[S$outcome == outcome & S$truth == truth & S$method %in% methods &
         S$dist %in% dists, ]
  d$cond <- factor(d$cond, levels = cond_order)
  conds <- cond_order[cond_order %in% as.character(unique(d$cond))]
  out <- data.frame(Method = methods, stringsAsFactors = FALSE)
  for (cc in conds) {
    dc <- d[as.character(d$cond) == cc, ]
    mean_v <- vapply(methods, function(m) mean(dc$reject[dc$method == m]), numeric(1))
    if (show_max && length(dists) > 1) {
      max_v <- vapply(methods, function(m) {
        v <- dc$reject[dc$method == m]; if (length(v)) max(v) else NA_real_
      }, numeric(1))
      out[[cc]] <- paste0(fmt2(mean_v), " (", fmt2(max_v), ")")
    } else {
      out[[cc]] <- fmt2(mean_v)
    }
  }
  out
}

# power (ordinary or size-adjusted) method x condition, mean over a dist set
pmat <- function(outcome, methods, dists, value = "power") {
  d <- PA[PA$outcome == outcome & PA$method %in% methods & PA$dist %in% dists, ]
  d$cond <- factor(d$cond, levels = cond_order)
  conds <- cond_order[cond_order %in% as.character(unique(d$cond))]
  out <- data.frame(Method = methods, stringsAsFactors = FALSE)
  for (cc in conds) {
    dc <- d[as.character(d$cond) == cc, ]
    out[[cc]] <- fmt2(vapply(methods,
      function(m) mean(dc[[value]][dc$method == m]), numeric(1)))
  }
  out
}

# headline: Type-I, ordinary power, size-adjusted power at one fixed cell
headline <- function(outcome, truth_pow, methods, mech, rate, dists) {
  ti <- S[S$outcome == outcome & S$truth == "h0" & S$mech == mech &
          S$rate == rate & S$dist %in% dists & S$method %in% methods, ]
  pw <- PA[PA$outcome == outcome & PA$mech == mech & PA$rate == rate &
           PA$dist %in% dists & PA$method %in% methods, ]
  g <- function(df, col, m) unname(mean(df[[col]][df$method == m]))
  out <- data.frame(
    Method = methods,
    `Type-I` = fmt2(vapply(methods, function(m) g(ti, "reject", m), numeric(1))),
    `Power` = fmt2(vapply(methods, function(m) g(pw, "power", m), numeric(1))),
    `Size-adj power` =
      fmt2(vapply(methods, function(m) g(pw, "power_adj", m), numeric(1))),
    check.names = FALSE, stringsAsFactors = FALSE)
  rownames(out) <- NULL
  out
}

# method x distribution matrix at one fixed condition
grad <- function(outcome, truth, methods, mech, rate,
                 dists = c("norm", nn_fams)) {
  d <- S[S$outcome == outcome & S$truth == truth & S$method %in% methods &
         S$mech == mech & S$rate == rate, ]
  out <- data.frame(Method = methods, stringsAsFactors = FALSE)
  for (dd in dists) {
    dc <- d[d$dist == dd, ]
    out[[dd]] <- fmt2(vapply(methods, function(m) {
      v <- dc$reject[dc$method == m]; if (length(v)) v[1] else NA_real_
    }, numeric(1)))
  }
  out
}

kab <- function(df, cap) knitr::kable(df, align = "l", caption = cap, format = "pipe")

con <- file(file.path(res_dir, "prelim_tables.md"), open = "wt")
emit <- function(...) { cat(..., "\n", sep = ""); cat(..., "\n", sep = "", file = con) }
emit_tab <- function(k) { print(k); writeLines(as.character(k), con); cat("\n", file = con) }

emit("# Experiment 23 -- preliminary method comparison (",
     "reject rate at alpha=", alpha, ", ~5000 reps)")
emit("")
emit("Two regimes: under **MCAR** normal-theory FIML stays consistent, so the ",
     "h0 columns isolate the *reference-law correction*. Under **MAR + non-",
     "normality** FIML is itself biased (linear conditional-mean imputation), so ",
     "no test correction can restore level -- read those columns as a bound, not ",
     "a test failure. Normal-data MAR is the control (FIML stays consistent there).")
emit("")
emit("Legend: naive=uncorrected LRT; MLR=Yuan-Bentler mean scaling (lavaan/Mplus ",
     "default); YB_exact=exact saturated-space mean scaling (==SB); SS=mean+",
     "variance; SF=scaled-F; EBA4/pEBA4=eigenvalue-block family; pall/pOLS=",
     "penalized variants; all=full FMG mixture. Non-normal = mean over 6 families ",
     "{vm1,vm2,ig1,ig2,pl1,pl2}; '(x)' = worst family.")
emit("")

emit("## GOF Type-I error (truth = h0; nominal .05)")
emit("")
emit_tab(kab(mat("gof", "h0", gof_methods, "norm"),
    "Normal data (control). MCAR and MAR both stay calibrated -- isolates that MAR breakage below is a non-normality x FIML interaction, not MAR per se."))
emit("")
emit_tab(kab(mat("gof", "h0", gof_methods, nn_fams, show_max = TRUE),
    "Non-normal data: mean (worst) over the 6 families. MCAR columns are the clean calibration claim; MAR columns are FIML's own bias."))
emit("")

emit("## GOF Type-I non-normality gradient (severe missingness, by family)")
emit("")
emit_tab(kab(grad("gof", "h0", gof_methods, "MCAR", 0.3),
    "MCAR 30%: reference-law correction in isolation. *1 = skew2/kurt7, *2 = skew3/kurt21."))
emit("")
emit_tab(kab(grad("gof", "h0", gof_methods, "MAR", 0.3),
    "MAR 30%: same cells under MAR -- the FIML-bias regime."))
emit("")

emit("## GOF power: ordinary vs size-adjusted")
emit("")
emit("Size-adjusted power scores each test at the cutoff that *would* give 5% ",
     "rejection of a true model in that cell, so the over-rejecting tests cannot ",
     "borrow detection from a broken Type-I rate. Read it as: 'at equal honest ",
     "size, who actually detects the misspecification?'")
emit("")
if (!is.null(PA)) {
  emit_tab(kab(headline("gof", "gof_power", gof_methods, "MCAR", 0.3, nn_fams),
    "Headline cell (MCAR 30%, non-normal mean): Type-I, ordinary power, size-adjusted power. Ordinary power tracks the Type-I inflation; size-adjusted power is ~equal for every valid test."))
  emit("")
  emit_tab(kab(pmat("gof", gof_methods, nn_fams, "power"),
    "Ordinary GOF power, non-normal mean over 6 families. Not size-corrected."))
  emit("")
  emit_tab(kab(pmat("gof", gof_methods, nn_fams, "power_adj"),
    "Size-adjusted GOF power, non-normal mean. The apparent spread above is almost entirely the Type-I inflation."))
} else {
  emit_tab(kab(mat("gof", "gof_power", gof_methods, nn_fams),
    "Ordinary GOF power, non-normal mean. (Run scripts/size_adjusted_power.R for the size-corrected version.)"))
}
emit("")

emit("## Nested (configural vs metric) Type-I and power")
emit("")
emit_tab(kab(mat("nested", "h0", nested_methods, nn_fams, show_max = TRUE),
    "Nested Type-I, non-normal mean (worst). Methods: naive / SB (mean) / adjusted (mean+var) / mixture (exact)."))
emit("")
emit_tab(kab(mat("nested", "nested_power", nested_methods, nn_fams),
    "Nested power, non-normal mean. Not size-adjusted."))

close(con)
cat("\nWrote ", file.path(res_dir, "prelim_tables.md"), "\n", sep = "")
