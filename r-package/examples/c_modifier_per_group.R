## magmaan R bindings — per-group `c(...)` modifiers + group identity on the
## partable, cross-checked against lavaan's parameter table.
##
## Run from the repo root (after `R CMD INSTALL r-package`, or
## `devtools::load_all("r-package")`):
##     Rscript r-package/examples/c_modifier_per_group.R
##
## `c(v1, …, vk)*x` applies the g-th atom for group g (a fixed value, `NA` for
## free, a label, or `start(…)`). The grouping variable name + per-group labels
## live on the partable (as the `magmaan.group_var` / `magmaan.group_labels`
## attributes; on a fit object as `$group_var` / `$group_labels`).

suppressMessages({ library(magmaan); library(lavaan) })

ok <- function(cond) if (isTRUE(cond)) "ok" else "MISMATCH"

## A 2-group model where group 1's x2 loading is fixed at 0.8 and group 2's at
## 1.2; everything else free.  (`c(0.8, 1.2)*x2` is two fixed values; `NA` would
## free that group's loading; `c(L, L)*x2` would tie it across groups.)
m  <- "f =~ x1 + c(0.8, 1.2)*x2 + x3"
pt <- magmaan_core$lavaan_lavaanify(m, n_groups = 2L, group_var = "school",
                      group_labels = c("Pasteur", "Grant-White"))

cat("--- group identity on the partable ---\n")
cat(sprintf("  attr(pt, \"magmaan.group_var\")    = %s\n", attr(pt, "magmaan.group_var")))
cat(sprintf("  attr(pt, \"magmaan.group_labels\") = %s\n\n", paste(attr(pt, "magmaan.group_labels"), collapse = ", ")))

cat("--- the f =~ x2 rows (one per group) ---\n")
sub <- pt[pt$lhs == "f" & pt$rhs == "x2", c("block", "group", "free", "ustart", "label")]
print(sub, row.names = FALSE)
cat(sprintf("  group 1 ustart = %.2f (fixed: %s) | group 2 ustart = %.2f (fixed: %s)\n\n",
            sub$ustart[sub$group == 1], ok(sub$free[sub$group == 1] == 0),
            sub$ustart[sub$group == 2], ok(sub$free[sub$group == 2] == 0)))

## Cross-check the lavaanify output against lavaan's parTable.
## (lavaan's `c()` modifier on the marker indicator would clash with auto.fix.first,
## so we put the per-group values on x2; x1 stays the auto-fixed marker in both.)
hs    <- HolzingerSwineford1939
df_hs <- as.data.frame(hs[c("school", paste0("x", 1:3))])
lav_pt <- parTable(lavaan::cfa("f =~ x1 + c(0.8, 1.2)*x2 + x3",
                               data = df_hs, group = "school", do.fit = FALSE))
lav_sub <- lav_pt[lav_pt$lhs == "f" & lav_pt$rhs == "x2", c("block", "group", "free", "ustart")]
cat("--- lavaan parTable, f =~ x2 rows ---\n")
print(lav_sub, row.names = FALSE)
cat(sprintf("  match: ustart %s | fixed-ness (free == 0) %s\n",
            ok(isTRUE(all.equal(sub$ustart[order(sub$group)], lav_sub$ustart[order(lav_sub$group)]))),
            ok(all((sub$free[order(sub$group)] == 0) == (lav_sub$free[order(lav_sub$group)] == 0)))))

## And it fits like any other multi-group model:
ssg <- magmaan_core$data_sample_stats_from_raw(lapply(split(df_hs[paste0("x", 1:3)], df_hs$school), as.matrix))
fit <- magmaan_core$fit_fit(pt, ssg)
cat(sprintf("\n--- fit ---\n  ngroups = %d (%s), npar = %d, free λ(f=~x2)? %s\n",
            fit$ngroups, paste(fit$group_labels, collapse = "/"), fit$npar,
            ok(all(fit$partable$free[fit$partable$lhs == "f" & fit$partable$rhs == "x2"] == 0))))
