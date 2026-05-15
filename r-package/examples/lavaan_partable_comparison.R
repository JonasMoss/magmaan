## Semantic partable comparison against lavaan.
##
## Run from the repo root after installing the package:
##     Rscript r-package/examples/lavaan_partable_comparison.R

suppressMessages({ library(magmaan); library(lavaan) })

check_partable <- function(label, fit, lav, tol = 2e-4) {
  cmp <- lavaan_compare_partable(fit, lavaan::parTable(lav), est_tolerance = tol)
  if (!isTRUE(cmp$ok)) {
    print(cmp$counts)
    print(utils::head(cmp$failures, 12L))
    stop(label, ": partable comparison failed", call. = FALSE)
  }
  cat(sprintf("%s: ok\n", label))
}

hs <- as.data.frame(lavaan::HolzingerSwineford1939)
model <- "visual =~ x1 + x2 + x3"

spec_mean <- model_spec(model, meanstructure = TRUE)
data_mean <- df_to_data(hs, spec_mean)
fit_mean <- fit_ml(spec_mean, data_mean)
lav_mean <- lavaan::cfa(model, data = hs, meanstructure = TRUE)
check_partable("single-group meanstructure", fit_mean, lav_mean)

group_labels <- unique(as.character(hs$school))
spec_group <- model_spec(model, group = "school", group_labels = group_labels,
                         meanstructure = TRUE)
data_group <- df_to_data(hs, spec_group, group = "school")
fit_group <- fit_ml(spec_group, data_group)
lav_group <- lavaan::cfa(model, data = hs, group = "school",
                         meanstructure = TRUE)
check_partable("multi-group meanstructure", fit_group, lav_group, tol = 5e-4)
