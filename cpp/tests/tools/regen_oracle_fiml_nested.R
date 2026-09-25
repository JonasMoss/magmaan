# FIML scalar-invariance regression. Sourced by regen_oracle.R; also runnable
# alone to refresh this slice. Only installed lavaan output is the oracle.
if (!exists("fixtures")) {
  arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  here <- dirname(normalizePath(sub("^--file=", "", arg[1])))
  fixtures <- normalizePath(file.path(here, "..", "fixtures"))
}
suppressMessages({ library(lavaan); library(jsonlite) })
stopifnot(packageVersion("lavaan") ==
            package_version(trimws(readLines(file.path(fixtures, "lavaan_version.txt"))[1])))
local({
  d <- HolzingerSwineford1939
  d$x2[seq(1L, nrow(d), 7L)] <- NA_real_
  syntax <- "visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6"
  metric <- cfa(syntax, d, group = "school", group.equal = "loadings",
                estimator = "MLR", missing = "ml")
  scalar <- cfa(syntax, d, group = "school", group.equal = c("loadings", "intercepts"),
                estimator = "MLR", missing = "ml")
  stopifnot(lavInspect(metric, "converged"), lavInspect(scalar, "converged"))
  lr <- lavTestLRT(metric, scalar, method = "satorra.2000", A.method = "delta",
                   scaled.shifted = FALSE)
  model <- function(fit) {
    pt <- parTable(fit)
    pt <- pt[pt$op != "==", c("lhs", "op", "rhs", "group", "est")]
    list(rows = pt, chisq = unname(fitMeasures(fit, "chisq")),
         df = unname(fitMeasures(fit, "df")))
  }
  delta <- unname(fitMeasures(scalar, "chisq") - fitMeasures(metric, "chisq"))
  raw <- lapply(unique(as.character(d$school)), function(g) {
    X <- unname(as.matrix(d[d$school == g, paste0("x", 1:6)]))
    list(X = X, mask = 1L * !is.na(X))
  })
  out <- list(`_meta` = list(lavaan_version = as.character(packageVersion("lavaan")),
                            fixture_kind = "fiml.scalar_nested",
                            information = "observed", h1_information = "structured"),
              input = syntax, raw = raw, metric = model(metric), scalar = model(scalar),
              df_diff = as.numeric(lr[2, "Df diff"]), T_diff = delta,
              T_scaled = as.numeric(lr[2, "Chisq diff"]),
              scale_c = delta / as.numeric(lr[2, "Chisq diff"]))
  write_json(out, file.path(fixtures, "fiml", "scalar_invariance_hs_nested.json"),
             pretty = TRUE, auto_unbox = TRUE, na = "null", digits = NA)
})
