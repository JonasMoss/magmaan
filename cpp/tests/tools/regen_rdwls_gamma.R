# Dedicated lavaan 0.7-3 mixed Gamma oracle; does not change the pinned oracle.
# R_LIBS=~/.cache/magmaan-rlib/lavaan-073 Rscript cpp/tests/tools/regen_rdwls_gamma.R
stopifnot(as.character(packageVersion("lavaan")) == "0.7.3")
d <- lavaan::HolzingerSwineford1939[paste0("x", 1:6)]
for (j in 1:3) d[[j]] <- cut(d[[j]], 3, labels = FALSE)
f <- lavaan::cfa("f =~ x1+x2+x3+x4+x5+x6", d, ordered = paste0("x", 1:3),
                 estimator = "DWLS", parameterization = "delta")
g <- lavaan::lavInspect(f, "gamma")
jsonlite::write_json(list(source = "lavaan 0.7-3", data = "HolzingerSwineford1939; x1-x3 equal-width cut(..., 3)",
  model = "f =~ x1+x2+x3+x4+x5+x6", ordered = c(1,1,1,0,0,0),
  X = unname(as.matrix(d)), gamma_diag = unname(diag(g))),
  "cpp/tests/fixtures/rdwls_gamma_073.json", digits = 16, pretty = TRUE, auto_unbox = TRUE)
