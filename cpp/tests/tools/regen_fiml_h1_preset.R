#!/usr/bin/env Rscript
# Small-N saturated moments used by the lavaan-0.7.2 FIML fitting preset.
# Literal two-factor Gaussian population with independent unit residuals.
stopifnot(as.character(utils::packageVersion("lavaan")) == "0.7.2")
model <- "Y =~ y1+y2+y3\nX =~ x1+x2+x3\nY ~ X"
cases <- list()
for (seed in c(12953003L, 12953002L, 12954001L)) {
  n <- if (seed == 12954001L) 40L else 20L
  set.seed(seed)
  x <- rnorm(n); y <- .25*x+rnorm(n)
  d <- as.data.frame(cbind(outer(y,c(1,.8,.6))+matrix(rnorm(n*3),n),
                           outer(x,c(1,.8,.6))+matrix(rnorm(n*3),n)))
  names(d) <- c(paste0("y",1:3),paste0("x",1:3))
  for (v in names(d)) d[runif(n)<.2,v] <- NA_real_
  l <- suppressWarnings(lavaan::sem(model,d,missing="ml",meanstructure=TRUE,
    fixed.x=FALSE,se="none",test="none",do.fit=FALSE))
  h <- lavaan::lavInspect(l,"h1")
  cases[[length(cases)+1L]] <- list(seed=seed,n=n,raw=unname(as.matrix(d)),
    mean=unname(h$mean),cov=unname(h$cov),
    starts=unname(lavaan::parTable(l)$start),em_options=l@Options$em.h1.args)
}
jsonlite::write_json(list(source="regen_fiml_h1_preset.R; D5 MCAR 20% per variable",
  lavaan_version="0.7.2",model=model,cases=cases),
  "cpp/tests/fixtures/fitting/lavaan_fiml_h1_0_7_2.json",
  auto_unbox=TRUE,digits=17,pretty=TRUE,na="null")
