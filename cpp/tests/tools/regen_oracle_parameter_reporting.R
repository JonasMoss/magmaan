# Targeted identification/reporting oracle. Run standalone or from regen_oracle.R.
suppressPackageStartupMessages({library(lavaan); library(jsonlite)})
if (!exists("repo_root")) {
  script <- sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1])
  repo_root <- normalizePath(file.path(dirname(script), "../../.."))
}
fixture_dir <- file.path(repo_root, "cpp/tests/fixtures")
stopifnot(gsub("-", ".", readLines(file.path(fixture_dir, "lavaan_version.txt"))[1]) ==
            as.character(packageVersion("lavaan")))
rows <- function(fit) {
  p <- parTable(fit)
  p[, c("lhs", "op", "rhs", "group", "free", "ustart", "est")]
}
base_model <- "f =~ x1 + x2 + x3 + x4"
dat <- HolzingerSwineford1939
cases <- list()
for (id in c("single", "groups", "scalar", "fixed_intercept", "fixed_mean", "fixed_loading")) {
  model <- switch(id, fixed_intercept=paste(base_model, "x1 ~ 2*1", sep="\n"),
                  fixed_mean=paste(base_model, "f ~ 2*1", sep="\n"),
                  fixed_loading="f =~ 1*x1 + x2 + x3 + x4", base_model)
  grouped <- id %in% c("groups", "scalar")
  equal <- if (id == "scalar") c("loadings", "intercepts") else character()
  fit <- cfa(model, data=dat, group=if (grouped) "school" else NULL,
             group.equal=equal, meanstructure=TRUE, effect.coding=TRUE,
             se="none", test="standard")
  stopifnot(lavInspect(fit, "converged"))
  ss <- lavInspect(fit, "sampstat"); implied <- fitted(fit)
  if (!grouped) {ss <- list(ss); implied <- list(implied)}
  cases[[id]] <- list(input=model, n_groups=length(ss), group_equal=equal,
    rows=rows(fit), nobs=as.integer(lavInspect(fit, "nobs")),
    sample=unname(lapply(ss, function(s) list(cov=unname(s$cov), mean=unname(s$mean)))),
    implied=unname(lapply(implied, function(s) list(cov=unname(s$cov), mean=unname(s$mean)))),
    chisq=unname(fitMeasures(fit,"chisq")), df=unname(fitMeasures(fit,"df")))
}
# Independent deterministic synthetic data; no third-party raw data is stored.
set.seed(28410)
n <- 900
f <- rnorm(n)
x <- sapply(c(.8, .7, .6, .9), function(l) l*f + rnorm(n))
d <- as.data.frame(x); names(d) <- paste0("x",1:4)
d$g <- rep(c("a","b"), each=n/2)
ord_cases <- list()
for (mixed in c(FALSE,TRUE)) for (grouped in c(FALSE,TRUE)) for (param in c("delta","theta")) {
  z <- d
  ordered <- paste0("x",if (mixed) 1:2 else 1:4)
  for (v in ordered) z[[v]] <- ordered(cut(z[[v]], c(-Inf,-.6,.4,Inf)))
  fit <- cfa(base_model,data=z,ordered=ordered,group=if(grouped) "g" else NULL,
             estimator="DWLS",parameterization=param,se="none",test="standard")
  stopifnot(lavInspect(fit,"converged"))
  id <- paste(if(mixed) "mixed" else "ordinal",if(grouped) "groups" else "single",param,sep="_")
  ord_cases[[id]] <- list(input=paste(base_model,
    paste(paste0(ordered," | t1 + t2"),collapse="\n"),sep="\n"),
    n_groups=if(grouped) 2L else 1L, parameterization=param, rows=rows(fit))
}
# group.equal is processed before effect coding: retain this ordering even
# where latent-mean equality has no free coordinates to tie at that stage.
identification <- lapply(list(means="means", scalar_means=c("loadings","intercepts","means")), function(equal) {
  fit <- cfa(base_model, data=dat, group="school", group.equal=equal,
             meanstructure=TRUE, effect.coding=TRUE, do.fit=FALSE)
  list(input=base_model, group_equal=equal, rows=rows(fit))
})
write_json(list(lavaan_version=as.character(packageVersion("lavaan")),
                effect_coding=cases, effect_identification=identification, ordinal=ord_cases),
           file.path(fixture_dir,"parameter_reporting.json"),
           auto_unbox=TRUE, pretty=TRUE, digits=16, na="null")
