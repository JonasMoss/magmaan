# End-to-end checks of fitted values and identification against lavaan.
suppressPackageStartupMessages({library(magmaanlab); library(lavaan)})
core <- magmaan_core
ctrl <- list(max_iter=4000, ftol=1e-13, gtol=1e-8)
model <- "f =~ x1 + x2 + x3 + x4"
key <- function(p) paste(p$lhs,p$op,p$rhs,p$group,sep="|")
compare <- function(fit, reference, tol=2e-4) {
  p <- fit$partable
  p <- p[p$op %in% c("=~","~~","~1","|","~"),]
  r <- parTable(reference)
  idx <- match(key(p),key(r))
  stopifnot(!anyNA(idx), max(abs(p$est-r$est[idx])) < tol)
}
for (grouped in c(FALSE,TRUE)) {
  m <- model_spec(model,meanstructure=TRUE,effect_coding=TRUE,
                  group=if(grouped) "school" else NULL,
                  group_labels=if(grouped) c("Pasteur","Grant-White") else NULL)
  d <- df_to_data(HolzingerSwineford1939,m,scaling="n")
  fit <- core$fit_ml(m,d,optimizer="nlopt-slsqp",control=ctrl)
  lav <- cfa(model,data=HolzingerSwineford1939,meanstructure=TRUE,effect.coding=TRUE,
             group=if(grouped) "school" else NULL)
  compare(fit,lav)
  p <- fit$partable
  for (g in unique(p$group[p$group>0])) {
    stopifnot(abs(sum(p$est[p$op=="=~" & p$group==g])-4)<1e-8,
              abs(sum(p$est[p$op=="~1" & p$lhs!="f" & p$group==g]))<1e-8,
              p$free[p$op=="~1" & p$lhs=="f" & p$group==g]>0)
  }
}
# Same identification also flows through FIML.
m <- model_spec(model,meanstructure=TRUE,effect_coding=TRUE)
x <- HolzingerSwineford1939
x$x2[seq(2,nrow(x),9)] <- NA
fit <- core$fit_fiml(m,df_to_fiml_data(x,m),optimizer="nlopt-slsqp",control=ctrl)
compare(fit,cfa(model,data=x,missing="fiml",effect.coding=TRUE),tol=5e-4)

set.seed(28410)
n <- 900
f <- rnorm(n)
x <- as.data.frame(sapply(c(.8,.7,.6,.9),function(l) l*f+rnorm(n)))
names(x) <- paste0("x",1:4); x$g <- rep(c("a","b"),each=n/2)
for (mixed in c(FALSE,TRUE)) for (grouped in c(FALSE,TRUE)) for (param in c("delta","theta")) {
  d <- x
  ord <- paste0("x",if(mixed) 1:2 else 1:4)
  for (v in ord) d[[v]] <- ordered(cut(d[[v]],c(-Inf,-.6,.4,Inf)))
  m <- model_spec(model,ordered=ord,parameterization=param,meanstructure=TRUE,
                  group=if(grouped) "g" else NULL,
                  group_labels=if(grouped) c("a","b") else NULL)
  stats <- if(mixed) core$data_mixed_ordinal_stats_from_df(d,m) else core$data_ordinal_stats_from_df(d,m)
  fit <- if(mixed) core$fit_dwls_mixed_ordinal(m,stats,control=ctrl) else core$fit_dwls_ordinal(m,stats,control=ctrl)
  lav <- cfa(model,data=d,ordered=ord,parameterization=param,estimator="DWLS",
             group=if(grouped) "g" else NULL)
  compare(fit,lav,tol=2e-3)
  p <- fit$partable
  idx <- p$op=="~~" & p$lhs==p$rhs & p$lhs %in% ord
  stopifnot(all(p$ustart[idx]==1), all(p$free[idx]==0))
  if(param=="delta") stopifnot(all(p$est[idx]<1)) else stopifnot(all(p$est[idx]==1))
}
cat("Parameter reporting and effect-coding checks passed.\n")
