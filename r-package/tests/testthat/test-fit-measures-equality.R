test_that("fit measures use equality-reduced ordinal and continuous moments", {
  skip_if_not_installed("lavaan")
  set.seed(5162)
  gen <- function(n) {
    eta <- matrix(rnorm(2*n), n, 2)
    eta[,2] <- .4*eta[,1] + sqrt(.84)*eta[,2]
    x <- cbind(outer(eta[,1], c(.8,.7,.65)),
               outer(eta[,2], c(.85,.75,.6))) + matrix(rnorm(6*n, sd=.7), n, 6)
    # A small omitted residual association gives nonzero RMSEA and CFI gaps.
    x[,4] <- x[,4] + .3*x[,1]
    as.data.frame(x)
  }
  d <- rbind(gen(500), gen(450))
  vars <- paste0("x", 1:6)
  names(d) <- vars
  d$g <- rep(c("a", "b"), c(500,450))
  ordinal <- d
  for (v in vars) ordinal[[v]] <- as.integer(cut(d[[v]], c(-Inf,-.5,.5,Inf)))
  syntax <- "f1 =~ x1+x2+x3\nf2 =~ x4+x5+x6"
  labelled <- "f1 =~ x1+c(l,l)*x2+x3\nf2 =~ x4+x5+x6"
  measures <- c("df", "chisq", "pvalue", "rmsea", "cfi", "tli",
                "rmsea.ci.lower", "rmsea.ci.upper", "rmsea.pvalue")
  for (parameterization in c("delta", "theta", "continuous")) {
    categorical <- parameterization != "continuous"
    cases <- if (categorical) list("thresholds", c("thresholds","loadings"), character())
             else list("loadings", c("loadings","intercepts"), character())
    for (equal in cases) {
      model <- if (length(equal)) syntax else labelled
      data <- if (categorical) ordinal else d
      args <- list(syntax=model, group="g", group_labels=c("a","b"),
                   group_equal=equal, meanstructure=TRUE)
      if (categorical) args <- c(args, list(ordered=vars, parameterization=parameterization))
      actual <- fit_model(do.call(model_spec,args), data,
                          estimator=if (categorical) "DWLS" else "ML")
      oracle_args <- args
      names(oracle_args) <- gsub("_", ".", names(oracle_args), fixed=TRUE)
      names(oracle_args)[names(oracle_args)=="syntax"] <- "model"
      names(oracle_args)[names(oracle_args)=="group.labels"] <- "group.label"
      oracle <- do.call(lavaan::cfa, c(oracle_args, list(data=data,
        estimator=if (categorical) "WLSMV" else "ML")))
      info <- paste(parameterization, paste(equal, collapse="+"), model)
      expect_true(actual$converged, info=info)
      expect_true(lavaan::lavInspect(oracle,"converged"), info=info)
      fm <- fit_measures(actual)
      reference <- lavaan::fitMeasures(oracle, measures)
      if (categorical) {
        expect_equal(fm$df, unname(reference["df"]), tolerance=0, info=info)
        # Native reporting uses sum(n_g F_g), lavaan sum((n_g - 1) F_g).
        # n/(n-G) is exact only for equally sized groups.
        n <- nrow(data)
        g <- length(unique(data$g))
        group_n <- as.numeric(lavaan::lavInspect(oracle, "nobs"))
        factors <- group_n / (group_n - 1)
        oracle_test <- lavaan::lavInspect(oracle, "test")$standard
        reference["chisq"] <- sum(oracle_test$stat.group * factors)
        baseline <- lavaan::fitMeasures(oracle, c("baseline.chisq", "baseline.df"))
        x2 <- unname(reference["chisq"])
        df <- unname(reference["df"])
        x2_null <- sum(oracle@baseline$test$standard$stat.group * factors)
        df_null <- unname(baseline["baseline.df"])
        expect_equal(fm$baseline.chisq, x2_null, tolerance=1e-5, info=info)
        expect_equal(fm$baseline.df, df_null, tolerance=0, info=info)
        reference["pvalue"] <- pchisq(x2, df, lower.tail=FALSE)
        reference["rmsea"] <- getFromNamespace("lav_fit_rmsea", "lavaan")(x2, df, n, g=g)
        reference[c("rmsea.ci.lower", "rmsea.ci.upper")] <-
          unlist(getFromNamespace("lav_fit_rmsea_ci", "lavaan")(x2, df, n, g=g))
        reference["rmsea.pvalue"] <-
          getFromNamespace("lav_fit_rmsea_closefit", "lavaan")(x2, df, n, g=g)
        reference["cfi"] <- getFromNamespace("lav_fit_cfi", "lavaan")(x2, df, x2_null, df_null)
        reference["tli"] <- getFromNamespace("lav_fit_tli", "lavaan")(x2, df, x2_null, df_null)
        expect_equal(convention_inference(actual,"DWLS")$test$unscaled_statistic,
          oracle_test$stat, tolerance=1e-5, info=info)
      }
      expect_equal(unname(unlist(fm[measures])), unname(reference), tolerance=1e-5, info=info)
      if (categorical) {
        expect_equal(fm$df, convention_inference(actual,"WLSMV")$test$df, info=info)
        # Serialized fits lose the native cache; reconstruction must agree too.
        reconstructed <- unserialize(serialize(actual,NULL))
        expect_equal(fit_measures(reconstructed)[measures], fm[measures], tolerance=1e-10, info=info)
      }
    }
  }
})
