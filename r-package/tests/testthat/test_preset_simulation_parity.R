# Opt in with MAGMAAN_PARITY=1; oracle and population parameters are pinned
# to installed lavaan 0.7.2. Seeds identify every replicate independently.
.preset_simulation_models <- list(
  hs_ml = list(source = "hs", estimator = "ML"),
  democracy_ml = list(source = "democracy", estimator = "ML"),
  hs_school_ml = list(source = "hs", estimator = "ML", grouped = TRUE,
                      equal = c("loadings", "intercepts")),
  hs_mcar_fiml = list(source = "hs", estimator = "FIML", holes = "MCAR"),
  hs_mar_fiml = list(source = "hs", estimator = "FIML", holes = "MAR"),
  hs_delta_dwls = list(source = "hs", estimator = "DWLS", parameterization = "delta"),
  hs_theta_dwls = list(source = "hs", estimator = "DWLS", parameterization = "theta"),
  hs_school_theta_dwls = list(source = "hs", estimator = "DWLS", grouped = TRUE,
      parameterization = "theta", equal = c("loadings", "thresholds")))

.preset_simulation_key <- function(pt) paste(pt$lhs, pt$op, pt$rhs, pt$group, sep = "\r")

test_that("named simulation models retain lavaan preset parity", {
  skip_if(Sys.getenv("MAGMAAN_PARITY") != "1", "opt-in simulated preset parity")
  skip_if_not_installed("lavaan")
  skip_if(as.character(utils::packageVersion("lavaan")) != "0.7.2",
          "simulation oracle requires lavaan 0.7.2")
  started <- proc.time()[["elapsed"]]
  hs <- "visual =~ x1+x2+x3\ntextual =~ x4+x5+x6\nspeed =~ x7+x8+x9"
  democracy <- paste(
    "ind60 =~ x1+x2+x3", "dem60 =~ y1+y2+y3+y4", "dem65 =~ y5+y6+y7+y8",
    "dem60 ~ ind60", "dem65 ~ ind60+dem60", "y1 ~~ y5", "y2 ~~ y4+y6",
    "y3 ~~ y7", "y4 ~~ y8", "y6 ~~ y8", sep = "\n")
  populations <- list(
    hs = lavaan::sem(hs, lavaan::HolzingerSwineford1939, meanstructure = TRUE),
    democracy = lavaan::sem(democracy, lavaan::PoliticalDemocracy, meanstructure = TRUE),
    school = lavaan::sem(hs, lavaan::HolzingerSwineford1939, group = "school",
        group.equal = c("loadings", "intercepts"), meanstructure = TRUE))
  disagreements <- list()
  results <- list()
  for (case_id in seq_along(.preset_simulation_models)) {
    name <- names(.preset_simulation_models)[case_id]
    case <- .preset_simulation_models[[case_id]]
    grouped <- isTRUE(case$grouped)
    population <- populations[[if (grouped) "school" else case$source]]
    model <- if (case$source == "hs") hs else democracy
    variables <- lavaan::lavNames(population, "ov")
    labels <- if (grouped) lavaan::lavInspect(population, "group.label") else NULL
    moments <- lavaan::fitted(population)
    if (!grouped) moments <- list(moments)
    for (replicate in seq_len(20L)) {
      seed <- 590000L + 100L * case_id + replicate
      set.seed(seed)
      # N=300 per group; simulateData uses the fitted public-data population.
      data <- lavaan::simulateData(lavaan::parTable(population),
          sample.nobs = if (grouped) c(300L, 300L) else 300L, meanstructure = TRUE)
      if (grouped) data$school <- labels[data$group]
      if (!is.null(case$holes)) {
        for (j in if (case$holes == "MCAR") seq_len(9L) else c(2L, 5L, 8L)) {
          probability <- if (case$holes == "MCAR") rep(.15, nrow(data)) else {
            driver <- paste0("x", j - 1L)
            z <- (data[[driver]] - moments[[1]]$mean[driver]) /
              sqrt(moments[[1]]$cov[driver, driver])
            plogis(qlogis(.15) + .5 * z)
          }
          data[runif(nrow(data)) < probability, paste0("x", j)] <- NA_real_
        }
      }
      ordered <- if (case$estimator == "DWLS") variables else NULL
      if (!is.null(ordered)) for (variable in variables) {
        # Common population quantiles preserve thresholds across school groups.
        breaks <- moments[[1]]$mean[variable] +
          sqrt(moments[[1]]$cov[variable, variable]) * qnorm(c(.25, .5, .75))
        data[[variable]] <- as.integer(cut(data[[variable]], c(-Inf, breaks, Inf)))
      }
      args <- list(model = model, data = data, meanstructure = TRUE,
                   fixed_x = FALSE, estimator = case$estimator,
                   options = list(preset = "lavaan-0.7.2"))
      oracle_args <- list(model = model, data = data, meanstructure = TRUE,
          fixed.x = FALSE, estimator = if (case$estimator == "DWLS") "WLSMV" else "ML",
          se = "standard", test = "none")
      if (grouped) {
        args$groups <- oracle_args$group <- "school"
        args$group_equal <- oracle_args$group.equal <- case$equal
      }
      if (case$estimator == "FIML") oracle_args$missing <- "ml"
      if (!is.null(ordered)) {
        args$ordered <- oracle_args$ordered <- ordered
        args$parameterization <- oracle_args$parameterization <- case$parameterization
      }
      actual <- tryCatch(suppressWarnings(do.call(fit_model, args)), error = identity)
      oracle <- tryCatch(suppressWarnings(do.call(lavaan::sem, oracle_args)), error = identity)
      issue <- character()
      retry <- FALSE
      difference <- NA_real_
      endpoint_contract <- FALSE
      chisq_difference <- max_gradient <- max_se_difference <- NA_real_
      if (inherits(actual, "error") || inherits(oracle, "error")) {
        issue <- paste("fit error:", if (inherits(actual, "error")) conditionMessage(actual),
                       if (inherits(oracle, "error")) conditionMessage(oracle))
      } else {
        # Executed rescaled retries retain the task-47 endpoint contract.
        retry <- any(vapply(actual$fitting$attempts, function(a)
          isTRUE(a$standardized), logical(1)))
        converged <- lavaan::lavInspect(oracle, "converged")
        if (!identical(actual$converged, converged) ||
            !identical(actual$verdict$status, if (converged) "passed" else "failed"))
          issue <- c(issue, "convergence/verdict disagreement")
        if (isTRUE(actual$converged) && isTRUE(converged)) {
          mp <- actual$partable[actual$partable$free > 0L, ]
          lp <- lavaan::parTable(oracle)
          lp <- lp[lp$free > 0L, ]
          mk <- .preset_simulation_key(mp)
          lk <- .preset_simulation_key(lp)
          if (!setequal(mk, lk) || anyDuplicated(mk) || anyDuplicated(lk)) {
            issue <- c(issue, "free parameter keys differ")
          } else {
            oracle_est <- lp$est[match(mk, lk)]
            errors <- abs(mp$est - oracle_est)
            difference <- max(errors)
            # Match the pinned doctest Approx(...).epsilon(1e-5) gate.
            within_tolerance <- all(errors <= 1e-5 * (1 + pmax(abs(mp$est), abs(oracle_est))))
            if (!retry && !within_tolerance && is.finite(difference)) {
              # Approved task-59 endpoint contract. Seed 590214's PORT trace
              # and same-point derivatives locate divergence at rounding level;
              # see project/architecture/capabilities/optimizers.md.
              # Evaluate both endpoints in the oracle's search coordinates and
              # units, rather than comparing differently scaled fit summaries.
              theta <- oracle@optim$x
              theta[lp$free[match(mk, lk)]] <- mp$est
              endpoint <- lavaan:::lav_model_set_parameters(oracle@Model, theta)
              objective <- as.numeric(lavaan:::lav_model_objective(
                  endpoint, endpoint@GLIST, oracle@SampleStats, oracle@Data))
              gradient <- lavaan:::lav_model_grad(
                  endpoint, endpoint@GLIST, oracle@SampleStats, oracle@Data)
              oracle_objective <- as.numeric(oracle@optim$fx)
              # Objective agreement on the scale users see: lavaan's statistic
              # is 2 N fx. Near an optimum the gap is second order in the
              # estimate difference, so it is judged in chi-square units.
              chisq_difference <- 2 * lavaan::lavInspect(oracle, "ntotal") *
                abs(objective - oracle_objective)
              max_gradient <- max(abs(c(gradient, oracle@optim$dx)))
              oracle_se <- lp$se[match(mk, lk)]
              max_se_difference <- max(errors / oracle_se)
              endpoint_contract <- !length(issue) &&
                all(is.finite(c(chisq_difference, max_gradient,
                                max_se_difference))) && all(oracle_se > 0) &&
                max_gradient <= 1e-3 && chisq_difference <= 1e-6 &&
                max_se_difference <= 1e-3
            }
            if (!is.finite(difference) || (!retry && !within_tolerance && !endpoint_contract))
              issue <- c(issue, "estimate disagreement")
          }
        }
      }
      results[[length(results) + 1L]] <- data.frame(model = name, replicate = replicate,
          seed = seed, rescaled_retry = retry, endpoint_contract = endpoint_contract,
          max_abs_difference = difference,
          chisq_difference = chisq_difference,
          max_gradient = max_gradient, max_se_difference = max_se_difference)
      if (length(issue)) disagreements[[length(disagreements) + 1L]] <- data.frame(
          model = name, replicate = replicate, seed = seed, rescaled_retry = retry,
          max_abs_difference = difference, chisq_difference = chisq_difference,
          max_gradient = max_gradient, max_se_difference = max_se_difference,
          issue = paste(issue, collapse = "; "))
    }
  }
  results <- do.call(rbind, results)
  for (name in names(.preset_simulation_models)) {
    rows <- results[results$model == name, ]
    cat(sprintf("\n%s: %d replicates, %d rescaled retries, %d endpoint contracts, max abs difference %.9g\n",
        name, nrow(rows), sum(rows$rescaled_retry), sum(rows$endpoint_contract),
        if (all(is.na(rows$max_abs_difference))) NA_real_ else
          max(rows$max_abs_difference, na.rm = TRUE)))
  }
  cat(sprintf("Elapsed: %.1f seconds\n", proc.time()[["elapsed"]] - started))
  print(results[results$endpoint_contract, ], row.names = FALSE)
  if (length(disagreements)) print(do.call(rbind, disagreements), row.names = FALSE)
  expect_length(disagreements, 0L)
})
