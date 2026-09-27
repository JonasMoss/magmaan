#!/usr/bin/env Rscript
script <- normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
here <- dirname(dirname(script))
source(file.path(here, "R", "designs.R")); source(file.path(here, "R", "fit.R"))
# No candidate implies unresolved, never nonexistent. A lower ordinary fit
# must remain visible and must not silently redefine the sphere reference.
x <- data.frame(design = "test", n = 20L, rep = 1L, transform = "native", domain = "ML",
  route = c("sphere", "sphere", "ordinary"), screened = c(FALSE, FALSE, TRUE),
  objective = c(.3, .2, .1), start_id = c("canonical", "random_1", "default"),
  backend = c("port", "port", "port"), chart_1e6 = "representable", chart_level = .5,
  label = c("accuracy_unresolved", "accuracy_unresolved", "screened_candidate"))
r <- reference_rows(x)
stopifnot(r$reference_label == "no_screened_reference", is.na(r$best_objective),
          compare_rows(x, r)$comparison[3] == "no_sphere_reference")
x$screened[1:2] <- TRUE; x$objective[1:2] <- .2
r <- reference_rows(x)
stopifnot(r$reference_label == "repeated_best_observed", r$matching_starts == 2,
          compare_rows(x, r)$comparison[3] == "better_than_sphere_reference")
x$start_id[2] <- "canonical"
stopifnot(reference_rows(x)$reference_label == "single_start_best_observed")
stopifnot(chart_label(5e-7, 1e-6) == "near_chart_boundary",
          chart_label(5e-7, 1e-8) == "representable", chart_label(NA_real_, 1e-6) == "unavailable")

# Changing the reporting marker must preserve the point/objective, even when
# the selected marker is not the first indicator. Variable order is retained.
library(magmaanlab)
data <- draw_data(design_sigma(designs_all()$weak_marker), 100, 91842)
sample <- sample_moments(data); spec <- model_spec(model_syntax)
f <- frontier_fit_sphere(spec, data, polish = FALSE)
target <- strong_marker_model(f$gauge$sphere_partable, sample)
stopifnot(any(target$partable$free != spec$partable$free))
a <- assess_endpoint(f, sample, "ML", TRUE)
stopifnot(is.finite(a$record$objective_gap), a$record$objective_gap < 1e-8,
          isTRUE(a$record$full_sphere))
# Random starts transform in their proper units and enter the sphere as user
# hints; neither backend silently replaces them with the canonical start.
theta <- random_start(spec, sample, 172)
sample2 <- sample; sample2$S[[1]] <- sample2$S[[1]] * 1e4
theta2 <- random_start(spec, sample2, 172)
pt <- spec$partable; factors <- rep(1, length(theta))
factors[pt$free[pt$op == "~~" & pt$free > 0]] <- 1e4
stopifnot(max(abs(theta2 - theta * factors)) < 1e-8)
for (backend in c("port", "nlopt-lbfgs")) {
  a <- run_fit(spec, data, sample, "ML", "sphere", backend, "random_1", theta)
  stopifnot(a$record$returned, a$record$start_used == "user")
}
cat("Label, chart-translation and start checks passed.\n")
