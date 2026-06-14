core <- magmaan::magmaan_core

target_g1 <- matrix(c(
  1.00, 0.25, 0.30,
  0.25, 1.00, 0.20,
  0.30, 0.20, 1.00
), 3, 3, byrow = TRUE)
target_g2 <- matrix(c(
  1.00, -0.15, 0.45,
 -0.15,  1.00, 0.10,
  0.45,  0.10, 1.00
), 3, 3, byrow = TRUE)

marginals_g1 <- list(
  y_ord = c(0.20, 0.50, 0.30),
  x_cont = NULL,
  z_ord = c(0.40, 0.60)
)
marginals_g2 <- list(
  y_ord = c(0.30, 0.40, 0.30),
  x_cont = NULL,
  z_ord = c(0.55, 0.45)
)
labels <- c("school_a", "school_b")

cal_mg <- core$sim_ordcorr_mg_calibrate(
  list(target_g1, target_g2),
  list(marginals_g1, marginals_g2),
  group_labels = labels,
  metric = "pearson_codes"
)
cal_g1 <- core$sim_ordcorr_calibrate(
  target_g1, marginals_g1, metric = "pearson_codes")
cal_g2 <- core$sim_ordcorr_calibrate(
  target_g2, marginals_g2, metric = "pearson_codes")
summary_g1 <- core$sim_ordcorr_summary_calibrate(
  cal_g1$latent_corr, cal_g1$kinds, cal_g1$thresholds,
  metric = "pearson_codes")
summary_mg <- core$sim_ordcorr_mg_summary_calibrate(
  list(cal_g1$latent_corr, cal_g2$latent_corr),
  cal_g1$kinds,
  list(cal_g1$thresholds, cal_g2$thresholds),
  group_labels = labels,
  metric = "pearson_codes"
)

close_matrix <- function(x, y, tol = 1e-12) {
  max(abs(x - y)) <= tol
}

stopifnot(
  inherits(cal_mg, "magmaan_ordcorr_mg_calibration"),
  identical(cal_mg$group_labels, labels),
  identical(cal_mg$variable_names, names(marginals_g1)),
  close_matrix(cal_mg$groups[[1]]$latent_corr, cal_g1$latent_corr),
  close_matrix(cal_mg$groups[[2]]$latent_corr, cal_g2$latent_corr),
  close_matrix(cal_mg$groups[[1]]$achieved_corr, cal_g1$achieved_corr),
  close_matrix(cal_mg$groups[[2]]$achieved_corr, cal_g2$achieved_corr),
  close_matrix(summary_g1$latent_corr, cal_g1$latent_corr),
  close_matrix(summary_g1$target_corr, cal_g1$latent_corr),
  summary_g1$max_abs_error == 0,
  inherits(summary_mg, "magmaan_ordcorr_mg_calibration"),
  identical(summary_mg$group_labels, labels),
  close_matrix(summary_mg$groups[[1]]$latent_corr, cal_g1$latent_corr),
  close_matrix(summary_mg$groups[[2]]$latent_corr, cal_g2$latent_corr)
)

draw_mg <- core$sim_ordcorr_mg_draw(
  cal_mg, n = c(80L, 95L), reps = 1L, seed_base = 20260613)
draw_g1 <- core$sim_ordcorr_draw(
  cal_g1, n = 80L, reps = 1L, seed_base = 20260613)

g1 <- draw_mg$draws[[1]]$groups[[1]]
g2 <- draw_mg$draws[[1]]$groups[[2]]
stopifnot(
  identical(draw_mg$group_labels, labels),
  nrow(g1$X) == 80L,
  nrow(g2$X) == 95L,
  close_matrix(g1$X, draw_g1$draws[[1]]$X, tol = 0),
  length(g1$category_proportions) == 3L,
  length(g2$category_proportions) == 3L,
  length(g1$category_proportions[[1]]) == 3L,
  is.null(g1$category_proportions[[2]]),
  length(g1$category_proportions[[3]]) == 2L,
  abs(sum(g1$category_proportions[[1]]) - 1) < 1e-12,
  abs(sum(g2$category_proportions[[3]]) - 1) < 1e-12
)

batch <- core$sim_ordcorr_mg_batch(
  list(target_g1, target_g2),
  list(marginals_g1, marginals_g2),
  n = 25L,
  reps = 1L,
  seed_base = 20260614,
  group_labels = labels,
  metric = "pearson_codes"
)
stopifnot(
  identical(batch$group_labels, labels),
  nrow(batch$draws[[1]]$groups[[1]]$X) == 25L,
  nrow(batch$draws[[1]]$groups[[2]]$X) == 25L
)

cat("sim_ordcorr_mg_* workflow: ok\n")
