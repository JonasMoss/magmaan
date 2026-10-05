# Gates the ordinary rows of the primary capability inventory
# (project/validation/capabilities.md#primary-inventory-020): each estimator
# and covariance policy computes exactly the components the inventory lists,
# and every other component carries the listed typed reason.

inventory_states <- function(fit) {
  s <- fit$inference$status
  setNames(ifelse(s$available, "computed", s$reason), s$component)
}

nested_states <- function(f0, f1) {
  a <- anova(f0, f1)
  out <- setNames(ifelse(is.finite(a$statistic), "computed", "unavailable"),
                  c("score", "lr"))
  out[c("lr", "score")]
}

restricted_cfa <- "visual =~ x1 + a*x2 + a*x3\ntextual =~ x4 + x5 + x6"

test_that("the inventory's ordinary policy rows match each estimator's components", {
  d <- hs()
  incomplete <- d
  set.seed(1)
  incomplete$x2[runif(nrow(d)) < plogis(-1.5 + 0.8 * scale(d$x1)[, 1])] <- NA
  ord <- ordinal_hs()
  mixed <- d
  for (v in paste0("x", 1:3)) mixed[[v]] <- ord[[v]]
  all3 <- function(reason) c(covariance = reason, global_score = reason, global_lr = reason)
  rows <- list(
    list("ML", d, NULL, c(covariance = "computed", global_score = "computed", global_lr = "computed")),
    list("ML", incomplete, NULL, c(covariance = "computed", global_score = "computed", global_lr = "computed")),
    list("DWLS", ord, paste0("x", 1:6), c(covariance = "computed", global_score = "computed", global_lr = "inapplicable")),
    list("FIML", incomplete, NULL, all3("computed")),
    list("ML2S", incomplete, NULL, all3("unsupported_model")),
    list("GLS", d, NULL, all3("unsupported_model")),
    list("ULS", d, NULL, all3("unsupported_model")),
    list("ULS", ord, paste0("x", 1:6), all3("unsupported_model")),
    list("WLS", ord, paste0("x", 1:6), all3("unsupported_model")),
    list("DWLS", mixed, paste0("x", 1:3), c(covariance = "computed", global_score = "computed", global_lr = "inapplicable"))
  )
  for (r in rows) {
    label <- paste(r[[1]], if (is.null(r[[3]])) "continuous" else paste(length(r[[3]]), "ordered"))
    model <- if (is.null(r[[3]])) magmaan_model(cfa, prototype = r[[2]]) else
      magmaan_model(cfa, prototype = r[[2]], ordered = r[[3]])
    fit <- magmaan(model, r[[2]], estimator = r[[1]])
    expect_true(isTRUE(as_lab_fit(fit)$converged), label = label)
    expect_identical(inventory_states(fit)[names(r[[4]])], r[[4]], label = label)
    if (r[[1]] == "DWLS" && all(r[[4]][1:2] == "computed")) {
      expect_identical(fit$inference$global_score$reference, "all")
      expect_true(is.finite(fit$inference$global_score$p_all))
    }
    if (all(r[[4]] == "computed")) {
      expect_identical(fit$inference$global_score$reference, "sb_peba4")
      expect_identical(fit$inference$global_lr$reference, "sb_peba4")
    }
    if (all(r[[4]] == "unsupported_model")) {
      expect_error(vcov(fit), class = "magmaan_inference_unavailable")
      expect_error(confint(fit), class = "magmaan_inference_unavailable")
    }
  }
  expect_error(magmaan(cfa, d, estimator = "WLS"), "magmaanlab::estimate")
  expect_error(magmaan(magmaan_model(cfa, prototype = mixed, ordered = paste0("x", 1:3)),
                       mixed, estimator = "ULS"), "mixed")
})

test_that("the inventory's nested rows match ML, DWLS and the unsupported setups", {
  d <- hs()
  ord <- ordinal_hs()
  for (case in list(list("ML", d, NULL, c(lr = "computed", score = "computed")),
                    list("FIML", d, NULL, c(lr = "computed", score = "computed")),
                    list("DWLS", ord, paste0("x", 1:6), c(lr = "computed", score = "unavailable")),
                    list("GLS", d, NULL, c(lr = "unavailable", score = "unavailable")))) {
    mk <- function(syntax) if (is.null(case[[3]])) magmaan_model(syntax, prototype = case[[2]]) else
      magmaan_model(syntax, prototype = case[[2]], ordered = case[[3]])
    f1 <- magmaan(mk(cfa), case[[2]], estimator = case[[1]])
    f0 <- magmaan(mk(restricted_cfa), case[[2]], estimator = case[[1]])
    expect_identical(nested_states(f0, f1), case[[4]], label = case[[1]])
  }
})

test_that("the inventory's domain rows: PSD computes, barrier is penalized everywhere", {
  d <- hs()
  for (g in list(NULL, "school")) {
    model <- if (is.null(g)) magmaan_model(cfa, prototype = d) else
      magmaan_model(cfa, prototype = d, group = g)
    psd <- magmaan(model, d, covariance = "psd")
    expect_identical(unname(inventory_states(psd)), rep("computed", 3L))
    penalized <- suppressMessages(magmaan(model, d, covariance = barrier(0.25)))
    expect_identical(unname(inventory_states(penalized)), rep("penalized", 3L))
    zero <- magmaan(model, d, covariance = barrier(0))
    expect_equal(coef(zero), coef(magmaan(model, d)))
  }
})
