# Data sets and the model catalogue for the recovery section.

ernst_data <- function(n, seed, beta = 0.4) {
  set.seed(seed)
  lam <- c(1, 0.8, 0.6)
  X <- rnorm(n)
  Y <- beta * X + rnorm(n)
  d <- cbind(outer(X, lam), outer(Y, lam)) + matrix(rnorm(6 * n), n)
  colnames(d) <- c(paste0("x", 1:3), paste0("y", 1:3))
  as.data.frame(d)
}

# MCAR cells on `cols`, never emptying a row.
mcar <- function(d, cols, rate, seed) {
  set.seed(seed)
  for (v in cols) d[[v]][runif(nrow(d)) < rate] <- NA
  empty <- rowSums(!is.na(d[cols])) == 0L
  if (any(empty)) d[empty, cols[1L]] <- 0
  d
}

exp_datasets <- function() {
  hs <- lavaan::HolzingerSwineford1939[c("school", "ageyr", "sex", paste0("x", 1:9))]
  pd <- lavaan::PoliticalDemocracy
  gr <- lavaan::Demo.growth[c("t1", "t2", "t3", "t4")]
  er <- ernst_data(200, seed = 87001)
  list(
    hs = hs, pd = pd, growth = gr, ernst = er,
    hs_miss = mcar(hs, paste0("x", 1:9), 0.08, 87002),
    pd_miss = mcar(pd, names(pd), 0.06, 87003),
    growth_miss = mcar(gr, names(gr), 0.08, 87004),
    ernst_miss = mcar(er, names(er), 0.08, 87005))
}

hs3 <- "visual =~ x1 + x2 + x3
textual =~ x4 + x5 + x6
speed =~ x7 + x8 + x9"

hs_with <- function(visual = "visual =~ x1 + x2 + x3",
                    textual = "textual =~ x4 + x5 + x6",
                    speed = "speed =~ x7 + x8 + x9", extra = NULL) {
  paste(c(visual, textual, speed, extra), collapse = "\n")
}

pd_labels <- "ind60 =~ x1 + x2 + x3
dem60 =~ y1 + a*y2 + b*y3 + c*y4
dem65 =~ y5 + a*y6 + b*y7 + c*y8
dem60 ~ ind60
dem65 ~ ind60 + dem60
y1 ~~ y5
y2 ~~ y4 + y6
y3 ~~ y7
y4 ~~ y8
y6 ~~ y8"

# lavaan frees the group-2+ latent means when intercepts are held equal;
# magmaan's group_equal does not (see the report), so the scalar-type cases
# free them explicitly and match lavaan's model exactly.
hs3_means <- paste0(hs3, "\nvisual ~ c(0, NA)*1\ntextual ~ c(0, NA)*1\nspeed ~ c(0, NA)*1")

ernst_sem <- "X =~ x1 + x2 + x3\nY =~ y1 + y2 + y3\nY ~ X"
ernst_cfa <- "X =~ x1 + x2 + x3\nY =~ y1 + y2 + y3\nX ~~ Y"

# A case: syntax, data key, model options (magmaan names), estimators to run,
# and the expected gauge classification (latent names, pooled over blocks).
mk_case <- function(id, family, data, syntax, units, pass = character(),
                    opts = list(), estimators = "ML", groups = NULL,
                    optimizer = NULL, label = id) {
  list(id = id, family = family, data = data, syntax = syntax, units = units,
       pass = pass, opts = opts, estimators = estimators, groups = groups,
       optimizer = optimizer, label = label)
}

all_cases <- function() {
  core3 <- c("visual", "textual", "speed")
  cases <- list(
    # Identification conventions.
    mk_case("hs_marker", "identification", "hs", hs3, core3,
            estimators = c("ML", "ULS", "GLS", "WLS", "FIML", "PSD"),
            label = "HS 3-factor, marker"),
    mk_case("hs_stdlv", "identification", "hs", hs3, core3,
            opts = list(std_lv = TRUE),
            estimators = c("ML", "ULS", "GLS", "FIML", "PSD"),
            label = "HS 3-factor, std.lv"),
    mk_case("hs_effect", "identification", "hs", hs3, core3,
            opts = list(effect_coding = TRUE),
            estimators = c("ML", "ULS", "GLS", "FIML"),
            label = "HS 3-factor, effect coding"),
    mk_case("hs_marker2", "identification", "hs",
            hs_with("visual =~ NA*x1 + 1*x2 + x3", "textual =~ NA*x4 + 1*x5 + x6",
                    "speed =~ NA*x7 + 1*x8 + x9"), core3,
            label = "HS 3-factor, marker on the 2nd indicator"),
    mk_case("hs_fixed_nonunit", "identification", "hs",
            hs_with("visual =~ 2*x1 + x2 + x3"), core3,
            label = "Marker fixed at 2"),
    mk_case("hs_lvvar2", "identification", "hs",
            hs_with("visual =~ NA*x1 + x2 + x3", extra = "visual ~~ 2*visual"), core3,
            label = "Latent variance fixed at 2"),
    mk_case("hs_orthogonal", "identification", "hs", hs3, core3,
            opts = list(orthogonal = TRUE), label = "Orthogonal factors"),
    mk_case("hs_bounded", "identification", "hs", hs3, character(), core3,
            opts = list(bounds = "standard"), label = "Bounded estimation (standard)"),

    # Measurement invariance ladder (HS by school).
    mk_case("hs_configural", "invariance", "hs", hs3, core3, groups = "school",
            estimators = c("ML", "FIML"), label = "Configural"),
    mk_case("hs_metric", "invariance", "hs", hs3, core3, groups = "school",
            opts = list(group_equal = "loadings"),
            estimators = c("ML", "ULS", "GLS", "FIML"), label = "Metric"),
    mk_case("hs_metric_stdlv", "invariance", "hs", hs3, core3, groups = "school",
            opts = list(group_equal = "loadings", std_lv = TRUE),
            label = "Metric, std.lv"),
    mk_case("hs_scalar", "invariance", "hs", hs3_means, core3, groups = "school",
            opts = list(group_equal = c("loadings", "intercepts"), meanstructure = TRUE),
            estimators = c("ML", "FIML"), label = "Scalar"),
    mk_case("hs_strict", "invariance", "hs", hs3_means, core3, groups = "school",
            opts = list(group_equal = c("loadings", "intercepts", "residuals"),
                        meanstructure = TRUE), label = "Strict"),
    mk_case("hs_partial", "invariance", "hs", hs3_means, c("textual", "speed"), "visual",
            groups = "school",
            opts = list(group_equal = c("loadings", "intercepts"),
                        group_partial = "visual=~x2", meanstructure = TRUE),
            estimators = c("ML", "FIML"), label = "Partial scalar (visual=~x2 free)"),

    # Equality and constraint syntax.
    mk_case("hs_equal_within", "constraints", "hs",
            hs_with("visual =~ x1 + a*x2 + a*x3"), core3,
            label = "Equal loadings within a factor"),
    mk_case("hs_two_fixed", "constraints", "hs",
            hs_with(textual = "textual =~ 1*x4 + 1*x5 + x6"), core3,
            label = "Two loadings fixed at 1"),
    mk_case("hs_tau_labels", "constraints", "hs",
            hs_with(textual = "textual =~ 1*x4 + a*x5 + a*x6"), core3,
            label = "Marker plus equal free loadings"),
    mk_case("hs_tau_fixed", "constraints", "hs",
            hs_with(speed = "speed =~ 1*x7 + 1*x8 + 1*x9"), c("visual", "textual"),
            "speed", label = "All loadings fixed (tau-equivalent)"),
    mk_case("hs_ratio", "constraints", "hs",
            hs_with("visual =~ x1 + a*x2 + b*x3", extra = "b == 1.5*a"), core3,
            label = "Loading ratio b == 1.5*a"),
    mk_case("hs_inhom", "constraints", "hs",
            hs_with("visual =~ x1 + a*x2 + b*x3", extra = "a + b == 1.2"), core3,
            label = "Inhomogeneous loading sum a + b == 1.2"),
    mk_case("hs_nonlinear", "constraints", "hs",
            hs_with("visual =~ x1 + a*x2 + b*x3", extra = "b == a^2 + 0.4"),
            c("textual", "speed"), "visual", optimizer = "nlopt-slsqp",
            label = "Nonlinear loading constraint"),
    mk_case("hs_equal_lv_var", "constraints", "hs",
            hs_with(extra = c("visual ~~ v*visual", "textual ~~ v*textual")),
            "speed", c("visual", "textual"), label = "Equal latent variances"),
    mk_case("hs_equal_reg", "constraints", "hs",
            hs_with(extra = "speed ~ b*visual + b*textual"), character(),
            core3, label = "Equal regression slopes"),
    mk_case("hs_marker_fixedvar", "constraints", "hs",
            hs_with(extra = "visual ~~ 1*visual"), c("textual", "speed"), "visual",
            label = "Marker plus fixed variance"),
    mk_case("hs_fixed_cov_stdlv", "constraints", "hs",
            hs_with(extra = "visual ~~ 0.4*textual"), "speed",
            c("visual", "textual"), opts = list(std_lv = TRUE),
            label = "std.lv with a fixed covariance 0.4"),

    # Structure.
    mk_case("hs_crossload", "structure", "hs",
            hs_with("visual =~ x1 + x2 + x3 + x9"), core3, label = "Cross-loading"),
    mk_case("hs_higher", "structure", "hs",
            hs_with(extra = "g =~ visual + textual + speed"), c("textual", "speed"),
            c("visual", "g"), label = "Second-order factor"),
    mk_case("hs_single", "structure", "hs",
            "visual =~ x1 + x2 + x3\ntx =~ x4\ntx ~ visual", "visual", "tx",
            label = "Single-indicator latent"),
    mk_case("hs_mimic", "structure", "hs",
            "visual =~ x1 + x2 + x3\nvisual ~ ageyr + sex", "visual",
            label = "MIMIC"),
    mk_case("hs_mediation", "structure", "hs",
            hs_with(extra = c("textual ~ a*visual", "speed ~ b*textual + c*visual",
                              "ind := a*b", "tot := c + a*b")), core3,
            estimators = c("ML", "FIML"), label = "Latent mediation with := "),
    mk_case("pd_labels", "structure", "pd", pd_labels, "ind60", c("dem60", "dem65"),
            estimators = c("ML", "ULS", "GLS", "FIML"),
            label = "PoliticalDemocracy, cross-factor equal loadings"),
    mk_case("pd_labels_stdlv", "structure", "pd", pd_labels, "ind60",
            c("dem60", "dem65"), opts = list(std_lv = TRUE),
            label = "PoliticalDemocracy, equal loadings, std.lv"),
    mk_case("pd_free", "structure", "pd", gsub("[abc]\\*", "", pd_labels),
            c("ind60", "dem60", "dem65"), estimators = c("ML", "PSD"),
            label = "PoliticalDemocracy, free loadings"),
    mk_case("growth", "structure", "growth",
            "i =~ 1*t1 + 1*t2 + 1*t3 + 1*t4\ns =~ 0*t1 + 1*t2 + 2*t3 + 3*t4",
            character(), c("i", "s"), opts = list(model_type = "growth"),
            estimators = c("ML", "FIML"), label = "Linear growth"),
    mk_case("growth_basis", "structure", "growth",
            "i =~ 1*t1 + 1*t2 + 1*t3 + 1*t4\ns =~ 0*t1 + 1*t2 + t3 + t4",
            "s", "i", opts = list(model_type = "growth"),
            estimators = c("ML", "FIML"), label = "Latent-basis growth"),
    mk_case("ernst_sem", "structure", "ernst", ernst_sem, c("X", "Y"),
            estimators = c("ML", "ULS", "GLS", "WLS", "FIML", "PSD"),
            label = "Ernst two-factor SEM (N = 200)"),
    mk_case("ernst_cfa", "structure", "ernst", ernst_cfa, c("X", "Y"),
            estimators = c("ML", "PSD"), label = "Ernst two-factor CFA (N = 200)"))
  names(cases) <- vapply(cases, `[[`, "", "id")
  cases
}

smoke_case_ids <- c("hs_marker", "hs_metric", "hs_scalar", "hs_partial", "pd_labels",
                    "hs_nonlinear", "hs_mediation", "ernst_sem")
