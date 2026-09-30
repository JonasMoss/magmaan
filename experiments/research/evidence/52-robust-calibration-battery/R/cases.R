# Case specifications. Each case is a textbook-corpus model (H1, the model whose
# global fit is tested) and, where a non-invariance restriction is natural, a
# nested null H0 = H1 plus linear equality constraints in the same parameter
# slots. Populations are the model-implied moments of H0 (or H1 when there is no
# H0) fitted by lavaan to the book's data, so both hypotheses hold exactly.
# R/build_populations.R turns these specs into populations/<id>.json; the
# simulation reads only the JSON, never the corpus.
#
# disc_pattern picks the 5-point marginal shapes of the discretized condition:
# "alternate" (default) alternates a symmetric and a right-skewed shape over
# items; "skewed" gives every item the right-skewed shape (repeated measures of
# a symptom scale).
#
# rescale = TRUE divides each variable by its pooled population standard
# deviation, the same factor in every group. Both hypotheses are invariant to
# that, so calibration is unchanged; it only repairs conditioning. The source
# variances of mg_path5 span a ratio near 300 (IQ-scale against 0-1 scales),
# enough for robust_nested_lrt's relative rank check on the pooled information
# to reject a well-identified model in about a fifth of the draws.

label_lagged_cus <- function(model, prefix = "cu") {
  lines <- strsplit(model, "\n", fixed = TRUE)[[1]]
  k <- 0L
  for (i in seq_along(lines)) {
    m <- regmatches(lines[i], regexec("^\\s*([A-Za-z0-9_]+)\\s*~~\\s*([A-Za-z0-9_]+)\\s*$", lines[i]))[[1]]
    if (length(m) == 3L && m[2] != m[3] && grepl("_T[0-9]P", m[2]) && grepl("_T[0-9]P", m[3])) {
      k <- k + 1L
      lines[i] <- sprintf("%s ~~ %s%d*%s", m[2], prefix, k, m[3])
    }
  }
  attr_k <- k
  structure(paste(lines, collapse = "\n"), n_labels = attr_k)
}

case_specs <- function() {
  list(
    list(id = "cfa_ptsd8", corpus = "brown_2015_tab8_8_reliability_ptsd",
         type = "CFA, 2 factors, one correlated residual",
         h1 = function(m) m, h0 = NULL),
    list(id = "cfa_mtmm9", corpus = "brown_2015_tab6_3_mtmm_correlated_uniqueness",
         type = "CFA, MTMM correlated uniquenesses",
         h1 = function(m) m, h0 = NULL),
    list(id = "cfa_second12", corpus = "brown_2015_tab8_2_higher_order_coping",
         type = "second-order CFA",
         h1 = function(m) m, h0 = NULL),
    list(id = "cfa_long18", corpus = "little_2013_ch5_tab5_5_configfi_2by3",
         type = "longitudinal CFA, means, effects coding, lagged residual covariances",
         h1 = function(m) as.character(label_lagged_cus(m)),
         h0 = function(h1) paste(h1, paste(sprintf("cu%d == 0", 1:18), collapse = "\n"), sep = "\n"),
         restriction = "no lagged residual covariances (18 constraints)"),
    list(id = "sem_worland11", corpus = "kline_2023_ch15_worland_sr_step2a",
         type = "latent SEM, two structural equations",
         h1 = function(m) {
           m <- sub("Achieve ~ Cognitive + Risk", "Achieve ~ Cognitive + r1*Risk", m, fixed = TRUE)
           sub("Adjust ~ Cognitive + Risk", "Adjust ~ Cognitive + r2*Risk", m, fixed = TRUE)
         },
         h0 = function(h1) paste(h1, "r1 == 0\nr2 == 0", sep = "\n"),
         restriction = "no effect of Risk (2 constraints)"),
    list(id = "growth6", corpus = "newsom_2015_ex7_6c",
         type = "linear growth of CES-D, 6 waves, structured means",
         disc_pattern = "skewed",
         h1 = function(m) paste(m, paste(sprintf("cesd%d ~~ v%d*cesd%d", 1:6, 1:6, 1:6), collapse = "\n"), sep = "\n"),
         h0 = function(h1) paste(h1, paste(sprintf("v%d == v%d", 1:5, 2:6), collapse = "\n"), sep = "\n"),
         restriction = "equal residual variances (5 constraints)"),
    list(id = "mg_path5", corpus = "kline_2023_ch12_lynam_indirect",
         type = "two-group path model, means, cross-group equalities",
         rescale = TRUE,
         h1 = function(m) m,
         h0 = function(h1) paste(h1, "b4 == b5", sep = "\n"),
         restriction = "equal achieve -> delinq path across groups (1 constraint)")
  )
}
