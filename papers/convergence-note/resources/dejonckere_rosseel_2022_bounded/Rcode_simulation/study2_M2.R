# Using bounded estimation to avoid nonconvergence in small sample
# structural equation modeling
#
# Julie De Jonckere and Yves Rosseel
# Department of Data Analysis, Ghent University
#
# simulation script model 'M2' study 2 

library(lavaan)

pop.model2 <- '

# factor loadings
eta1 =~ 1*y1 + 1*y2 + 1*y3 + 0.3*y4
eta2 =~ 1*y4 + 1*y5 + 1*y6 + 0.3*y7
eta3 =~ 1*y7 + 1*y8 + 1*y9 + 0.3*y6

# latent (residual) variances
eta1 ~~ 0.49*eta1
eta2 ~~ 0.3136*eta2
eta3 ~~ 0.3136*eta3

# regression part
eta2 ~ 0.6*eta1
eta3 ~ 0.6*eta2

# residual variances
y1 ~~ 0.51*y1 
y2 ~~ 0.51*y2 
y3 ~~ 0.51*y3 
y4 ~~ 0.2895*y4 
y5 ~~ 0.51*y5 
y6 ~~ 0.2895*y6
y7 ~~ 0.2895*y7
y8 ~~ 0.51*y8
y9 ~~ 0.51*y9
'

# two omitted cross-loadings
model2 <- '
# factor loadings
eta1 =~ y1 + start(1)*y2 + start(1)*y3 + start(0.3)*y4
eta2 =~ y4 + start(1)*y5 + start(1)*y6 + 0*y7
eta3 =~ y7 + start(1)*y8 + start(1)*y9 + 0*y6

# regression part
eta2 ~ start(0.6)*eta1 
eta3 ~ start(0.6)*eta2

# factor variances
eta1 ~~ eta1 + lower(-Inf)*eta1 + upper(+Inf)*eta1 + start(0.51)*eta1
eta2 ~~ eta2 + lower(-Inf)*eta2 + upper(+Inf)*eta2 + start(0.3136)*eta2
eta3 ~~ eta3 + lower(-Inf)*eta3 + upper(+Inf)*eta3 + start(0.3136)*eta3

# observed residual variances
y1 ~~ y1 + start(0.51)*y1
y2 ~~ y2 + start(0.51)*y2
y3 ~~ y3 + start(0.51)*y3
y4 ~~ y4 + start(0.2895)*y4
y5 ~~ y5 + start(0.51)*y5
y6 ~~ y6 + start(0.2895)*y6
y7 ~~ y7 + start(0.2895)*y7
y8 ~~ y8 + start(0.51)*y8
y9 ~~ y9 + start(0.51)*y9
'


FIT <- sem(model2)              # model zonder bounds 
OV.NAMES <- lavNames(FIT, "ov") # observed variables
LV.NAMES <- lavNames(FIT, "lv") # latent variables
COEF.NAMES <- names(coef(FIT))  # (free) parameter names
NPAR <- length(COEF.NAMES)      # amount of parameters
NVAR <- length(OV.NAMES)        # amount of observed variables

PTE <- parTable(FIT)
ov.var.idx <- which(PTE$op == "~~" & PTE$lhs %in% OV.NAMES & PTE$lhs == PTE$rhs) # lines that correspond to observed variables in parTable
lv.var.idx <- which(PTE$op == "~~" & PTE$lhs %in% LV.NAMES & PTE$lhs == PTE$rhs) # lines that correspond to latent variables in parTable
beta.idx <- which(PTE$op == "~") # beta estimate

COEF.NAMES.ALL <- c("eta1=~y1", "eta1=~y2", "eta1=~y3", "eta1=~y4", "eta2=~y4",  # parameter names (free + fixed)
                    "eta2=~y5", "eta2=~y6", "eta2=~y7", "eta3=~y7", "eta3=~y8",
                    "eta3=~y9", "eta3=~y6",   "eta2~eta1", "eta3~eta2",
                    "eta1~~eta1", "eta2~~eta2", "eta3~~eta3", "y1~~y1",
                    "y2~~y2", "y3~~y3", "y4~~y4", "y5~~y5", "y6~~y6", "y7~~y7",
                    "y8~~y8", "y9~~y9")
SAMPLE.SIZES <- c(10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100)
NSIZES <- length(SAMPLE.SIZES)
REP <- 10000

# FIT containers
col.names <- c(COEF.NAMES.ALL,
               paste(COEF.NAMES.ALL, ".lb", sep = ""),
               paste(COEF.NAMES.ALL, ".ub", sep = ""),
               paste(COEF.NAMES.ALL, ".lba", sep = ""),
               paste(COEF.NAMES.ALL, ".uba", sep = ""),
               "betaE1E2.se", "betaE2E3.se", "optim", "se",
               "ov.var.pos", "lv.var.pos")
NRESULTS <- length(col.names)

# internal function to extract the needed results from a fitted
# lavaan object
extract_results_from_fit_MIS <- function(fit) {

  OPTIM <- SE <- OV.VAR.POS <- LV.VAR.POS <- 0L; BETA.SE <- NA

  PT <- partable(fit)
  THETA <- PT$est
  LB    <- PT$lower
  UB    <- PT$upper
  ALB   <- THETA == LB
  AUB   <- THETA == UB
  BETA.SE <- PT$se[ beta.idx ]

  if(lavInspect(fit, "converged")) {
    OPTIM <- 1L
  }
  if(!all(is.na(PT$se[PT$free > 0]))) {
    SE <- 1L
  }

  if(all(PT$est[ov.var.idx] >= 0)) {
    OV.VAR.POS <- 1
  }

  if(all(PT$est[lv.var.idx] >= 0)) {
    LV.VAR.POS <- 1
  }

  c(THETA, LB, UB, ALB, AUB,
    BETA.SE, OPTIM, SE, OV.VAR.POS, LV.VAR.POS)

}

ML   <- matrix(as.numeric(NA), nrow = REP*NSIZES, ncol = 136L)
colnames(ML) <- col.names

WIDE <- matrix(as.numeric(NA), nrow = REP*NSIZES, ncol = 136L)
colnames(WIDE) <- col.names

STAND <- matrix(as.numeric(NA), nrow = REP*NSIZES, ncol = 136L)
colnames(STAND) <- col.names

OVVAR <- matrix(as.numeric(NA), nrow = REP*NSIZES, ncol = 136L)
colnames(OVVAR) <- col.names


for(ss in 1:length(SAMPLE.SIZES)) {
  N <- SAMPLE.SIZES[ss]

  for(i in seq_len(REP)) {

    cat("sample size = ", sprintf("%4d", N), " rep = ",
        sprintf("%4d", i), "\n")

    this.seed <- 120000 + i + (ss - 1)*REP
    set.seed(this.seed)

    # row number in container
    idx <- i + (ss - 1)*REP

    # generate data
    Data <- simulateData(pop.model2, sample.nobs = N, empirical = FALSE)


    # ML
    fit <- try(sem(model2, data = Data, estimator = "ML",
                                baseline = FALSE, h1 = FALSE,
                                check.post = FALSE, optim.attempts = 1L),
                         silent = TRUE)
    if(!inherits(fit, "try-error")) {
        ML[idx, ] <- extract_results_from_fit_MIS(fit)
    }

    # OVVAR
    fit <- try(sem(model2, data = Data, estimator = "ML",
                                baseline = FALSE, h1 = FALSE,
                                check.post = FALSE, optim.attempts = 1L,
                                optim.bounds = list(lower = "ov.var")),
                         silent = TRUE)
    if(!inherits(fit, "try-error")) {
        OVVAR[idx, ] <- extract_results_from_fit_MIS(fit)
    }

    # standard
    fit <- try(sem(model2, data = Data, estimator = "ML",
                                baseline = FALSE, h1 = FALSE,
                                check.post = FALSE, optim.attempts = 1L,
                                optim.bounds =
                                list(upper = c("ov.var", "lv.var", "loadings"),
                                     lower = c("ov.var", "lv.var", "loadings"),
                                     min.reliability.marker = 0.1,
                                     min.var.lv.endo = 0.005)),
                         silent = TRUE)
    if(!inherits(fit, "try-error")) {
        STAND[idx, ] <- extract_results_from_fit_MIS(fit)
    }

    # wide
    fit <- try(sem(model2, data = Data, estimator = "ML",
                                baseline = FALSE, h1 = FALSE,
                                check.post = FALSE, optim.attempts = 1L,
                                optim.bounds =
                                list(upper = c("ov.var", "lv.var", "loadings"),
                                     lower = c("ov.var", "lv.var", "loadings"),
                                     lower.factor = c(1.05, 1.0, 1.1),
                                     upper.factor = c(1.20, 1.3, 1.1),
                                     min.reliability.marker = 0.1,
                                     min.var.lv.endo = 0.005)),
                         silent = TRUE)
    if(!inherits(fit, "try-error")) {
        WIDE[idx, ] <- extract_results_from_fit_MIS(fit)
    }

  } # i (rep)
} # ss (N)

save(ML,    file = "M2ML.RData")
save(OVVAR, file = "M2OVVAR.RData")
save(STAND, file = "M2STAND.RData")
save(WIDE,  file = "M2WIDE.RData")

