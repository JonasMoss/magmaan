# Using bounded estimation to avoid nonconvergence in small sample
# structural equation modeling
#
# Julie De Jonckere and Yves Rosseel
# Department of Data Analysis, Ghent University
#
# simulation script model study 1

library(lavaan) # 0.6-9

## Define the population model
pop.model <- '
    # factor loadings
    Y =~ 1*y1 + 0.8*y2 + 0.6*y3
    X =~ 1*x1 + 0.8*x2 + 0.6*x3

    # regression part
    Y ~ 0.25*X
'

## Model without (active) bounds
model <- '
    # factor loadings
    Y =~ y1 + start(1)*y2 + start(1)*y3
    X =~ x1 + start(1)*x2 + start(1)*x3

    # regression part
    Y ~ start(0)*X

    # factor variances
    Y ~~ Y + lower(-Inf)*Y + upper(+Inf)*Y + start(1)*Y
    X ~~ X + start(1)*X

    # observed residual variances
    y1 ~~ start(1)*y1 
    y2 ~~ start(1)*y2 
    y3 ~~ start(1)*y3 
    x1 ~~ start(1)*x1 
    x2 ~~ start(1)*x2 
    x3 ~~ start(1)*x3 
'

FIT <- sem(model)               # model zonder bounds 
OV.NAMES <- lavNames(FIT, "ov") # observed variables
LV.NAMES <- lavNames(FIT, "lv") # latent variables
COEF.NAMES <- names(coef(FIT))  # (free) parameter names
NPAR <- length(COEF.NAMES)      # amount of parameters
NVAR <- length(OV.NAMES)        # amount of observed variables

PTE <- parTable(FIT)
ov.var.idx <- which(PTE$op == "~~" & PTE$lhs %in% OV.NAMES & PTE$lhs == PTE$rhs) # lines that correspond to observed variables in parTable
lv.var.idx <- which(PTE$op == "~~" & PTE$lhs %in% LV.NAMES & PTE$lhs == PTE$rhs) # lines that correspond to latent variables in parTable
beta.idx <- which(PTE$op == "~") # line corresponding to beta estimate

theta.idx <- which(PTE$free > 0L)

SAMPLE.SIZES <- c(10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100)
#SAMPLE.SIZES <- 10

NSIZES <- length(SAMPLE.SIZES)
REP <- 10000


col.names <- c(COEF.NAMES,
               paste(COEF.NAMES, ".lb", sep = ""),
               paste(COEF.NAMES, ".ub", sep = ""),
               paste(COEF.NAMES, ".lba", sep = ""),
               paste(COEF.NAMES, ".uba", sep = ""),
               "beta.se", "optim", "se", "ov.var.pos", "lv.var.pos")
NRESULTS <- length(col.names)

# internal function to extract the needed results from a fitted
# lavaan object
extract_results_from_fit_MLB <- function(fit) {

  OPTIM <- SE <- OV.VAR.POS <- LV.VAR.POS <- 0L; BETA.SE <- NA

  PT <- fit@ParTable
  THETA <- PT$est[   theta.idx ]
  LB    <- PT$lower[ theta.idx ]
  UB    <- PT$upper[ theta.idx ]
  ALB   <- THETA == LB
  AUB   <- THETA == UB

  if(lavInspect(fit, "converged")) {
    OPTIM <- 1L
  }
  if(!all(is.na(PT$se[theta.idx]))) {
    SE <- 1L
    BETA.SE <- PT$se[ beta.idx ]
  }
  if(all(PT$est[ov.var.idx] >= 0)) {  # include zero(?)
    OV.VAR.POS <- 1
  }
  if(all(PT$est[lv.var.idx] >= 0)) {  # include zero(?)
    LV.VAR.POS <- 1
  }

  c(THETA, LB, UB, ALB, AUB,
    BETA.SE, OPTIM, SE, OV.VAR.POS, LV.VAR.POS)
}

ML <- matrix(as.numeric(NA), nrow = REP*NSIZES, ncol = 70L)
colnames(ML) <- col.names

OVAR000 <- matrix(as.numeric(NA), nrow = REP*NSIZES, ncol = 70L)
colnames(OVAR000) <- col.names

STAND <- matrix(as.numeric(NA), nrow = REP*NSIZES, ncol = 70L)
colnames(STAND) <- col.names

WIDE <- matrix(as.numeric(NA), nrow = REP*NSIZES, ncol = 70L)
colnames(WIDE) <- col.names

for(ss in 1:length(SAMPLE.SIZES)) {
    N <- SAMPLE.SIZES[ss]

    for(i in seq_len(REP)) {

        cat("sample size = ", sprintf("%4d", N), " rep = ", sprintf("%4d", i),
            "\n")
        this.seed <- 50000 + i + (ss - 1)*REP
        set.seed(this.seed)

        # row number in container
        idx <- i + (ss - 1)*REP

        # generate data
        Data <- simulateData(pop.model, sample.nobs = N, empirical = FALSE)


        # 1. no bounds
        fit <- try(sem(model, data = Data, estimator = "ML",
                       baseline = FALSE, h1 = FALSE,
                       check.post = FALSE, optim.attempts = 1L,
                       optim.bounds =  list()),
               silent = TRUE)
        if(!inherits(fit, "try-error")) {
          ML[idx, ] <- extract_results_from_fit_MLB(fit)
        }

        # 2. OVAR000
        fit <- try(sem(model, data = Data, estimator = "ML",
                       baseline = FALSE, h1 = FALSE,
                       check.post = FALSE, optim.attempts = 1L,
                       optim.bounds = list(lower = c("ov.var"))),
               silent = TRUE)
        if(!inherits(fit, "try-error")) {
          OVAR000[idx, ] <- extract_results_from_fit_MLB(fit)
        }

        # 3. standard
        fit <- try(sem(model, data = Data, estimator = "ML",
                       baseline = FALSE, h1 = FALSE,
                       check.post = FALSE, optim.attempts = 1L,
                       optim.bounds =
                           list(upper = c("ov.var", "lv.var", "loadings"),
                                lower = c("ov.var", "lv.var", "loadings"),
                                min.reliability.marker = 0.1,
                                min.var.lv.endo = 0.005)),
                         silent = TRUE)
        if(!inherits(fit, "try-error")) {
            STAND[idx, ] <- extract_results_from_fit_MLB(fit)
        }

        # 4. wide
        fit <- try(sem(model, data = Data, estimator = "ML",
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
            WIDE[idx, ] <- extract_results_from_fit_MLB(fit)
        }

     } # i
} # ss

save(ML,      file = "ML.RData")
save(OVAR000, file = "OVVAR000.RData")
save(STAND,   file = "STANDARD.RData")
save(WIDE,    file = "WIDE.RData")


