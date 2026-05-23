## ---------------------------
##
## Script name:          00_Pre_Simulation_Study2.R
##
## Purpose of script:    Script to define population and analysis model, 
##                       parameter values from output, extract function for data    
##
## ---------------------------
## Notes:
##      1. Model specification
##      2. Utilities
##      3. Data extract function
##      4. First sim run to define convergence
## ---------------------------

## -------------------------------------------------------------------------- ##
##                          1. Model Specification                            ##      
## -------------------------------------------------------------------------- ##

## -------------------
## POPULATION MODEL 
## -------------------

pop.model.3LV <- '

  # factor loadings
    eta1 =~ 1*y1 + 1*y2 + 1*y3 + 0.3*y4
    eta2 =~ 1*y4 + 1*y5 + 1*y6 + 0.3*y7
    eta3 =~ 1*y7 + 1*y8 + 1*y9 + 0.3*y6

  # latent (residual) variances
    eta1 ~~ 0.49*eta1
    eta2 ~~ 0.3136*eta2
    eta3 ~~ 0.3136*eta3

  # regression part
    eta2 ~ 0.1*eta1
    eta3 ~ 0.2*eta2

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

## -------------------
## MODEL for analysis
## -------------------

model.3LV <- '

  # factor loadings
    eta1 =~ y1 + y2 + y3 + y4
    eta2 =~ y4 + y5 + y6 + y7
    eta3 =~ y7 + y8 + y9 + y6

  # latent (residual) variances
    eta1 ~~ eta1
    eta2 ~~ eta2
    eta3 ~~ eta3

  # regression part
    eta2 ~ eta1
    eta3 ~ eta2

  # residual variances
    y1 ~~ y1 
    y2 ~~ y2 
    y3 ~~ y3 
    y4 ~~ y4 
    y5 ~~ y5 
    y6 ~~ y6
    y7 ~~ y7
    y8 ~~ y8
    y9 ~~ y9
'

## -------------------------------------------------------------------------- ##
##                               2. Utilities                                 ##      
## -------------------------------------------------------------------------- ##

# parameter values
FIT_C      <- lavaan::sem(model.3LV)               
OV.NAMES_C <- lavNames(FIT_C, "ov") # observed variables
LV.NAMES_C <- lavNames(FIT_C, "lv") # latent variables
COEF.NAMES_C    <- names(coef(FIT_C))  # (free) parameter names

col.names_c     <- c(COEF.NAMES_C, 
                     "beta.se1", 
                     "beta.se2",
                     "optim", 
                     "se", 
                     "ov.var.pos", 
                     "lv.var.pos")

NPAR_C <- length(COEF.NAMES_C)      
NVAR_C <- length(OV.NAMES_C) 

n_cov_elements_c <- NVAR_C * (NVAR_C + 1) / 2  
n_eig_values_c <- NVAR_C                     
NDATA_C <- n_cov_elements_c + n_eig_values_c

PTE_C <- parTable(FIT_C)
ov.var.idx_c <- which(PTE_C$op == "~~" & PTE_C$lhs %in% OV.NAMES_C 
                      & PTE_C$lhs == PTE_C$rhs) 
lv.var.idx_c <- which(PTE_C$op == "~~" & PTE_C$lhs %in% LV.NAMES_C 
                      & PTE_C$lhs == PTE_C$rhs) 
beta.idx_c <- which(PTE_C$op == "~")

NRESULTS_C <- length(col.names_c)
REP <- 1000
N   <- ss <- 20


## -------------------------------------------------------------------------- ##
##                        3. Data extract function                            ##      
## -------------------------------------------------------------------------- ##

extract_results_from_fit_C <- function(fit) {
  
  OPTIM <- SE <- OV.VAR.POS <- LV.VAR.POS <- 0L; BETA.SE <- NA
  
  PT <- partable(fit)
  THETA <- PT$est[   PT$free > 0L ]
  BETA.SE <- PT$se[ beta.idx_c ]
  
  if(lavInspect(fit, "converged")) {
    OPTIM <- 1L
  }
  if(!all(is.na(PT$se[PT$free > 0]))) {
    SE <- 1L
  }
  
  if(all(PT$est[ov.var.idx_c] >= 0)) {
    OV.VAR.POS <- 1
  }
  
  if(all(PT$est[lv.var.idx_c] >= 0)) {
    LV.VAR.POS <- 1
  }
  
  c(THETA, BETA.SE, OPTIM, SE, OV.VAR.POS, LV.VAR.POS)
  
}


## -------------------------------------------------------------------------- ##
##                   4. Simulation convergence check                          ##      
## -------------------------------------------------------------------------- ##

## -------------------
## DATA CONTAINERS   
## -------------------

RUN_C                   <- matrix(as.numeric(NA), nrow = REP, ncol = 3L)
colnames(RUN_C)         <- c("size", "rep", "seed")


DATA_C                  <- matrix(as.numeric(NA), nrow = REP, ncol = NDATA_C) 


ML_COMPLEX              <- matrix(as.numeric(NA), nrow = REP, ncol = NRESULTS_C)
colnames(ML_COMPLEX)    <- col.names_c


## -------------------
## SIMULATION START
## -------------------


for(i in seq_len(REP)){
  
  cat("sample size = ", sprintf("%4d", N), " rep = ", 
      sprintf("%4d", i), "\n")
  
  this.seed <- i + (ss - 1)*REP
  set.seed(this.seed)
  
  # generate data
  Data      <- simulateData(pop.model.3LV, sample.nobs = N, empirical = FALSE)
  
  COV         <- cov(Data)
  EV          <- eigen(COV)$values
  DATA_C[i, ] <- c(lav_matrix_vech(COV), EV)
  
  fit.lavaan <- try(sem(model.3LV, 
                        data = Data, 
                        estimator = "ML",
                        baseline = FALSE, 
                        h1 = FALSE,
                        check.post = FALSE, 
                        optim.attempts = 1L),
                    silent = TRUE)
  if(!inherits(fit.lavaan, "try-error")){
    ML_COMPLEX[i, ]  <- extract_results_from_fit_C(fit.lavaan)
  }
  
}
save(DATA_C,      file = "DATA_C.RData")
save(ML_COMPLEX,  file = "ML_COMPLEX.RData")


## ----------------------------------END------------------------------------- ##