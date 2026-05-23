## ---------------------------
##
## Script name:         SIM_6OV_Study2.R            
##
## Purpose of script:   Simulation script for two target matrices (model-based and
##                      constant correlation target) and standard ML for 6 observed
##                      variables.
##
##
## Authors:             Julie De Jonckere and Yves Rosseel
##                      Department of Data Analysis, Ghent University
##
## ---------------------------
## Notes:
##  - COR (MB) = 0, P = 6, beta = c(0, 0.25, 1, 2)
## ---------------------------

## ------------------------- ##
##          GENERAL          ##
## ------------------------- ##
#library(ltm)
#library(lavaan)
#library(CovTools)

source("Pre_Simulation_Study2.R")

# population models
beta <- c(pop.model6a, pop.model6b, pop.model6c, pop.model6d)
NBETA <- length(beta)


# parameter values
FIT <- sem(model6)               
OV.NAMES <- lavNames(FIT, "ov") # observed variables
LV.NAMES <- lavNames(FIT, "lv") # latent variables
COEF.NAMES <- names(coef(FIT))  # (free) parameter names
col.names <- c(COEF.NAMES, "beta.se", "optim", "se", "ov.var.pos", "lv.var.pos")
col.names.l <- c(COEF.NAMES, "beta.se", "optim", "se", "ov.var.pos", "lv.var.pos", "lambda")
NPAR <- length(COEF.NAMES)      # amount of parameters
NVAR <- length(OV.NAMES) 

PTE <- parTable(FIT)
ov.var.idx <- which(PTE$op == "~~" & PTE$lhs %in% OV.NAMES & PTE$lhs == PTE$rhs) 
lv.var.idx <- which(PTE$op == "~~" & PTE$lhs %in% LV.NAMES & PTE$lhs == PTE$rhs) 
beta.idx <- which(PTE$op == "~") 

SAMPLE.SIZES <- c(10, 20, 30, 40, 50, 100)
NSIZES <- length(SAMPLE.SIZES)
NRESULTS <- length(col.names)
NRESULTS.L <- length(col.names.l)
REP <- 1000

## ------------------------- ##
##         Containers        ##
## ------------------------- ##

# ML
ML_standard_6ov <- matrix(as.numeric(NA), 
                          nrow = REP*NSIZES*NBETA, 
                          ncol = NRESULTS)
colnames(ML_standard_6ov) <- col.names

# Ledoit & Wolf
LW_6ov <- matrix(as.numeric(NA), 
                 nrow = REP*NSIZES*NBETA, 
                 ncol = NRESULTS.L)
colnames(LW_6ov) <- col.names.l

# Model-based target
MBLW_6ov <- matrix(as.numeric(NA), 
                   nrow = REP*NSIZES*NBETA, 
                   ncol = NRESULTS.L)
colnames(MBLW_6ov) <- col.names.l


## ------------------------- ##
##      Simulation loop      ##
## ------------------------- ##

for(b in 1:length(beta)){             # 4 pop models
  
  for(ss in 1:length(SAMPLE.SIZES)) { # 5 sample sizes
    
    N <- SAMPLE.SIZES[ss]
    
    for(i in seq_len(REP)){           # 1000 replications
      
      cat("sample size = ", sprintf("%4d", N), " rep = ", 
          sprintf("%4d", i), "\n")
      
      this.seed <- 34 + (i + (ss - 1)*REP)
      set.seed(this.seed)
      
      idx <- (i + (ss - 1)*REP) + (b-1)*(NSIZES*REP)
      
      # generate data
      Data      <- simulateData(beta[b], sample.nobs = N, empirical = FALSE)
      cov.ML    <- cov(Data) * ((N-1)/N)
      fit.dummy <- sem(model6, data = Data, do.fit = FALSE)
      
      #------------------------------------------------------
      # fit 1: standard ML  
      #------------------------------------------------------
      fit.ML <- try(sem(model6, 
                        sample.cov = cov.ML, 
                        sample.nobs = N, 
                        sample.cov.rescale = FALSE), 
                    silent = TRUE)
      if(!inherits(fit.ML, "try-error")) {
        ML_standard_6ov[idx, ] <- extract_results_from_fit_MB(fit.ML) 
      }
      
      #------------------------------------------------------
      # fit 2: Ledoit & Wolf  
      #------------------------------------------------------
      cov.LW <- CovEst.2003LW(as.matrix(Data))$S
      lambda.LW <- CovEst.2003LW(as.matrix(Data))$delta
      
      fit.LW.6ov <- try(sem(model6, 
                            sample.cov = cov.LW, 
                            sample.nobs = N, 
                            sample.cov.rescale = FALSE), 
                        silent = TRUE)
      if(!inherits(fit.LW.6ov, "try-error")) {
        LW_6ov[idx, ] <- c(extract_results_from_fit_MB(fit.LW.6ov),lambda.LW)
      }
      
      #------------------------------------------------------
      # fit 3: Model-based target with L&W lambda   
      #------------------------------------------------------
      MB_LW.6ov <- lav_samplestats_modify_target(lavsamplestats = fit.dummy@SampleStats,
                                                 lavmodel = fit.dummy@Model,
                                                 cor = 0,
                                                 reliability = 0.8)
      
      cov.MB_LW <- CovEst.2003LW(as.matrix(Data), target = MB_LW.6ov$cov)$S
      lambda.MB_LW <- CovEst.2003LW(as.matrix(Data), target = MB_LW.6ov$cov)$delta
      
      fit.MB_LW.6ov <- try(sem(model6, 
                               sample.cov = cov.MB_LW, 
                               sample.nobs = N, 
                               sample.cov.rescale = FALSE), 
                           silent = TRUE)
      if(!inherits(fit.MB_LW.6ov, "try-error")) {
        MBLW_6ov[idx, ] <- c(extract_results_from_fit_MB(fit.MB_LW.6ov),lambda.MB_LW)
      }
      
    } # i
  } # ss
} # b

save(ML_standard_6ov,  file = "ML_standard_6ov.RData")
save(LW_6ov,           file = "LW_6ov.RData")
save(MBLW_6ov,         file = "MBLW_6ov.RData")

## ---------------------------------- END ----------------------------------- ##

