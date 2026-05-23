## ---------------------------
##
## Script name:        SIM_Targets_M2.R
##
## Purpose of script:  Performance of adjusted covariance matrix using different
##                     target matrices: (1 − λ)S + λT. λ is chosen via grid.
##                     First value that leads to convergence is used. Which 
##                     target performs best in terms of convergence rate, bias, 
##                     and MSE?
##
## Authors:            Julie De Jonckere and Yves Rosseel
##                     Department of Data Analysis, Ghent University
##
## ---------------------------
##
## Notes:  
##        - P    = 6  
##        - Beta = 0.3  
##        - N    = c(10, 20, 30, 40, 50, 100)
##        - Population and analysis model 2
## ---------------------------


## --------------------------------- ##
##     Source necessary files        ##
## --------------------------------- ##
source("Pre_Simulation_Study2.R")
Ò
## --------------------------------- ##
##            Pre Simulation         ##
## --------------------------------- ##
SAMPLE.SIZES <- c(10, 20, 30, 40, 50, 100)
NSIZES <- length(SAMPLE.SIZES)
NRESULTS <- length(col.names)
NRESULTS.L <- length(col.names.l)
REP <- 1000

# 51 values for lambda are considered
valuesL <- c(0,    0.01, 0.02, 
             0.03, 0.04, 0.05, 
             0.06, 0.07, 0.08,
             0.09, 0.10, 0.15, 
             0.17, 0.20, 0.23, 
             0.25, 0.27, 0.30, 
             0.32, 0.35, 0.37, 
             0.40, 0.42, 0.45, 
             0.47, 0.50, 0.52, 
             0.55, 0.57, 0.60, 
             0.62, 0.65, 0.67, 
             0.70, 0.72, 0.75, 
             0.77, 0.80, 0.82, 
             0.85, 0.87, 0.90, 
             0.92, 0.95, 0.96,
             0.97, 0.98, 0.99, 
             0.999,0.9999, 1)
VALUESL <- length(valuesL)

## --------------------------------- ##
##  Containers to store results      ##
## --------------------------------- ##

# standard ML
RES_ML_standard_M2 <- matrix(as.numeric(NA), 
                             nrow = REP*NSIZES, 
                             ncol = NRESULTS)
colnames(RES_ML_standard_M2) <- col.names


# shrinkage targets 
RES_id_target_M2 <- matrix(as.numeric(NA), 
                           nrow = REP*NSIZES, 
                           ncol = NRESULTS.L)
colnames(RES_id_target_M2) <- col.names.l

RES_cc_target_M2 <- matrix(as.numeric(NA), 
                           nrow = REP*NSIZES, 
                           ncol = NRESULTS.L)
colnames(RES_cc_target_M2) <- col.names.l

RES_cv_target_M2 <- matrix(as.numeric(NA), 
                           nrow = REP*NSIZES, 
                           ncol = NRESULTS.L)
colnames(RES_cv_target_M2) <- col.names.l


# model-based targets
RES_MB1_target_M2 <- matrix(as.numeric(NA), 
                            nrow = REP*NSIZES, 
                            ncol = NRESULTS.L)
colnames(RES_MB1_target_M2) <- col.names.l

RES_MB2_target_M2 <- matrix(as.numeric(NA), 
                            nrow = REP*NSIZES, 
                            ncol = NRESULTS.L)
colnames(RES_MB2_target_M2) <- col.names.l



## -----------------------------------------------------------------------------
## SIMULATION: 6 OV - beta = 0.3                
## -----------------------------------------------------------------------------
for(ss in 1:length(SAMPLE.SIZES)) {
  
  N <- SAMPLE.SIZES[ss]
  
  for(i in seq_len(REP)){
    
    cat("sample size = ", sprintf("%4d", N), " rep = ", 
        sprintf("%4d", i), "\n")
    
    this.seed <- 34 + i + (ss - 1)*REP
    set.seed(this.seed)
    
    idx <- i + (ss - 1)*REP 
    
    # generate data
    Data      <- simulateData(pop.model.2, sample.nobs = N, empirical = FALSE)
    cov.ML    <- cov(Data) * ((N-1)/N)
    fit.dummy <- sem(model, data = Data, do.fit = FALSE)
    
    
    
    #------------------------------------------------------
    # fit 1: standard ML - Model 2  
    #------------------------------------------------------
    fit.ML <- try(sem(model, 
                      sample.cov = cov.ML, 
                      sample.nobs = N, 
                      sample.cov.rescale = FALSE,
                      optim.attempts = 1L,
                      baseline = FALSE), 
                  silent = TRUE)
    if(!inherits(fit.ML, "try-error")) {
      RES_ML_standard_M2[idx, ] <- extract_results_from_fit_MB(fit.ML) 
    }
    
    #------------------------------------------------------
    # fit 2: S adjusted - identity T - Model 2  
    #------------------------------------------------------
    for(k in 1:length(valuesL)){  
      
      id_cov <- cov_est_ID(Data, 
                           lambda = valuesL[k],
                           lavmodel = fit.dummy@Model,
                           lavsamplestats = fit.dummy@SampleStats)
      fit.id <- try(sem(model, 
                        sample.cov = id_cov, 
                        sample.nobs = N, 
                        sample.cov.rescale = FALSE,
                        optim.attempts = 1L,
                        baseline = FALSE), 
                    silent = TRUE)
      if(!inherits(fit.id, "try-error") && lavInspect(fit.id, "converged") && 
         !any(is.na(parameterEstimates(fit.id)$se))){
        break
      }
    }
    RES_id_target_M2[idx, ] <- c(extract_results_from_fit_MB(fit.id),valuesL[k])
    
    #------------------------------------------------------
    # fit 3: S adjusted - constant correlation T - Model 2  
    #------------------------------------------------------
    for(k in 1:length(valuesL)){  
      
      cc_cov <- cov_est_CC(Data, 
                           lambda = valuesL[k],
                           lavmodel = fit.dummy@Model,
                           lavsamplestats = fit.dummy@SampleStats)
      fit.cc <- try(sem(model, 
                        sample.cov = cc_cov, 
                        sample.nobs = N, 
                        sample.cov.rescale = FALSE,
                        optim.attempts = 1L,
                        baseline = FALSE), 
                    silent = TRUE)
      if(!inherits(fit.cc, "try-error") && lavInspect(fit.cc, "converged") && 
         !any(is.na(parameterEstimates(fit.cc)$se))){
        break
      }
    }
    
    RES_cc_target_M2[idx, ] <- c(extract_results_from_fit_MB(fit.cc),valuesL[k])
    
    #------------------------------------------------------
    # fit 4: S adjusted - common variance T - Model 2  
    #------------------------------------------------------
    for(k in 1:length(valuesL)){  
      
      cv_cov <- cov_est_CV(Data, 
                           lambda = valuesL[k],
                           lavmodel = fit.dummy@Model,
                           lavsamplestats = fit.dummy@SampleStats)
      
      fit.cv <- try(sem(model, 
                        sample.cov = cv_cov, 
                        sample.nobs = N, 
                        sample.cov.rescale = FALSE,
                        optim.attempts = 1L,
                        baseline = FALSE), 
                    silent = TRUE)
      if(!inherits(fit.cv, "try-error") && lavInspect(fit.cv, "converged") && 
         !any(is.na(parameterEstimates(fit.cv)$se))){
        break
      }
    }
    RES_cv_target_M2[idx, ] <- c(extract_results_from_fit_MB(fit.cv),valuesL[k])
    
    #------------------------------------------------------
    # fit 5: S adjusted - MB T - single COR - Model 2  
    #------------------------------------------------------
    for(k in 1:length(valuesL)){  
      
      MB1_cov <- lav_samplestats_modify_cov(lavsamplestats = fit.dummy@SampleStats,
                                            lavmodel = fit.dummy@Model,
                                            lambda = valuesL[k],
                                            cor = 0.2,
                                            reliability = 0.8)
      
      fit.MB1 <- try(sem(model, 
                         sample.cov = MB1_cov[[1]][[1]]$cov, 
                         sample.nobs = N, 
                         sample.cov.rescale = FALSE,
                         optim.attempts = 1L,
                         baseline = FALSE),
                     silent = TRUE)
      if(!inherits(fit.MB1, "try-error") && lavInspect(fit.MB1, "converged") && 
         !any(is.na(parameterEstimates(fit.MB1)$se))){
        break
      }
    }
    RES_MB1_target_M2[idx, ] <- c(extract_results_from_fit_MB(fit.MB1),valuesL[k])
    
    #------------------------------------------------------
    # fit 6: S adjusted - M-B T - zero COR - Model 2  
    #------------------------------------------------------
    for(k in 1:length(valuesL)){  
      
      MB2_cov <- lav_samplestats_modify_cov(lavsamplestats = fit.dummy@SampleStats,
                                            lavmodel = fit.dummy@Model,
                                            lambda = valuesL[k],
                                            cor = 0,
                                            reliability = 0.8)
      
      fit.MB21 <- try(sem(model, 
                          sample.cov = MB2_cov[[1]][[1]]$cov, 
                          sample.nobs = N, 
                          sample.cov.rescale = FALSE,
                          optim.attempts = 1L,
                          baseline = FALSE),
                      silent = TRUE)
      if(!inherits(fit.MB21, "try-error") && lavInspect(fit.MB21, "converged") && 
         !any(is.na(parameterEstimates(fit.MB21)$se))){
        break
      }
    }
    RES_MB2_target_M2[idx, ] <- c(extract_results_from_fit_MB(fit.MB21),valuesL[k])
    
  } # END - inner loop
} # END - outer loop


save(RES_ML_standard_M2,  file = "RES_ML_standard_M2.RData")
save(RES_id_target_M2,    file = "RES_id_target_M2.RData")
save(RES_cc_target_M2,    file = "RES_cc_target_M2.RData")
save(RES_cv_target_M2,    file = "RES_cv_target_M2.RData")
save(RES_MB1_target_M2,   file = "RES_MB1_target_M2.RData")
save(RES_MB2_target_M2,   file = "RES_MB2_target_M2.RData")
