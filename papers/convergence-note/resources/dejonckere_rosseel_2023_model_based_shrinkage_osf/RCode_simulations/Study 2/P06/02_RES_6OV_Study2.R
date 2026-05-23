## ---------------------------
##
## Script name:         RES_6OV_Study2.R            
##
## Purpose of script:   Script for analyzing simulation results from 
##                      SIM_6OV_Study2.R 
##              
## Authors:             Julie De Jonckere and Yves Rosseel
##                      Department of Data Analysis, Ghent University
##
## ---------------------------
## Notes:
##      - specifically for population model A. Adjustment 
##        required if other pop is considered.
## ---------------------------

SAMPLE.SIZES <- c(10, 20, 30, 40, 50, 100)
NSIZES <- length(SAMPLE.SIZES)
REP <- 1000

RFILES <- list.files(path = ".", pattern = "\\.RData")
N <- length(RFILES)

OUT_MB_6ova <- vector("list", length=N)
#names(OUT) <- TOTAL$NAME
for(r in 1:N) {
  
  CONV <- integer(NSIZES)
  BAD    <- integer(NSIZES)
  MEAN   <- numeric(NSIZES)
  MEAN2  <- numeric(NSIZES)
  MEDIAN <- numeric(NSIZES)
  SD     <- numeric(NSIZES)
  MSE    <- numeric(NSIZES)
  VARX   <- numeric(NSIZES)
  VARY   <- numeric(NSIZES)
  #LAMBDA <- numeric(NSIZES)

  
  Name <- load(RFILES[r])
  RES <- get(Name)
  NAME <- rep(Name, NSIZES)
  RES <- get(Name)
  
  # population model A
  RES <- RES[1:(REP*NSIZES),]
  
  cat("r = ", r, "NAME = ", NAME[1], " ... ")
  
  if(nrow(RES) != REP*NSIZES) {
    cat(" SKIP.\n")
    next
  }
  
  
  for(i in seq_len(NSIZES)) {
    ss.idx <- (i-1)*REP + seq_len(REP)
    # converged
    idx <- ss.idx[RES[ss.idx, "optim"] == 1L & RES[ss.idx, "se"] == 1L]
    CONV[i] <- length(idx)
    
    BAD[i] <- REP - CONV[i]
    
    # mean
    MEAN[i] <- mean(RES[idx, "Y~X"], na.rm = TRUE)
    
    # trimmed mean
    MEAN2[i] <- mean(RES[idx, "Y~X"], trim = 0.005, na.rm = TRUE)
    
    # median
    MEDIAN[i] <- median(RES[idx, "Y~X"], na.rm = TRUE)
    
    # sd
    SD[i] <- sd(RES[idx, "Y~X"], na.rm = TRUE)
    
    # mse
    MSE[i] <- SD[i]^2 + (MEAN[i] - 0.25)^2        # OR 1 - 2 - 0
    
    # varx
    VARX[i] <- mean(RES[idx, "X~~X"], trim = 0.005)
    
    # vary
    VARY[i] <- mean(RES[idx, "Y~~Y"], trim = 0.005)
    
    
    
  }
  OUT_MB_6ova[[r]] <- data.frame(NAME = NAME, SAMPLE.SIZE = SAMPLE.SIZES, 
                                 CONV = CONV, MEAN = MEAN, MEAN2 = MEAN2,
                                 MEDIAN = MEDIAN, SD = SD, MSE = MSE, 
                                 VARX = VARX, VARY = VARY, BAD = BAD)
  cat("OK.\n")
}

names(OUT_MB_6ova) <- c("LW", "MBLW", "ML")


## ----------------------------------
##  Table: nonconvergence  
## ----------------------------------
Table6ov_convergence <- cbind(SAMPLE.SIZES,
                              OUT_MB_6ova$MBLW$BAD,
                              OUT_MB_6ova$LW$BAD,
                              OUT_MB_6ova$ML$BAD)
colnames(Table6ov_convergence) <- c("sample size", "MBLW", "L&W", "ML")
Table6ov_convergence



## ----------------------------------
##  Table: mean bias
## ----------------------------------
Table6ov_bias <- cbind(SAMPLE.SIZES,
                       (OUT_MB_6ova$MBLW$MEAN  - 0.25)*100,
                       (OUT_MB_6ova$LW$MEAN    - 0.25)*100,
                       (OUT_MB_6ova$ML$MEAN    - 0.25)*100)
colnames(Table6ov_bias) <- c("sample size", "MBLW", "L&W", "ML")
round(Table6ov_bias)



## ----------------------------------
##  Table: MSE
## ----------------------------------
Table6ov_mse <- cbind(SAMPLE.SIZES,
                      OUT_MB_6ova$MBLW$MSE*100,
                      OUT_MB_6ova$LW$MSE*100,
                      OUT_MB_6ova$ML$MSE*100)
colnames(Table6ov_mse) <- c("sample size", "MBLW", "L&W", "ML")
round(Table6ov_mse)
