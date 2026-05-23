# new simultions random starts lavaan

# install from github
library("remotes")
remotes::install_github("yrosseel/lavaan")
library("lavaan")


#############################################
##
##              STUDY 1
##
#############################################

setwd("/home/semlab/Julie/New Test Rstart/New Test Rstart")


load("OPTIM.NEW.RData")
load("OPTIM.NEW07.RData")

load("tmp.results.RData")


pop.model <- '
   
   FX =~ 1*y1 + 0.8*y2 + 0.6*y3
   FY =~ 1*y4 + 0.7*y5 + 0.9*y6

   # regression
   FY ~ 0.1*FX        
   
   # factor (residual) variances
   FX ~~ 1*FX
   FY ~~ 1*FY

   # exo
   y1 ~~ ((1 - 0.1)*1*1.0^2/0.1)*y1       #rel 0.1
   y2 ~~ ((1 - 0.1)*1*0.8^2/0.1)*y2
   y3 ~~ ((1 - 0.1)*1*0.6^2/0.1)*y3

   # endo
   y4 ~~ ((1 - 0.1)*1.01*1.0^2/0.1)*y4
   y5 ~~ ((1 - 0.1)*1.01*0.7^2/0.1)*y5
   y6 ~~ ((1 - 0.1)*1.01*0.9^2/0.1)*y6
'



## -------------------
## POPULATION MODEL 2
## -------------------

pop.model2 <- '
   
   FX =~ 1*y1 + 0.8*y2 + 0.6*y3
   FY =~ 1*y4 + 0.7*y5 + 0.9*y6

   # regression
   FY ~ 0.7*FX # total variance = 0.7^2 + 1 = 1.49
   
   
   # factor (residual) variances
   FX ~~ 1*FX
   FY ~~ 1*FY

   # exo
   y1 ~~ ((1 - 0.1)*1*1.0^2/0.1)*y1       #rel 0.1
   y2 ~~ ((1 - 0.1)*1*0.8^2/0.1)*y2
   y3 ~~ ((1 - 0.1)*1*0.6^2/0.1)*y3

   # endo
   y4 ~~ ((1 - 0.1)*1.49*1.0^2/0.1)*y4
   y5 ~~ ((1 - 0.1)*1.49*0.7^2/0.1)*y5
   y6 ~~ ((1 - 0.1)*1.49*0.9^2/0.1)*y6
'


## -------------------
## MODEL for analysis
## -------------------

model <- '
    
    FX =~ y1 + y2 + y3
    FY =~ y4 + y5 + y6

    # regression part 
    FY ~ FX
    
    # factor variances
    FX ~~ FX
    FY ~~ FY
    
    # observed residual variances
    y1 ~~ y1
    y2 ~~ y2
    y3 ~~ y3
    y4 ~~ y4
    y5 ~~ y5
    y6 ~~ y6
'

## -------------------------------------------------------------------------- ##
##                               2. Utilities                                 ##      
## -------------------------------------------------------------------------- ##

# parameter values
FIT1       <- lavaan::sem(model)               
OV.NAMES   <- lavNames(FIT1, "ov") # observed variables
LV.NAMES   <- lavNames(FIT1, "lv") # latent variables
COEF.NAMES <- c(names(coef(FIT1)), "nr_sets_needed_conv" ) # (free) parameter names

col.names  <- c(COEF.NAMES, 
                "beta.se", 
                "optim", 
                "se", 
                "ov.var.pos", 
                "lv.var.pos")

NPAR <- length(COEF.NAMES)      
NVAR <- length(OV.NAMES) 

n_cov_elements <- NVAR * (NVAR + 1) / 2  
n_eig_values   <- NVAR                     
NDATA          <- n_cov_elements + n_eig_values

PTE1       <- parTable(FIT1)
ov.var.idx <- which(PTE1$op == "~~" & PTE1$lhs %in% OV.NAMES & PTE1$lhs == PTE1$rhs) 
lv.var.idx <- which(PTE1$op == "~~" & PTE1$lhs %in% LV.NAMES & PTE1$lhs == PTE1$rhs) 
beta.idx   <- which(PTE1$op == "~")

OV <- c("y1","y2","y3","y4","y5","y6")

NRESULTS <- length(col.names)
REP <- 1000
N <- 20


## -------------------------------------------------------------------------- ##
##                        3. Data extract function                            ##      
## -------------------------------------------------------------------------- ##
extract_results_from_fit <- function(fit) {
  
  OPTIM <- SE <- OV.VAR.POS <- LV.VAR.POS <- 0L; BETA.SE <- NA
  
  PT <- parTable(fit)
  THETA <- PT$est[   PT$free > 0L ]
  
  
  if(lavInspect(fit, "converged")) {
    OPTIM <- 1L
  } 
  if(!all(is.na(PT$se[PT$free > 0]))) {
    SE <- 1L
    BETA.SE <- PT$se[ beta.idx ]
  }
  if(all(PT$est[ov.var.idx] >= 0)) {  
    OV.VAR.POS <- 1
  }
  if(all(PT$est[lv.var.idx] >= 0)) { 
    LV.VAR.POS <- 1
  }
  
  c(THETA, BETA.SE, OPTIM, SE, OV.VAR.POS, LV.VAR.POS)
}




## -------------------------------------------------------------------------- ##
##                   		4. Simulation                           	 ##      
## -------------------------------------------------------------------------- ##
# select nonconverged cases
nonconv.idx <- which(tmp.results[,"optim"] == 0L | tmp.results[,"optim"] == 1L & tmp.results[,"se"] == 0L) 
L.nonconv   <- length(nonconv.idx) #363

## -------------------
## DATA CONTAINERS   
## -------------------

RESULTS_STUDY1_RSTART_NEW  <- matrix(as.numeric(NA), 
                                     nrow = length(nonconv.idx), 
                                     ncol = length(COEF.NAMES))
colnames(RESULTS_STUDY1_RSTART_NEW) <- COEF.NAMES


## -------------------
## SIMULATION START
## -------------------

start_time <- Sys.time()
for(i in 1:length(nonconv.idx)) {
  
  cat("sample size = ", sprintf("%4d", N), " rep = ", 
      sprintf("%4d", i), "\n")
  
  this.seed <-  nonconv.idx[i]
  set.seed(this.seed)
  
  Data      <- simulateData(pop.model, 
                            sample.nobs = N, 
                            empirical = FALSE)
  
    
  fit.start1 <- try(lavaan::sem(model, 
                                data   = Data, 
                                rstarts = 150), 
                     silent = TRUE)
  first.conv <- which(sapply(fit.start1@optim$x.rstarts, attr, "converged"))[1]

  RESULTS_STUDY1_RSTART_NEW[i,] <-  c(fit.start1@optim$x.rstarts[[first.conv]][1:(length(COEF.NAMES)-1)], first.conv)

  
}
end_time <- Sys.time()
end_time - start_time # 5.87 uur
save(RESULTS_STUDY1_RSTART_NEW, file = "RESULTS_STUDY1_RSTART_NEW.RData")


## -------------------
## BOUNDS
## -------------------
COEF.NAMES <- names(coef(FIT1))  # (free) parameter names

col.names  <- c(COEF.NAMES, 
                "beta.se", 
                "optim", 
                "se", 
                "ov.var.pos", 
                "lv.var.pos")
 NRESULTS <- length(col.names)
 
                
BOUNDS_RES_NEW <- matrix(as.numeric(NA), nrow =L.nonconv, ncol = NRESULTS)
colnames(BOUNDS_RES_NEW) <- col.names

for(i in 1:length(nonconv.idx)) {
  
  cat("sample size = ", sprintf("%4d", N), " rep = ", 
      sprintf("%4d", i), "\n")
  
  this.seed <-  nonconv.idx[i]
  set.seed(this.seed)
  
  Data       <- simulateData(pop.model, 
                             sample.nobs = N, 
                             empirical = FALSE)
  
  fit.bounds <- try(lavaan::sem(model, 
                                data = Data, 
                                bounds = "standard"), 
                    silent = TRUE)
  if(!inherits(fit.bounds, "try-error")){
    BOUNDS_RES_NEW[i,] <- extract_results_from_fit(fit.bounds)
  }
}
save(BOUNDS_RES_NEW, file = "BOUNDS_RES_NEW.RData")

## -------------------------------------------------------------------------- ##
##                          5. POPULATION MODEL 2                             ##      
## -------------------------------------------------------------------------- ##

# select nonconverged cases
nc.idx  <- which(OPTIM.NEW07[,"optim"] == 0L | OPTIM.NEW07[,"optim"] == 1L & OPTIM.NEW07[,"se"] == 0L)
nc.L    <- length(nc.idx)
data_nonconverged <- OPTIM.NEW07[nc.idx,]


## -------------------
## DATA CONTAINERS   
## -------------------

RESULTS_STUDY1_07_RSTART_NEW  <- matrix(as.numeric(NA), 
                                        nrow = length(nc.idx), 
                                        ncol = length(COEF.NAMES))
colnames(RESULTS_STUDY1_07_RSTART_NEW) <- COEF.NAMES


## -------------------
## SIMULATION START
## -------------------

start_time <- Sys.time()
for(i in 1:length(nc.idx)) {
  
  cat("sample size = ", sprintf("%4d", N), " rep = ", 
      sprintf("%4d", i), "\n")
  
  this.seed <-  nc.idx[i]
  set.seed(this.seed)
  
  Data      <- simulateData(pop.model2, 
                            sample.nobs = N, 
                            empirical = FALSE)
  
  
  fit.start1.07 <- try(lavaan::sem(model, 
                                data   = Data, 
                                rstarts = 150), 
                    silent = TRUE)
  first.conv.07 <- which(sapply(fit.start1.07@optim$x.rstarts, attr, "converged"))[1]
  
  RESULTS_STUDY1_07_RSTART_NEW[i,] <-  c(fit.start1.07@optim$x.rstarts[[first.conv.07]][1:(length(COEF.NAMES)-1)], first.conv.07)
  
  
}
end_time <- Sys.time()
end_time - start_time # 5.87 uur
save(RESULTS_STUDY1_07_RSTART_NEW, file = "RESULTS_STUDY1_07_RSTART_NEW.RData")

