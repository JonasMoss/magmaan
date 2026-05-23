## ---------------------------
##
## Script name: Rcode_example_ML_MB_Lambda.R
##
## Purpose:     Performance of adjusted covariance matrix using different model 
##              based target matrices: (1 − λ)S + λT.
##              
## Authors:     Julie De Jonckere and Yves Rosseel
##              Department of Data Analysis, Ghent University
##
## ---------------------------
## ---------------------------
source("Get_Options_TsemStar.R")

# population model
pop.model <- '
# factor loadings
Y =~ 1*y1 + 0.8*y2 + 0.6*y3
X =~ 1*x1 + 0.8*x2 + 0.6*x3

# regression part
Y ~ 0.25*X
'

# model to be fitted
model <- model <- '
    Y =~ y1 + y2 + y3
    X =~ x1 + x2 + x3
    
    Y ~ X
'


set.seed(12)
Data <- simulateData(pop.model, sample.nobs = 20L)
fit.dummy <- sem(model, data = Data, do.fit = FALSE)

## -------------------------------------------------------------------- ##
##                      Option 0: standard ML                           ##
## -------------------------------------------------------------------- ##
fit.ML <- sem(model, data = Data, estimator = "ML", optim.attempts = 1L, 
               verbose = TRUE)
summary(fit)

covmatrix <- lavInspect(fit.ML, 'samplestats', add.class = FALSE)$cov
covmatrix


## -------------------------------------------------------------------- ##
##              Option 1: user-defined correlation matrix               ##
## -------------------------------------------------------------------- ##
user.cor <- matrix(c(1,0.4,0.4,1), nrow = 2, ncol = 2)
cov.MB1  <- lav_samplestats_modify_cov_ud(lavsamplestats = fit.dummy@SampleStats,
                                          lavmodel = fit.dummy@Model,
                                          lambda = 0.2,
                                          COR.M = user.cor,
                                          reliability = 0.8)
fit.covDP1 <- sem(model, sample.cov = cov.MB1[[1]][[1]]$cov, sample.nobs = 20, 
                  sample.cov.rescale = FALSE)
summary(fit.covDP1)

cov.MB1[[1]][[1]]$cov

## --------------------------------------------------------------------- ##
##                     Option 2: single correlation                      ##
## --------------------------------------------------------------------- ##
cov.MB2 <- lav_samplestats_modify_cov(lavsamplestats = fit.dummy@SampleStats,
                                      lavmodel = fit.dummy@Model,
                                      lambda = 0.2,
                                      cor = 0.2,
                                      reliability = 0.8)

fit.covDP2 <- sem(model, sample.cov = cov.MB2[[1]][[1]]$cov, sample.nobs = 20, 
                 sample.cov.rescale = FALSE)
summary(fit.covDP2)

cov.MB2[[1]][[1]]$cov

## --------------------------------------------------------------------- ##
##                  Option 3: zero latent correlation                    ##
## --------------------------------------------------------------------- ##
# construct the model-based variance-covariance matrix
cov.MB3 <- lav_samplestats_modify_cov(lavsamplestats = fit.dummy@SampleStats,
                                      lavmodel = fit.dummy@Model,
                                      lambda = 0.2,
                                      cor = 0,
                                      reliability = 0.8)

fit.covDP3 <- sem(model, sample.cov = cov.MB3[[1]][[1]]$cov, sample.nobs = 20, 
                     sample.cov.rescale = FALSE)
summary(fit.covDP3)

cov.MB3[[1]][[1]]$cov

# try another value for lambda that is lower than 0.5. 
#cov.DP.lower <- lav_samplestats_modify_cov(lavsamplestats = fit.dummy@SampleStats,
#                                           lavmodel = fit.dummy@Model,
#                                           lambda = 0.1,
#                                           cor = 0,
#                                           reliability = 0.8)

#fit.covDP.lower <- sem(model, sample.cov = cov.DP.lower[[1]][[1]]$cov, 
#                       sample.nobs = 20, sample.cov.rescale = FALSE)
#summary(fit.covDP.lower)


## --------------------------------------------------------------------- ##
##                  Option 4: cov elements Sa = cov S                    ##
## --------------------------------------------------------------------- ##
# try model-based with covariances Sa = covariances S
cov.MB4 <- lav_samplestats_modify_cov2(lavsamplestats = fit.dummy@SampleStats,
                                       lavmodel = fit.dummy@Model,
                                       lambda = 0.25,
                                       cor = 0,
                                       reliability = 0.8)

fit.covDP4 <- sem(model, sample.cov = cov.MB4[[1]]$cov, sample.nobs = 20, 
                  sample.cov.rescale = FALSE)
summary(fit.covDP4)
cov.MB4[[1]]$cov
