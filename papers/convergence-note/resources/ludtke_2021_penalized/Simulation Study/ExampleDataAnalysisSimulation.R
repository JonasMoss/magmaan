#==================================================
# Alleviating Estimation Problems in Small Sample Structural Equation Modeling-A Comparison of Constrained Maximum Likelihood, Bayesian Estimation, and Fixed Reliability Approaches
#
# Exemplary Analysis Syntax
#==================================================

library(rstan)
options(mc.cores = 3)
library(lavaan)

# load generated data set
# (N=100, ab=0.50, c=0.00, omega=0.55)
load("Dat1.RData")

# constrained ML
#-------------------------------------------
m<-'F1 =~ NA*x1 + l1*x1 + l2*x2 + l3*x3  
  F2 =~ NA*x4 +l4*x4 + l5*x5 + l6*x6  
  F3 =~ NA*x7 +l7*x7 + l8*x8 + l9*x9  
  F1 ~~ 1*F1
  F2 ~~ 1*F2
  F3 ~~ 1*F3
  F2 ~~ cor12*F1    
  F3 ~~ cor13*F1
  F2 ~~ cor23*F3
  x1 ~~ resx1*x1
  x2 ~~ resx2*x2
  x3 ~~ resx3*x3
  x4 ~~ resx4*x4
  x5 ~~ resx5*x5
  x6 ~~ resx6*x6
  x7 ~~ resx7*x7
  x8 ~~ resx8*x8
  x9 ~~ resx9*x9
  # constraint residual variances
  resx1 > 0.01
  resx2 > 0.01
  resx3 > 0.01
  resx4 > 0.01
  resx5 > 0.01
  resx6 > 0.01
  resx7 > 0.01
  resx8 > 0.01
  resx9 > 0.01
  # constraint loadings
  l1 > 0.01
  l4 > 0.01
  l7 > 0.01
  # constraint correlations
  cor12 > -0.99
  cor13 > -0.99
  cor23 > -0.99
  cor12 < 0.99
  cor13 < 0.99
  cor23 < 0.99
  # constraint positive definite
  1+2*cor12*cor13*cor23-cor12^2-cor13^2-cor23^2 > 0.01
  a := cor12
  b := (cor23-cor13*cor12)/(1-cor12^2)
  c := (cor13-cor23*cor12)/(1-cor12^2)
  ab := a*b
'

X<-as.data.frame(X)
names(X)<-paste0("x",c(1:9))

fitSEM<-sem(m,data=X,se="boot",bootstrap=1000)
summary(fitSEM)

# fixed reliability SI, adapted from Savalei (2019) (e.g., SI9)
#------------------------------------------------------------------
relSI<-.9

c1<-rowMeans(X[1:3]) # composites
c2<-rowMeans(X[4:6])
c3<-rowMeans(X[7:9])
Cdata<-data.frame(cbind(c1,c2,c3)) 

PathMod<-'F1 =~ c1   
  F2 =~ c2  
  F3 =~ c3 
  F2 ~~ cov12*F1    
  F3 ~~ cov13*F1
  F2 ~~ cov23*F3
  F1 ~~ var1*F1
  F2 ~~ var2*F2
  F3 ~~ var3*F3
  cor13:= cov13/(sqrt(var1)*sqrt(var3))
  cor23:= cov23/(sqrt(var2)*sqrt(var3))
  cor12:= cov12/(sqrt(var1)*sqrt(var2))
  var1 > 0.01
  var2 > 0.01
  var3 > 0.01
  cor12 > -.99
  cor13 > -.99
  cor23 > -.99
  cor12 < .99
  cor13 < .99
  cor23 < .99
  # constraint positive definite
  1+2*cor12*cor13*cor23-cor12^2-cor13^2-cor23^2>0.01
  a := cor12*sqrt(var2)/sqrt(var1)
  b := (cor23-cor13*cor12)/(1-cor12^2)*sqrt(var3)/sqrt(var2)
  c := (cor13-cor23*cor12)/(1-cor12^2)*sqrt(var3)/sqrt(var1)  
  ab : = a*b
'
# add fixed reliabilities 
vars<-diag(cov(Cdata))*(nrow(X)-1)/nrow(X) #variances of observed variables     
addtoPathMod<-NULL
for (k in (1:3)) {
  part<-paste("c",k,"~~",(1-relSI)*vars[k],"*c",k," \n ",sep="")
  addtoPathMod<-paste(addtoPathMod,part,sep=" ")
}

PathModSI<-paste(PathMod,"\n", addtoPathMod,sep=" ") 
fitSI<-sem(PathModSI, data=Cdata, se="boot", bootstrap=1000)
summary(fitSI)

# Bayesian estimation with uniform priors
#----------------------------------------------------------------

# scatter matrix
X<-as.matrix(X)
ones.vec <- matrix(1,nrow(X), 1)
S<-t(X)%*%X - 1/nrow(X)*(t(X)%*%ones.vec)%*%(t(ones.vec)%*%X)  

Med_list<- list(
  J  = ncol(X), # number of items
  N  = nrow(X), # number of persons
  dd = c(rep(1,3),rep(2,3),rep(3,3)), # dimension identified 
  S = S, # scatter matrix
  xbar = colMeans(X) #  indicator means
) 

# starting values
init_fun <- function(){list(lambda1 = runif(3,0.01,.99),
                            lambda2 = runif(6,0.01,.99)
)} 

# estimation
modelUnif = stan_model(file = 'BEUnif.stan')
fitBEUnif <- sampling(modelUnif,  data = Med_list, iter=50000, chains = 3, control = list(adapt_delta = 0.99, max_treedepth=12),  init=init_fun, warmup=10000) 

summary(fitBEUnif, pars = c("a","b", "c", "ab", "correl[1,2]", "correl[2,3]","correl[1,3]"))[[1]] 

# point estimates regression coefficients
sumBEUnif<-summary(fitBEUnif)[[1]] 

cor12<-sumBEUnif["correl[1,2]",1]
cor13<-sumBEUnif["correl[1,3]",1]
cor23<-sumBEUnif["correl[2,3]",1]

a<-cor12
b<-(cor23-cor13*cor12)/(1-cor12^2)
c<-(cor13-cor23*cor12)/(1-cor12^2)
ab<-a*b

a
b
c
ab

# Bayesian estimation with weakly informative priors on correlations (e.g., nu_0=3, rho_0=0)
#----------------------------------------------------------------------------

Med_list<- list(
  J  = ncol(X), # number of items
  N  = nrow(X), # number of persons
  dd = c(rep(1,3),rep(2,3),rep(3,3)), # dimension identified 
  S = S, # scatter matrix
  xbar = colMeans(X), #  indicator means
  n0C = 3, # prior sample size
  pGC = c(0,0,0) # prior guess
) 

# starting values
init_fun <- function(){list(lambda1 = runif(3,0.01,.99),
                            lambda2 = runif(6,0.01,.99)
)} 

# estimation
modelInfCorr = stan_model(file = 'BEInfCorr.stan')
fitBEInfCorr <- sampling(modelInfCorr,  data = Med_list, iter=50000, chains = 3, control = list(adapt_delta = 0.99, max_treedepth=12), init=init_fun, warmup=10000) 

summary(fitBEInfCorr, pars = c("a","b", "c", "ab", "correl[1,2]", "correl[2,3]","correl[1,3]"))[[1]] 

# point estimates regression coefficients
sumBEInfCorr<-summary(fitBEInfCorr)[[1]] 

cor12<-sumBEInfCorr["correl[1,2]",c(1,4,8)]
cor13<-sumBEInfCorr["correl[1,3]",c(1,4,8)]
cor23<-sumBEInfCorr["correl[2,3]",c(1,4,8)]

a<-cor12
b<-(cor23-cor13*cor12)/(1-cor12^2)
c<-(cor13-cor23*cor12)/(1-cor12^2)
ab<-a*b

a
b
c
ab

# Bayesian estimation with weakly informative priors on standardized factor loadings (e.g., nu_0= 3, omega_0 = 0.55)
#-------------------------------------------------------------------------------

Med_list<- list(
  J  = ncol(X), # number of items
  N  = nrow(X), # number of persons
  dd = c(rep(1,3),rep(2,3),rep(3,3)), # dimension identified 
  S = S, # scatter matrix
  xbar = colMeans(X), #  indicator means
  n0L = 3, # prior sample size
  pGL = rep(0.54,9) # prior guess
) 

# starting values
init_fun <- function(){list(lambdaStar = (runif(9,0,1)-rep(c(0.01,-0.99,-0.99),3))/(0.99-rep(c(0.01,-0.99,-0.99),3))
)} 

# estimation
modelInfLoad = stan_model(file = 'BEInfLoad.stan')
fitBEInfLoad <- sampling(modelInfLoad,  data = Med_list, iter=50000, chains = 3, control = list(adapt_delta = 0.99, max_treedepth=12),  init=init_fun, warmup=10000) 

summary(fitBEInfLoad, pars = c("a","b", "c", "ab", "correl[1,2]", "correl[2,3]","correl[1,3]"))[[1]] 

# point estimates regression coefficients
sumBEInfLoad<-summary(fitBEInfLoad)[[1]] 

cor12<-sumBEInfLoad["correl[1,2]",c(1,4,8)]
cor13<-sumBEInfLoad["correl[1,3]",c(1,4,8)]
cor23<-sumBEInfLoad["correl[2,3]",c(1,4,8)]

a<-cor12
b<-(cor23-cor13*cor12)/(1-cor12^2)
c<-(cor13-cor23*cor12)/(1-cor12^2)
ab<-a*b

a
b
c
ab
