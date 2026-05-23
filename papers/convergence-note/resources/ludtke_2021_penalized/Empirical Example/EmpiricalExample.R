#==================================================
# Alleviating Estimation Problems in Small Sample Structural Equation Modeling-A Comparison of Constrained Maximum Likelihood, Bayesian Estimation, and Fixed Reliability Approaches
#
# Empirical Example
#==================================================

library(rstan)
options(mc.cores = 3)
library(lavaan)

X <- get(data("PoliticalDemocracy")) 
N<-nrow(X)
# F1: Ind60, x1-x3
# F2: Dem60, y1-y4
# F3: Dem65, y5-8

Results<-setNames(data.frame(matrix(ncol =8, nrow = 16)), c("specification","a","b","c","ab", "omega1", "omega2","omega3"))

ind<-1

# CML
#======================================================
CML<-'
  F1 =~ NA*x1 + l1*x1 + l2*x2 + l3*x3  
  F2 =~ NA*y1 + l4*y1 + l5*y2 + l6*y3 + l7*y4 
  F3 =~ NA*y5 + l8*y5 + l9*y6 + l10*y7 + l11*y8 
  F1 ~~ 1*F1
  F2 ~~ 1*F2
  F3 ~~ 1*F3
  F2 ~~ cor12*F1    
  F3 ~~ cor13*F1
  F2 ~~ cor23*F3
  x1 ~~ resx1*x1
  x2 ~~ resx2*x2
  x3 ~~ resx3*x3
  y1 ~~ resy1*y1
  y2 ~~ resy2*y2
  y3 ~~ resy3*y3
  y4 ~~ resy4*y4
  y5 ~~ resy5*y5
  y6 ~~ resy6*y6
  y7 ~~ resy7*y7
  y8 ~~ resy8*y8
  # constraint residual variances
  resx1 >0.01
  resx2 >0.01
  resx3 >0.01
  resy1 >0.01
  resy2 >0.01
  resy3 >0.01
  resy4 >0.01
  resy5 >0.01
  resy6 >0.01
  resy7 >0.01
  resy8 >0.01
  # constraint loadings
  l1 >0.01
  l4 >0.01
  l8 >0.01
  # constraint correlations
  cor12 > -.99
  cor13 > -.99
  cor23 > -.99
  cor12 < .99
  cor13 < .99
  cor23 < .99
  # constraint positive definite
  1+2*cor12*cor13*cor23-cor12^2-cor13^2-cor23^2>0.01
  a := cor12
  b := (cor23-cor13*cor12)/(1-cor12^2)
  c := (cor13-cor23*cor12)/(1-cor12^2)
  ab := a*b
'

fitCML<-sem(CML,data=X,se="boot",bootstrap=1000) 

estimatesCML<-parameterestimates(fitCML, boot.ci.type="perc",standardized = T)

Results[ind:(ind+1),"specification"]<-rep("CML",2)
Results[ind,"a"]<-sprintf(estimatesCML$est[estimatesCML$label=="a"], fmt = '%#.2f')
Results[ind,"b"]<-sprintf(estimatesCML$est[estimatesCML$label=="b"], fmt = '%#.2f')
Results[ind,"c"]<-sprintf(estimatesCML$est[estimatesCML$label=="c"], fmt = '%#.2f')
Results[ind,"ab"]<-sprintf(estimatesCML$est[estimatesCML$label=="ab"], fmt = '%#.2f')
Results[ind+1,"a"]<-paste0("[",sprintf(estimatesCML$ci.lower[estimatesCML$label=="a"], fmt = '%#.2f'),"; ",sprintf(estimatesCML$ci.upper[estimatesCML$label=="a"], fmt = '%#.2f'),"]")
Results[ind+1,"b"]<-paste0("[",sprintf(estimatesCML$ci.lower[estimatesCML$label=="b"], fmt = '%#.2f'),"; ",sprintf(estimatesCML$ci.upper[estimatesCML$label=="b"], fmt = '%#.2f'),"]")
Results[ind+1,"c"]<-paste0("[",sprintf(estimatesCML$ci.lower[estimatesCML$label=="c"], fmt = '%#.2f'),"; ",sprintf(estimatesCML$ci.upper[estimatesCML$label=="c"], fmt = '%#.2f'),"]")
Results[ind+1,"ab"]<-paste0("[",sprintf(estimatesCML$ci.lower[estimatesCML$label=="ab"], fmt = '%#.2f'),"; ",sprintf(estimatesCML$ci.upper[estimatesCML$label=="ab"], fmt = '%#.2f'),"]")

Results[ind,"omega1"]<-sprintf(sum(estimatesCML$std.all[estimatesCML$label%in%c("l1","l2","l3")])^2/(sum(estimatesCML$std.all[estimatesCML$label%in%c("l1","l2","l3")])^2+sum(1-estimatesCML$std.all[estimatesCML$label%in%c("l1","l2","l3")]^2)), fmt = '%#.2f')
Results[ind,"omega2"]<-sprintf(sum(estimatesCML$std.all[estimatesCML$label%in%c("l4","l5","l6","l7")])^2/(sum(estimatesCML$std.all[estimatesCML$label%in%c("l4","l5","l6","l7")])^2+sum(1-estimatesCML$std.all[estimatesCML$label%in%c("l4","l5","l6","l7")]^2)), fmt = '%#.2f')
Results[ind,"omega3"]<-sprintf(sum(estimatesCML$std.all[estimatesCML$label%in%c("l8","l9","l10","l11")])^2/(sum(estimatesCML$std.all[estimatesCML$label%in%c("l8","l9","l10","l11")])^2+sum(1-estimatesCML$std.all[estimatesCML$label%in%c("l8","l9","l10","l11")]^2)), fmt = '%#.2f')


ind<-ind+2

# MCMC
#=====================================================

X<-as.matrix(X)
ones.vec <- matrix(1,N, 1)
S<-t(X)%*%X - 1/N*(t(X)%*%ones.vec)%*%(t(ones.vec)%*%X) # SSCP matrix as in Carrol & green, 1997

# starting values
init_fun <- function(){list(lambda1 = runif(3,0.01,.99),
                            lambda2 = runif(8,0.01,.99)
)} 

# alternative: use estimates from CML as starting values
# init_fun <- function(){list(correl12Star = (parameterestimates(fitCML)$est[parameterestimates(fitCML)$label=="cor12"]+0.99)/1.98,
#correl13Star =  (parameterestimates(fitCML)$est[parameterestimates(fitCML)$label=="cor13"]+0.99)/1.98,
#correl23Star =  (parameterestimates(fitCML)$est[parameterestimates(fitCML)$label=="cor23"]+0.99)/1.98,lambda1 = parameterestimates(fitCML,standardized = T)$std.all[parameterestimates(fitCML)$label%in%c("l4","l8","l1")],
#                            lambda2 = parameterestimates(fitCML,standardized = T)$std.all[parameterestimates(fitCML)$label%in%c("l5","l6","l7","l9","l10","l11","l2","l3")]
#)} 

model = stan_model(file = 'BayesEmpiricalExample.stan')


# uniform prior for correlations
#---------------------------------------------

Med_list<- list(
  J  = ncol(X), # number of items
  N  = nrow(X), # number of persons
  dd = c(rep(2,4),rep(3,4),rep(1,3)), # dimension identifier
  S = S, # scatter matrix
  xbar = colMeans(X), # indicator means
  n0C = 0, # prior sample size
  pGC = rep(0,3) # prior guess
) 

fitMed <- sampling(model,  data = Med_list, iter=50000, chains = 3, control = list(adapt_delta = 0.99), init=init_fun, warmup=10000) 
sumMed<-summary(fitMed)[[1]] 

cor12<-sumMed["correl[1,2]",1]
cor13<-sumMed["correl[1,3]",1]
cor23<-sumMed["correl[2,3]",1]

a<-cor12
b<-(cor23-cor13*cor12)/(1-cor12^2);
c<-(cor13-cor23*cor12)/(1-cor12^2);
ab<-a*b;

Results[ind:(ind+1),"specification"]<-rep("BayesUnif",2)
Results[ind,"a"]<-sprintf(a, fmt = '%#.2f')
Results[ind,"b"]<-sprintf(b, fmt = '%#.2f')
Results[ind,"c"]<-sprintf(c, fmt = '%#.2f')
Results[ind,"ab"]<-sprintf(ab, fmt = '%#.2f')
Results[ind+1,"a"]<-paste0("[",sprintf(sumMed["a",4], fmt = '%#.2f'),"; ",sprintf(sumMed["a",8], fmt = '%#.2f'),"]")
Results[ind+1,"b"]<-paste0("[",sprintf(sumMed["b",4], fmt = '%#.2f'),"; ",sprintf(sumMed["b",8], fmt = '%#.2f'),"]")
Results[ind+1,"c"]<-paste0("[",sprintf(sumMed["c",4], fmt = '%#.2f'),"; ",sprintf(sumMed["c",8], fmt = '%#.2f'),"]")
Results[ind+1,"ab"]<-paste0("[",sprintf(sumMed["ab",4], fmt = '%#.2f'),"; ",sprintf(sumMed["ab",8], fmt = '%#.2f'),"]")

Results[ind,"omega1"]<-sprintf(sum(sumMed[paste0("lambda[",9:11,"]"),1])^2/(sum(sumMed[paste0("lambda[",9:11,"]"),1])^2+sum(1-sumMed[paste0("lambda[",9:11,"]"),1]^2)), fmt = '%#.2f')
Results[ind,"omega2"]<-sprintf(sum(sumMed[paste0("lambda[",1:4,"]"),1])^2/(sum(sumMed[paste0("lambda[",1:4,"]"),1])^2+sum(1-sumMed[paste0("lambda[",1:4,"]"),1]^2)), fmt = '%#.2f')
Results[ind,"omega3"]<-sprintf(sum(sumMed[paste0("lambda[",5:8,"]"),1])^2/(sum(sumMed[paste0("lambda[",5:8,"]"),1])^2+sum(1-sumMed[paste0("lambda[",5:8,"]"),1]^2)), fmt = '%#.2f')

ind<-ind+2


# weakly informative prior for correlations
#---------------------------------------------

for(n in c(3,10)){
  for(g in list(rep(0,3),
                c(0.3,0.3,0.5))){
    
    Med_list<- list(
      J  = ncol(X), # number of items
      N  = nrow(X), # number of persons
      dd = c(rep(2,4),rep(3,4),rep(1,3)), # dimension identifier
      S = S, # scatter matrix
      xbar = colMeans(X), # indicator means
      n0C = n, # prior sample size
      pGC = g # prior guess
    ) 
    
    fitMed <- sampling(model,  data = Med_list, iter=50000, chains = 3, control = list(adapt_delta = 0.99), init=init_fun, warmup=10000) 
    sumMed<-summary(fitMed)[[1]] 
    
    cor12<-sumMed["correl[1,2]",1]
    cor13<-sumMed["correl[1,3]",1]
    cor23<-sumMed["correl[2,3]",1]
    
    a<-cor12
    b<-(cor23-cor13*cor12)/(1-cor12^2);
    c<-(cor13-cor23*cor12)/(1-cor12^2);
    ab<-a*b;
    
    Results[ind:(ind+1),"specification"]<-rep(paste0("Bayes",n,"_",g[1]),2)
    Results[ind,"a"]<-sprintf(a, fmt = '%#.2f')
    Results[ind,"b"]<-sprintf(b, fmt = '%#.2f')
    Results[ind,"c"]<-sprintf(c, fmt = '%#.2f')
    Results[ind,"ab"]<-sprintf(ab, fmt = '%#.2f')
    Results[ind+1,"a"]<-paste0("[",sprintf(sumMed["a",4], fmt = '%#.2f'),"; ",sprintf(sumMed["a",8], fmt = '%#.2f'),"]")
    Results[ind+1,"b"]<-paste0("[",sprintf(sumMed["b",4], fmt = '%#.2f'),"; ",sprintf(sumMed["b",8], fmt = '%#.2f'),"]")
    Results[ind+1,"c"]<-paste0("[",sprintf(sumMed["c",4], fmt = '%#.2f'),"; ",sprintf(sumMed["c",8], fmt = '%#.2f'),"]")
    Results[ind+1,"ab"]<-paste0("[",sprintf(sumMed["ab",4], fmt = '%#.2f'),"; ",sprintf(sumMed["ab",8], fmt = '%#.2f'),"]")
    
    Results[ind,"omega1"]<-sprintf(sum(sumMed[paste0("lambda[",9:11,"]"),1])^2/(sum(sumMed[paste0("lambda[",9:11,"]"),1])^2+sum(1-sumMed[paste0("lambda[",9:11,"]"),1]^2)), fmt = '%#.2f')
    Results[ind,"omega2"]<-sprintf(sum(sumMed[paste0("lambda[",1:4,"]"),1])^2/(sum(sumMed[paste0("lambda[",1:4,"]"),1])^2+sum(1-sumMed[paste0("lambda[",1:4,"]"),1]^2)), fmt = '%#.2f')
    Results[ind,"omega3"]<-sprintf(sum(sumMed[paste0("lambda[",5:8,"]"),1])^2/(sum(sumMed[paste0("lambda[",5:8,"]"),1])^2+sum(1-sumMed[paste0("lambda[",5:8,"]"),1]^2)), fmt = '%#.2f')
    
    ind<-ind+2
  }
}


# manifest/ SI 
#======================================================

X<-as.data.frame(X)
c1<-rowMeans(X[paste0("x",1:3)]) # composites
c2<-rowMeans(X[paste0("y",1:4)])
c3<-rowMeans(X[paste0("y",5:8)])
Cdata<-data.frame(cbind(c1,c2,c3)) 
PathMod<-'
  F1 =~ NA*c1+ l1*c1   
  F2 =~ NA*c2+ l2*c2    
  F3 =~ NA*c3+ l3*c3    
  F2 ~~ cor12*F1    
  F3 ~~ cor13*F1
  F2 ~~ cor23*F3
  F1 ~~ 1*F1
  F2 ~~ 1*F2
  F3 ~~ 1*F3
  l1 > 0.01
  l2 > 0.01
  l3 > 0.01
  cor12 > -.99
  cor13 > -.99
  cor23 > -.99
  cor12 < .99
  cor13 < .99
  cor23 < .99
  # constraint positive definite
  1+2*cor12*cor13*cor23-cor12^2-cor13^2-cor23^2>0.01
  a := cor12
  b := (cor23-cor13*cor12)/(1-cor12^2)
  c := (cor13-cor23*cor12)/(1-cor12^2) 
  ab : = a*b
'
# add fixed reliabilities
vars<-diag(cov(Cdata))*(N-1)/N # variances of observed variables     

for(relSI in c(0.5,0.8,1)){
  
  addtoPathMod<-NULL
  for (k in (1:3)) {
    part<-paste("c",k,"~~",(1-relSI)*vars[k],"*c",k," \n ",sep="")
    addtoPathMod<-paste(addtoPathMod,part,sep=" ")
  }
  SI<-paste(PathMod,"\n", addtoPathMod,sep=" ") 
  fitSI<-sem(SI,data=Cdata,se="boot",bootstrap=1000) # bootstrap
  
  estimatesSI<-parameterestimates(fitSI, boot.ci.type="perc",standardized = T)
  
  Results[ind:(ind+1),"specification"]<-rep(paste0("SI",relSI*10),2)
  Results[ind,"a"]<-sprintf(estimatesSI$est[estimatesSI$label=="a"], fmt = '%#.2f')
  Results[ind,"b"]<-sprintf(estimatesSI$est[estimatesSI$label=="b"], fmt = '%#.2f')
  Results[ind,"c"]<-sprintf(estimatesSI$est[estimatesSI$label=="c"], fmt = '%#.2f')
  Results[ind,"ab"]<-sprintf(estimatesSI$est[estimatesSI$label=="ab"], fmt = '%#.2f')
  Results[ind+1,"a"]<-paste0("[",sprintf(estimatesSI$ci.lower[estimatesSI$label=="a"], fmt = '%#.2f'),"; ",sprintf(estimatesSI$ci.upper[estimatesSI$label=="a"], fmt = '%#.2f'),"]")
  Results[ind+1,"b"]<-paste0("[",sprintf(estimatesSI$ci.lower[estimatesSI$label=="b"], fmt = '%#.2f'),"; ",sprintf(estimatesSI$ci.upper[estimatesSI$label=="b"], fmt = '%#.2f'),"]")
  Results[ind+1,"c"]<-paste0("[",sprintf(estimatesSI$ci.lower[estimatesSI$label=="c"], fmt = '%#.2f'),"; ",sprintf(estimatesSI$ci.upper[estimatesSI$label=="c"], fmt = '%#.2f'),"]")
  Results[ind+1,"ab"]<-paste0("[",sprintf(estimatesSI$ci.lower[estimatesSI$label=="ab"], fmt = '%#.2f'),"; ",sprintf(estimatesSI$ci.upper[estimatesSI$label=="ab"], fmt = '%#.2f'),"]")
  
  ind<-ind+2

}


library(xtable)
print(xtable(Results[,1:5]),include.rownames=FALSE)
