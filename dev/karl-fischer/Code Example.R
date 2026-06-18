#Installing all relevant packages. 
install.packages(c("tidyverse","lavaan","psych", "semTools","lordif","sirt", "devtools", "car"))

#Installing the prototype of the ccpsyc package
devtools::install_github("Jo-Karl/ccpsyc")

# Loading Necessary Libraries ---------------------------------------------
## Should you get an error that a certain library is not available install it 
## via install.packages(), i.e install.packages("tidyverse").
## Spaces and linebreaks do not matter in R in most cases.

library(tidyverse)
library(lavaan)
library(psych)
library(semTools)
library(lordif)
library(sirt)
library(ccpsyc)
# Loading Data ------------------------------------------------------------
## Either set the working directory to the folder with your data or use R 
## studio projects.
overall <- read.csv("example.csv") 
example <-  overall %>%
  filter(., country %in% c("BRA", "NZ"))



# Multi-Group Confirmatory CFA --------------------------------------------
## In the next step we define the model we want to test across cultures
## latent variables are created using =~ .
cfa_model<- '
              help =~ help1 + help2 + help3 + help4 + help5 + help6 + help7
              voice =~ voice1 + voice2 + voice3 + voice4 + voice5
            '

## Fitting an overall CFA
fit_cfa <- cfa(cfa_model, data = example)
## Fitting a CFA seperated by country

fit_cfa_country <- cfa(cfa_model, data = example, group = "country")

## Fit inidices of the CFA
summary(fit_cfa_country, fit.measures = TRUE,
        standardized = TRUE, rsquare = TRUE)

# Testing Measurement invariance using an omnibus test. -------------------

measurementInvariance(model = cfa_model, data = example, group = "country")

## The measurementInvariance function of the SEMtools package has been updated
## and might become deprecated in the future.
## We provide a legacy version of this function (invariance_semtools)
## in the ccpsyc package. The usage is identical
## to the measurementInvariance function.

invariance_semtools(model = cfa_model, data = example, group = "country")
## Obtaining additional fit measures. It is important to note that all 
## models are fitted using an MLM estimation for multi-variate non-normal
## data. 
equival(cfa_model, dat = example, group = "country")

## Getting Modification indices for the specified model

mi <- modificationIndices(fit_cfa, minimum.value = 10, sort. = T)


# Testing Partial Measurement Invariance ----------------------------------
## The commnand below fits a metrically constrained model.
metric_test <-cfa(cfa_model, data = example,
                  group = "country", group.equal = c("loadings"))

## Requesting modificaiton indices for the metrically constrained model
mi_metric <- modificationIndices(metric_test, minimum.value = 10, sort. = T)
## Testing which loadings could be released between groups
lavTestScore.clean(metric_test)

fit_partial <
  - cfa(cfa_model, data = example, group = "country", group.equal = c("loadings"), group.partial = c("help =~ help3 "))

# Dmacs -------------------------------------------------------------------
## It is important to note that currently Dmacs are only supported for single
## factor models.

help_model <- 'help =~ help1 + help2 + help3 + help4 + help5 + help6 + help7'
help_cfa <- cfa(help_model, data = example, group = "country")
dMACS(help_cfa, group1 = "NZ", group2 = "BRA")


# Exploratory Strucutural Equation Modelling ------------------------------
## Creating a factor rotated object (exlduing the country column)
beh_efa <- fa(example[-1], nfact = 2, rotate = "geominQ", fm = "ml")

beh_loadmat <- zapsmall(matrix(round(beh_efa$loadings, 2), nrow = 12, ncol = 2))
rownames(beh_loadmat) <- colnames(example[-1]) 


new_model <- vector()
for (i in 1:2) {
  new_model[i] <- paste0("F",i,"=~ ", paste0(c(beh_loadmat[,i]), " * ", names(beh_loadmat[,1]), collapse = " + "))
}


## The subsequent step fits a classic CFA on the exploratory model/
beh_cfa_esem <- cfa(new_model, data = example, estimator = "ML") 
fitmeasures(beh_cfa_esem, c("cfi", "tli", "rmsea", "srmr"))

## Next we fit a configural model for each country

fitmeasures(cfa(
  model = new_model,
  data = example,
  group = "country",
  estimator = "ML"),
  c("cfi","tli","rmsea","srmr")
)


## Metric and Scalar Models can be fitted as for a normal CFA

fitmeasures(cfa(
  model = new_model,
  data = example,
  group = "country",
  estimator = "ML",
  group.equal = c("loadings", "intercepts")),
  c("cfi","tli","rmsea","srmr")
)
# Approximate Invariance Using Alignement ----------------------------------
## Below is example code to test the alignement procedure in R.

par <- invariance_alignment_cfa_config(dat = example[paste0("help", 1:7)], group = example$country)
mod1 <- invariance.alignment(lambda = par$lambda, nu = par$nu, align.scale = c(.2, .4), align.pow = c(.25, .25))
mod1$es.invariance["R2",]

cmod1 <- invariance_alignment_constraints(mod1, lambda_parm_tol = .4, nu_parm_tol = .2)
summary(cmod1)

# Logistic Regression to test for DIF in R --------------------------------

country <- car::recode(example$country, "'NZ' = 1; 'BRA' = 0")
response_data <- example[paste0("help", 1:7)]
countryDIF <- lordif(resp.data = response_data, group = country, criterion = "Chisqr",
                     alpha = 0.001, minCell = 5)

countryDIF_r2_change <- lordif(response_data, country,criterion = "R2", R2.change = 0.01, minCell = 5)

countryDIF_MC <- montecarlo(countryDIF, alpha = 0.001, nr=1000)
# EFA and PCA -------------------------------------------------------------
## The following section provides example code  for an EFA or PCA with
## Procrustes Rotation.

nz_example <- filter(example, country == "NZ")
br_example <- filter(example, country == "BRA")

nz_fa <- fa(nz_example[,-1], nfactors = 2, rotate = "varimax")
br_fa <- fa(br_example[,-1], nfactors = 2, rotate = "varimax")


prost(nz_fa$loadings, br_fa$loadings, rotated = T)


# EFA/ PCA for multiple Groups --------------------------------------------

## The steps below  outline the preparation of a multi-country dataset
## for a Procrustes analysis.

split_list <- split(overall, overall$country)
variable_list <- lapply(split_list, function(x){x[,-1]})
output_pca <- lapply(variable_list, function(x){principal(x, rotate = "varimax", nfactors = 2)})

## Creating an ideal matrix 1's represent that the item is loading on the factor
## 0 is indicating that the item is not loading on the factor.
ideal <- matrix(c(1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1,0,1),
                byrow = TRUE, ncol = 2)

## Subsequently each countries loading matrix can be rotated to the ideal matrix
PCA <- lapply(variable_list, function(x){principal(x, rotate = "varimax", nfactors = 2)})
lapply(PCA, function(x){prost(x$loadings, ideal)})


# EFA/PCA Procrustes against a Pan-Cultural Matrix ------------------------

overall$country_numeric <- as.numeric(factor(overall$country))
pooled_correlation <- psych::statsBy(data = overall[-1], group = "country_numeric",
                                     cors = TRUE, cor = "cor", method ="pearson")
pooled_pca <- psych::principal(r = pooled_correlation$pooled, nfactors = 2,
                               rotate = "varimax")
output3_pca <- lapply(variable_list, function(x){principal(x, rotate = "varimax", nfactors = 2)})
lapply(output3_pca, function(x){prost(x$loadings, pooled_pca$loadings)})

