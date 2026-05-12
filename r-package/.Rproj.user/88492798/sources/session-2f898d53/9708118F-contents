#x1 <- lav_ugamma_nested_2000(m0, m1)
#x2 <- lav_ugamma_nested_2001(m0, m1)

#x = sort(Re(eigen(x1)$values))
#y = sort(Re(eigen(x2)$values))

hs_model <- " visual  =~ x1 + x2 + x3
              textual =~ x4 + x5 + x6
              speed   =~ x7 + x8 + x9 "

## Estimation that IS allowed.
m1 <- lavaan::cfa(hs_model,
                  data = lavaan::HolzingerSwineford1939,
                  group = "school", estimator = "MLM"
)

m0 <- lavaan::cfa(hs_model,
                  data = lavaan::HolzingerSwineford1939,
                  group = "school", estimator = "ML", group.equal = "loadings", test = "satorra.bentler"
)

pvalues(m0, tests = "SB_UG")


pvalues_nested(m0, m1, tests = "SB", method = "2001")
lavaan::anova(m1, m0, method = "satorra.bentler.2001")


lavaan::anova(m1, m0, method = "satorra.2000", scaled.shifted = FALSE)
pvalues_nested(m0, m1, tests = "SB")


tr(ugamma_nested(m0, m1, method = "2001")[[1]])
tr(ugamma_nested(m0, m1, method = "2000")[[1]])


test = 2

T1 <- m1@test[[1]]$stat
r1 <- m1@test[[1]]$df
c1 <- m1@test[[test]]$scaling.factor

T0 <- m0@test[[1]]$stat
r0 <- m0@test[[1]]$df
c0 <- m0@test[[test]]$scaling.factor



u0 <- lavaan::lavInspect(m0, "U")
u1 <- lavaan::lavInspect(m1, "U")

sum(diag(ugamma_nested(m0, m1, method = "2001")[[1]]))
tr((u0 - u1) %*% get_gamma(m1, FALSE))
sum(sort(Re(eigen((u0 - u1) %*% get_gamma(m1, FALSE))$values), decreasing = TRUE)[1:m])
(r0 * c0 - r1 * c1)
m <- r0 - r1


diff <- make_chisqs("ml", m0, m1)
a1 <- tr((u0 - u1) %*% get_gamma(m1, FALSE))
a2 <- sum(sort(Re(eigen((u0 - u1) %*% get_gamma(m1, FALSE))$values), decreasing = TRUE)[1:m])
m <- r0 - r1

cd1 <- a1 / m
cd2 <- a2 / m

1 - pchisq(diff / cd1, m)
1 - pchisq(diff / cd2, m)

pvalues_nested(m0, m1, tests = "SB", method = "2001")
lavaan::anova(m1, m0, method = "satorra.bentler.2001")


(r0 * c0 - r1 * c1) / 2





test = 2

T0 <- m0@test[[1]]$stat
r0 <- m0@test[[1]]$df
c0 <- m0@test[[test]]$scaling.factor

# m = difference between the df's
m <- r0 - r1

# check for identical df setting
if (m == 0L) {
  return(list(
    T.delta = (T0 - T1), scaling.factor = as.numeric(NA),
    df.delta = m
  ))
}

# compute c_d
cd <- (r0 * c0 - r1 * c1) / m

# warn if cd is negative
if (cd < 0) {
  lav_msg_warn(gettext("scaling factor is negative"))
  cd <- as.numeric(NA)
}

# compute scaled difference test
T.delta <- (T0 - T1) / cd









tr <- \(x) sum(diag(x))
sum(diag(lavInspect(m0, "UGamma")))
tr(u0 %*% get_gamma(m1, FALSE))
r0 * c0

sum(diag(lavInspect(m1, "UGamma")))
r1 * c1

(u0 - u1) %*% get_gamma(m1, FALSE)
u0 <- lavaan::lavInspect(m0, "U")
u1 <- lavaan::lavInspect(m1, "U")

u0 %*% get_gamma(m1, FALSE)

(u0 - u1) %*% get_gamma(m1, FALSE)












pvalues_nested(m0, m1, tests = "SB_UG", method = "2000")
pvalues_nested(m0, m1, tests = "SB_UG_RLS", method = "2000")
pvalues_nested(m0, m1, tests = "SB_RLS", method = "2000")
pvalues_nested(m0, m1, tests = "SB", method = "2000")


pvalues_nested(m0, m1, tests = "SB_UG", method = "2001")
pvalues_nested(m0, m1, tests = "SB_UG_RLS", method = "2001")
pvalues_nested(m0, m1, tests = "SB_RLS", method = "2001")
pvalues_nested(m0, m1, tests = "SB", method = "2001")

pvalues_nested(m0, m1, tests = "SB", method = "2001")


pvalues_nested(m0, m1, tests = "SB", method = "2000")
lavaan::anova(m1, m0, method = "satorra.2000")
diff = lavaan::fitmeasures(m0, "chisq") - lavaan::fitmeasures(m1, "chisq")


pvalues_nested(m0, m1, tests = "SB", method = "2001")
lavaan::anova(m1, m0, method = "satorra.bentler.2001")
diff = lavaan::fitmeasures(m0, "chisq") - lavaan::fitmeasures(m1, "chisq")

pvalues_nested(m0, m1, tests = "EBA1", method = "2001")
pvalues_nested(m0, m1, tests = "SB", method = "2001")

r0 <- m0@test[[2]]$df
r1 <- m1@test[[2]]$df
c0 <- m0@test[[2]]$scaling.factor
c1 <- m1@test[[2]]$scaling.factor
cd <- (r0 * c0 - r1 * c1) / (r0 - r1)
diff / cd


r0 * c0 - r1 * c1
ll = ugamma_nested(m0, m1, "2001", 1)
sum(diag(ll[[1]]))

ug0 = lavaan::lavInspect(m0,"UGamma")
sum(diag(ug0))
r0*c0

ug1 = lavaan::lavInspect(m1,"UGamma")
sum(diag(ug1))
r1*c1

sum(diag(ug0 - ug1))


#

u0 <- lavaan::lavInspect(m0, "U")
u1 <- lavaan::lavInspect(m1, "U")
ug_diff_2  <- (u0 - u1) %*% get_gamma(m1, 1)


x1 <- get_gamma(m1, 1)
x2 <-lavaan::lavInspect(m1, "gamma")
x2_ <- lavaan:::lav_matrix_bdiag(x2)

sum(abs(x1-x2_))

sum(abs(x1 - x2_))
sum(diag(ug_diff_2))L

unbiased = FALSE
lavoptions = lavaan::lavInspect(m1, "options")
lavdata = m1@Data
lavoptions$gamma.unbiased <- unbiased
gamma_list <- lavaan:::lav_samplestats_from_data(lavdata, lavoptions)@NACOV

xx <- as.matrix(Matrix::bdiag(gamma_list))

sum(abs(xx - x2_))


#########

pvalues(m0, tests = "SB", extras = TRUE)

sum(abs(ugamma_nested(m0, m1, "2000", 2)[[1]] - ugamma_nested(m0, m1, "2000", 1)[[1]]))

-sort(-Re(eigen(x2)$values))[1:6]
m1
min(Re(eigen(x2)$values))


make_chisqs(c("rls"), m0)

lavaan::lavTest(object, test = "browne.residual.nt")


m0@Options$gamma.unbiased <- FALSE
unbiased <- lavaan:::lav_object_inspect_UGamma(m0)
m0@Options$gamma.unbiased <- TRUE
biased <- lavaan:::lav_object_inspect_UGamma(m0)

m0@Options$gamma.unbiased <- TRUE
out <- lavaan:::lav_test_satorra_bentler(lavobject     = m0,
                                method        = "original",
                                return.ugamma = TRUE)
biased <- out$UGamma

out <- lavaan:::lav_test_satorra_bentler(lavobject     = m0,
                                method        = "original",
                                return.u      = TRUE)
u = out$UfromUGamma



lavoptions = lavaan::lavInspect(object, "options")
lavdata = object@Data
lavoptions$gamma.unbiased <- TRUE

#| Extract gamma from lavaan object.
get_gamma <- \(object, unbiased = FALSE) {
  lavoptions = lavaan::lavInspect(object, "options")
  lavdata = object@Data
  lavoptions$gamma.unbiased <- unbiased
  gamma_list <- lavaan:::lav_samplestats_from_data(lavdata, lavoptions)@NACOV
  lavaan:::lav_matrix_bdiag(gamma_list)
}

sum(abs(get_gamma(object, TRUE) - get_gamma(object, FALSE)))

u <- lavaan:::lav_matrix_bdiag(lavaan::lavInspect(object, "gamma"))
sum(u - ugamma)
