# Independently written models for the three all-ordinal corpus slices.
# Mplus input defines meaning; these references are not frontend projections.
mplus_categorical_reference <- function(id) {
if(grepl('ex5_19$',id)) s<-paste('a1 =~ c(a,a)*u1; a2 =~ c(a,a)*u2; c1 =~ c(c,c)*u1; c2 =~ c(c,c)*u2','a1 ~~ c(1,1)*a1; a2 ~~ c(1,1)*a2; c1 ~~ c(1,1)*c1; c2 ~~ c(1,1)*c2','a1 ~~ c(1,0.5)*a2; c1 ~~ c(1,1)*c2','a1 ~ c(0,0)*1; a2 ~ c(0,0)*1; c1 ~ c(0,0)*1; c2 ~ c(0,0)*1','u1 | c(t,t)*t1; u2 | c(t,t)*t1','u1 ~ c(0,0)*1; u2 ~ c(0,0)*1','u1 ~*~ c(1,1)*u1; u2 ~*~ c(1,1)*u2',sep='\n') else if(grepl('ex5_10$',id)) s<-'f1 =~ 1*u1a + 1*u1b + 1*u1c; f2 =~ 1*u2a + 1*u2b + 1*u2c\nf1 ~~ f1; f2 ~~ f2; f1 ~~ f2\nf1 ~ 0*1; f2 ~ 0*1\nu1a | a*t1; u1b | a*t1; u1c | a*t1\nu2a | b*t1; u2b | b*t1; u2c | b*t1' else s<-'f1 =~ 1*u1+u2+u3; f2 =~ 1*u4+u5+u6\nf1 ~~ f1; f2 ~~ f2; f1 ~~ f2\nf1 ~ 0*1; f2 ~ 0*1\nu1 | t1; u2 | t1; u3 | t1; u4 | t1; u5 | t1; u6 | t1'
 s
}
