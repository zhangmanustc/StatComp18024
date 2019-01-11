# StatComp18024

## Overview
__StatComp18024__ is a simple R package developed to present the homework of statistic computing course by author 18024.

1.Antithetic.method:The antithetic variables method to variance reducing in samples from a Rayleigh distribution

2.Betacdf:A function to compute a Monte Carlo estimate of the Beta(a, b) cdf

3.chisq_test2:Make a faster version of chisq.test() when the input is two numeric vectors with no missing values.


__Antithetic.method__

The source R code for _Antithetic.method_ is as follows:

```{r,eval=FALSE}
Antithetic.method<- function(sigma, n) {#construct funtions to generate samples from  Rayleigh distribution
  X <- X_ <- numeric(n)
  for (i in 1:n) {
    U<- runif(n)#generate n observations from U~U(0,1)
    V<- 1-U#then V~U(0,1)
    X<- sigma*sqrt(-2*log(V))
    X_<-sigma*sqrt(-2*log(U))
    #according inverse transform from annalysis above,we obtain that $X=F_{X}^{-1}(U)=\sqrt{-2ln(1-U)}=\sqrt{-2ln(V)}$\qquad
    var1<-var(X)#if X1 and X2 are independent,then var((X1+X2)/2)=var(X)
    var2<-(var(X)+var(X_)+2*cov(X,X_))/4#compute the variance of (X+X')/2 generated by antithetic variables
    reduction <-((var1-var2)/var1)#compute the reduction in variance by using antithetic variables
  }
  percent_reduction<-paste0(format(100*reduction, format = "fg", digits = 4), "%")#set format type by function format
  return(percent_reduction)
}
set.seed(123)
sigma = 1#set the value of sigma-the scale parameter of Rayleigh distribution to 1
n <- 1000
Antithetic.method(sigma, n)
```
 Analysis of results:
We conclude that the percent reduction in variance, compared with X+X'/2qquad for independent X1+X2/2 is 97.15%,so the antithetic variables method to variance reducing is of significant effect.




__Betacdf__

The source R code for _Betacdf_ is as follows:
```{r}
Betacdf <- function( x, alpha=3, beta=3) {#construct cdf of Beta(3,3) by function()
m<-1e3
if ( any(x < 0) ) return (0)
stopifnot( x < 1 )#the range of x is between 0 and 1
t <- runif( m, min=0, max=x )#generate samples from U(0,x)
p<-(x-0)*(1/beta(alpha,beta)) * t^(alpha-1) * (1-t)^(beta-1)
cdf<-mean(p)#compute Monte Carlo estimate of cdf
return( min(1,cdf) )#ensure that cdf<=1
} #end function

set.seed(123)
for (i in 1:9) {
estimate<-Betacdf(i/10,3,3)#use function constructed to estimate F(x) when x=0.1,0.2,...,0.9
returnedvalues<-pbeta(i/10,3,3)#obtain values returned by the pbeta function
print( c(estimate,returnedvalues) )#compare the eatimates with returned values
} #end for loop


```
Analysis of results:
From the results above,the estimates(left) are very close to the actual value(right),they differ from the fourth decimal place,thus the estimation accuracy is very high.



__chisq_test2__
The source R code for _chisq_test2_ is as follows:

```{r,eval=FALSE}
#construct function to compute  the expected (theoretical) count
expected <- function(colsum, rowsum, total) {
  (colsum / total) * (rowsum / total) * total
}
#construct function to compute the value of the test-statistic
chi_stat <- function(observed, expected) {
  ((observed - expected) ^ 2) / expected
}


chisq_test2 <- function(x, y) {
  total <- sum(x) + sum(y)
  rowsum_x <- sum(x)
  rowsum_y <- sum(y)
  chistat <- 0
# computes the chi-square test statistic which is apparently different from chisq.test function
  for (i in seq_along(x)) {
    colsum <- x[i] + y[i]
    expected_x <- expected(colsum, rowsum_x, total)
    expected_y <- expected(colsum, rowsum_y, total)
    chistat <- chistat + chi_stat(x[i], expected_x)
    chistat <- chistat + chi_stat(y[i], expected_y)
  }
  chistat
}
o <- as.table(rbind(c(56, 67, 57,34), c(135, 125, 45,65)))
#compare the results of chi-square test
print(chisq_test2(c(56, 67, 57,34), c(135, 125, 45,65)))
print(chisq.test(as.table(rbind(c(56, 67, 57,34), c(135, 125, 45,65)))))
#compare the computing time
print(microbenchmark::microbenchmark(
  chisq_test2(c(56, 67, 57,34), c(135, 125, 45,65)),
  chisq.test(as.table(rbind(c(56, 67, 57,34), c(135, 125, 45,65))))
))
```
**Analysis of results:**
**First**:
According to the reults above,we can find that chisq-square statistics of the function we constructed is 0.04762132,which is less than $\chi_{0.95}(4)$\qquad,and the p-value of chisq.test() is  0.2243>0.05,so two functions obtain the same result to accept the null hypothesis.
**Second**:
According to the time comparison,we can find the function we constructed is much quicker than chisq.test().



