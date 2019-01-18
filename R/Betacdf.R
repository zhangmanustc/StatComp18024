#' @title A function to compute a Monte Carlo estimate of the Beta(a, b) cdf
#' @description A function to compute a Monte Carlo estimate of the Beta(a, b) cdf
#' @param alpha shape parameter
#' @param beta shape parameter
#' @param x observations
#' @return a Monte Carlo estimate of the Beta(a, b) cdf
#' @examples
#' \dontrun{
#'Betacdf(5,3,3)
#' }
#' @export

Betacdf <- function( x, alpha, beta) {#construct cdf of Beta(a,b) by function()
  m<-1e3
  if ( any(x < 0) ) return (0)
  stopifnot( x < 1 )#the range of x is between 0 and 1
  t <- runif( m, min=0, max=x )#generate samples from U(0,x)
  p<-(x-0)*(1/beta(alpha,beta)) * t^(alpha-1) * (1-t)^(beta-1)
  cdf<-mean(p)#compute Monte Carlo estimate of cdf
  return( min(1,cdf) )#ensure that cdf<=1
} #end function

