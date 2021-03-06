% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/chisq_test2.R
\name{chisq_test2}
\alias{chisq_test2}
\title{Make a faster version of chisq.test()}
\usage{
chisq_test2(x, y)
}
\arguments{
\item{x}{vector x}

\item{y}{vector y}
}
\value{
compute the chi-square test statistic to test the independence of observations in two vectors.
}
\description{
Make a faster version of chisq.test() when the input is two numeric vectors with no missing values.
}
\examples{
\dontrun{
chisq_test2(seq(1, 4,0.5), seq(2, 5,0.5))
}
}
