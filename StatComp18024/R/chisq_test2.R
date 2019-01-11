#' @title Make a faster version of chisq.test()
#' @description Make a faster version of chisq.test() when the input is two numeric vectors with no missing values.
#' @param x vector x
#' @param y vector y
#' @return compute the chi-square test statistic to test the independence of observations in two vectors.
#' @examples
#' \dontrun{
#' chisq_test2(seq(1, 4,0.5), seq(2, 5,0.5))
#' }
#' @export
chisq_test2 <- function(x, y) {
  #construct function to compute  the expected (theoretical) count
  expected <- function(colsum, rowsum, total) {
    (colsum / total) * (rowsum / total) * total
  }
  #construct function to compute the value of the test-statistic
  chi_stat <- function(observed, expected) {
    ((observed - expected) ^ 2) / expected
  }
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
