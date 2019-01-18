## ----eval=FALSE----------------------------------------------------------
#  ts(1:10,start=1959)
#  ts(1:47,frequency=12,start=c(1959,2))
#  ts(1:10,frequency=4,start=c(1959,2))
#  ts(matrix(rpois(36,5),12,3),start=c(1961,1),frequency=12)

## ----eval=FALSE----------------------------------------------------------
#  layout(matrix(1:4,2,2))
#  layout.show(4)
#  layout(matrix(1:6,3,2))
#  layout.show(6)
#  layout(matrix(1:6,2,3))
#  layout.show(6)
#  m<-matrix(c(1:3,3),2,2)
#  layout(m)
#  layout.show(3)
#  m<-matrix(1:4,2,2)
#  layout(m,widths=c(1,3),heights=c(3,1))
#  layout.show(4)
#  m<-matrix(c(1,1,2,1),2,2)
#  layout(m,widths=c(2,1),heights=c(1,2))
#  layout.show(2)
#  m<-matrix(0:3,2,2)
#  layout(m,c(1,3),c(1,3))
#  layout.show(3)
#  

## ----eval=FALSE----------------------------------------------------------
#  data(InsectSprays)
#  InsectSprays
#  aov.spray<-aov(sqrt(count)~spray,data=InsectSprays)
#  aov.spray<-aov(sqrt(InsectSprays$count)~InsectSprays$spray)
#  aov.spray<-aov(sqrt(InsectSprays[,1])~InsectSprays[,2])
#  aov.spray
#  summary(aov.spray)
#  opar<-par()
#  par(mfcol=c(2,2))
#  plot(aov.spray)
#  termplot(aov.spray,se=TRUE,partial.resid=TRUE,rug=TRUE)

## ----eval=FALSE----------------------------------------------------------
#  x<-rnorm(10)
#  y<-rnorm(10)
#  plot(x,y)
#  plot(x,y,xlab="Ten random values",ylab="Ten other values",xlim=c(-2,2),ylim=c(-2,2),pch=22,col="red",bg="yellow",bty="l",tcl=0.4,main="How to customize a plot with R",las=1,cex=1.5)
#  opar<-par()
#  par(bg="lightyellow",col.axis="blue",mar=c(4,4,2.5,0.25))
#  plot(x,y,xlab="Ten random values",ylab="Ten other values",xlim=c(-2,2),ylim=c(-2,2),pch=22,col="red",bg="yellow",bty="l",tcl=-.25,las=1,cex=1.5)
#  title("How to customize a plot with R(bis)",font.main=3,adj=1)
#  
#  opar<-par()
#  par(bg="lightgray",mar=c(2.5,1.5,2.5,0.25))
#  plot(x,y,type="n",xlab="",ylab="",xlim=c(-2,2),ylim=c(-2,2),xaxt="n",yaxt="n")
#  rect(-3,-3,3,3,col="cornsilk")
#  points(x,y,pch=10,col="red",cex=2)
#  axis(side=1,c(-2,0,2),tcl=-0.2,labels=FALSE)
#  axis(side=2,-1:1,tcl=-0.2,labels=FALSE)
#  title("How to customize a plot with R(ter)",font.main=4,adj=1,cex.main=1)
#  mtext("Ten random values",side=1,line=1,at=1,cex=0.9,font=3)
#  mtext("Ten other values",line=0.5,at=-1.8,cex=0.9,font=3)
#  mtext(c(-2,0,2),side=1,las=1,at=c(-2,0,2),line=0.3,col="blue",cex=0.9)
#  mtext(-1:1,side=2,las=1,at=-1:1,line=0.2,col="blue",cex=0.9)
#  

## ----eval=FALSE----------------------------------------------------------
#  x<-1:30
#  1:10-1
#  1:(10-1)
#  seq(1,5,0.5)
#  seq(length=9,from=1,to=5)
#  c(1,1.5,2,2.5,3,3.5,4,4.5,5)
#  rep(1,30)
#  sequence(4:5)
#  sequence(c(10,5))
#  gl(3,5)
#  gl(3,5,length=30)
#  gl(2,6,label=c("Male","Female"))
#  gl(2,10)
#  gl(2,1,length=20)
#  gl(2,2,length=20)
#  expand.grid(h=c(60,80),w=c(100,300),sex=c("Male","Female"))
#  
#  

## ----eval=FALSE----------------------------------------------------------
#  x<-c(0,1,2,3,4)#generate vector x=(0,1,2,3,4),let x take the value of 0,1,2,3,4
#  p<-c(0.1,0.2,0.2,0.2,0.3)#generate vector p=(0.1,0.2,0.2,0.2,0.3),Let x get 0,1,2,3,4 with probability of 0.1,0.2,0.2,0.2,0.3 respectively
#  cp<-cumsum(p)#returns to the cumulative sums of vector p
#  m<-1e3#m is set to 1000
#  r<-numeric(m)
#  r<-x[findInterval(runif(m),cp)+1]#find the interval containing samples
#  r#list the observations of generated sample
#  ct<-as.vector(table(r))#calculate the frequency column of different values of the sample
#  ct/sum(ct)/p#get the ratio of frequency to probability

## ----eval=FALSE----------------------------------------------------------
#  #beta(3,2) with pdf $f(x)=12x^{2}(1-x)$\qquad,0<x<1;g(x)=1,0<x<1
#  n<-1e3
#  j<-k<-0#both initial values of j and k are assigned to 0
#  y<-numeric(n)
#  while(k<n){
#    u<-runif(1)#generate random numbers U~U(0,1)
#    j<-j+1
#    x<-runif(1)#random variate from g
#  
#  #when x equals 2/3,the function $f(x)=12x^{2}(1-x)$\qquad attain the maximum 16/9.Due to c can't be too large, or the effect will be very low,thus the value of c is set to be the maximum of f(x).In this case, the number of iterations will be reduced to 1804.
#  
#    if(27/4*x*x*(1-x)>u){   #we accept x when U<=f(Y)/(c*g(Y)),c=16/9
#      k<-k+1
#      y[k]<-x
#    }
#  }
#  y#list the generated random sample of size 1000 from the Beta(3,2) distribution
#  j#(experiments)for n random numbers
#  hist(y,prob=TRUE,main=expression(f(x)==12*x^2*(1-x)))#Graph the histogram of the sample
#  y<-seq(0,1,0.01)
#  lines(y,12*y^2*(1-y),col="red",lwd=1.5)# add the theoretical Beta(3,2) density line

## ----eval=FALSE----------------------------------------------------------
#  n<-1e3#assign n to 1000 to generate 1000 random observations
#  r<-4#set the shape parameter r to 4
#  beta<-2#set scale parameter beta to 4
#  lambda<-rgamma(n,r,beta)#generate variable lambda subject to gamma distribution whose shape parameter is r,and scale parameter is beta
#  #generate 1000 random observations from this mixture
#  x<-rexp(n,rate=lambda)#generation for n observations subject to the exponential distribution with rate lambda
#  x#list the generated 1000 random observations

## ----eval=FALSE----------------------------------------------------------
#  Betacdf <- function( x, alpha=3, beta=3) {#construct cdf of Beta(3,3) by function()
#  m<-1e3
#  if ( any(x < 0) ) return (0)
#  stopifnot( x < 1 )#the range of x is between 0 and 1
#  t <- runif( m, min=0, max=x )#generate samples from U(0,x)
#  p<-(x-0)*(1/beta(alpha,beta)) * t^(alpha-1) * (1-t)^(beta-1)
#  cdf<-mean(p)#compute Monte Carlo estimate of cdf
#  return( min(1,cdf) )#ensure that cdf<=1
#  } #end function
#  
#  set.seed(123)
#  for (i in 1:9) {
#  estimate<-Betacdf(i/10,3,3)#use function constructed to estimate F(x) when x=0.1,0.2,...,0.9
#  returnedvalues<-pbeta(i/10,3,3)#obtain values returned by the pbeta function
#  print( c(estimate,returnedvalues) )#compare the eatimates with returned values
#  } #end for loop
#  
#  

## ----eval=FALSE----------------------------------------------------------
#  Rayleigh_distribution<- function(sigma, n) {#construct funtions to generate samples from  Rayleigh distribution
#    X <- X_ <- numeric(n)
#    for (i in 1:n) {
#      U<- runif(n)#generate n observations from U~U(0,1)
#      V<- 1-U#then V~U(0,1)
#      X<- sigma*sqrt(-2*log(V))
#      X_<-sigma*sqrt(-2*log(U))
#      #according inverse transform from annalysis above,we obtain that $X=F_{X}^{-1}(U)=\sqrt{-2ln(1-U)}=\sqrt{-2ln(V)}$\qquad
#      var1<-var(X)#if X1 and X2 are independent,then var((X1+X2)/2)=var(X)
#      var2<-(var(X)+var(X_)+2*cov(X,X_))/4#compute the variance of (X+X')/2 generated by antithetic variables
#      reduction <-((var1-var2)/var1)#compute the reduction in variance by using antithetic variables
#    }
#    percent_reduction<-paste0(format(100*reduction, format = "fg", digits = 4), "%")#set format type by function format
#    return(percent_reduction)
#  }
#  set.seed(123)
#  sigma = 1#set the value of sigma-the scale parameter of Rayleigh distribution to 1
#  n <- 1000
#  Rayleigh_distribution(sigma, n)

## ----eval=FALSE----------------------------------------------------------
#   x <- seq(1,4,0.01)#generate sequence x
#      g <- (x^2 * exp(-x^2 / 2)) / sqrt(2 *pi)
#      f1 <- exp(-x) #set the first importance function
#      f2 <- 4 / ((1 + x^2) * pi)#set the second importance function
#      gs <- c(expression(g(x)==e^{-x}/(1+x^2)),expression(f[1](x)==e^{-x}),expression(f[2](x)==4/((1+x^2)*pi)))
#      #figure "ratio of g and f" including the funtion of g/f1 and g/f2
#      plot(x, g/f1, type = "l", ylab = "",ylim = c(0,3.2), lwd = 2, lty = 2,col=2,main='(ratio of g and f)')
#      lines(x, g/f2, lty = 3, lwd = 2,col=3)
#      legend("topright", legend = gs[-1],lty = 2:3, lwd = 2, inset = 0.02,col=2:3)

## ----eval=FALSE----------------------------------------------------------
#  g <- function(x) { (x^2/ sqrt(2*pi) * exp(-x^2 / 2))* (x > 1)}#set the function g
#  f1 <- function(x) { exp(-x)}#set the importance funtion f1 and f2
#  f2 <- function(x) {  4 / ((1 + x^2) * pi)* (x >= 1)}
#  m <- 1e4
#  x1 <- rexp(m)#f1 is constructed to be the expression of exponential distribution,so sample x1 should be generated from exponential distribution
#  u <- runif(m) #generate samples from exponential distribution by inverse transform method
#  x2 <- tan(pi *(u+pi^2/16)/ 4)#obtain the function of x2 via inverse transform method
#  x2[which(x2 < 1)] <- 1# set the range of x2,to catch overflow errors in g(x)
#  ratio1<-g(x1) / f1(x1)
#  ratio2<-g(x2) / f2(x2)
#  theta.hat <- se <- numeric(2)
#  theta.hat<- c(mean(ratio1), mean(ratio2))#obtain a MC estimate of the integration
#  se <- c(sd(ratio1), sd(ratio2))#obtain  variance of the estimator
#  rbind(theta.hat, se)

## ----eval=FALSE----------------------------------------------------------
#  n<-20#number of samples
#  m<-1e3#number of MC simulations
#  G1<-G2<-G3<-numeric(m)
#  set.seed(123)
#  for (j in 1:m){# Monte Carlo simulations
#    x1<-sort(rlnorm(n))#sort x generated from standard lognormal
#    x2<-sort(runif(n))#sort x generated from uniform distribution
#    x3<-sort(rbinom(n,size=1,prob=0.1))#sort x generated from Bernoulli(0.1)
#    mu1<-mean(x1);mu2<-mean(x2);mu3<-mean(x3)#mu is replaced by mean(x)
#    z1<-z2<-z3<-numeric(m)
#      for(i in 1:n){
#        z1<-(2*i-n-1)*x1[i]
#        z2<-(2*i-n-1)*x2[i]
#        z3<-(2*i-n-1)*x3[i]
#      }
#      G1[j]<- sum(z1)/(n^2*mu1)
#      G2[j]<- sum(z2)/(n^2*mu2)
#      G3[j]<- sum(z3)/(n^2*mu3)#set the function of Gini ratio
#  }
#  
#  #Estimate by simulation the mean of Ghat
#  meanG1<-mean(G1);meanG2<-mean(G2);meanG3<-mean(G3,na.rm = TRUE)
#  #Estimate by simulation the median of Ghat
#  medianG1<-median(G1);medianG2<-median(G2);medianG3<-median(G3,na.rm = TRUE)
#  #Estimate by simulation the deciles of Ghat
#  decileG1<-quantile(G1,  seq(0, 1, 0.1));decileG2<-quantile(G2,  seq(0, 1, 0.1))
#  decileG3<-quantile(G3,seq(0, 1, 0.1),na.rm = TRUE)
#  #print results
#  print(c(meanG1,medianG1));print(decileG1)#results of the mean, median and deciles of Ghat if X is standard lognormal
#  print(c(meanG2,medianG2));print(decileG2)#results of the mean, median and deciles of Ghat if X~U(0,1)
#  print(c(meanG3,medianG3));print(decileG3)#results of the mean, median and deciles of Ghat if X~Bernoulli(0.1)
#  hist(G1,prob=TRUE,main="x is from standard lognormal")#construct density histograms if X is standard lognormal
#  hist(G2,prob=TRUE,main="x is from U(0,1) distribution")#construct density histograms if X~U(0,1)
#  hist(G3,prob=TRUE,main="x is from Bernoulli(0.1)")#construct density histograms if X~Bernoulli(0.1)
#  

## ----eval=FALSE----------------------------------------------------------
#  n<-20#number of samples
#  m<-1000#number of MC simulations
#  a=0;b=1#set the parameter of lognormal distribution
#  G<-numeric(m)
#  alpha <-0.05
#  set.seed(123)
#  for (j in 1:m){# Monte Carlo simulations
#    x<-sort(rlnorm(n,a,b))#sort x generated from standard lognormal
#    mu<-mean(x)#?? is replaced by mean(x)
#    z<-numeric(m)
#      for(i in 1:n){
#        z<-(2*i-n-1)*x[i]
#      }
#      G[j]<- sum(z)/(n^2*mu)#set the function of Gini ratio
#  
#  }
#  UCL <- mean(G)+qt(1-(alpha/2), df=n-1)*sd(G)/sqrt(n)#obtain the confidence interval upper limit
#  LCL <- mean(G)-qt(1-(alpha/2), df=n-1)*sd(G)/sqrt(n)#obtain the lower confidence interval
#  CI<-(c(LCL,UCL))#results of confidence interval
#  print(CI)
#  # Start another same simulation,and in this circle,we compare each estimation of G to judge if G is between confidence interval,and count it,then
#  for (j in 1:m){# Monte Carlo simulations
#    x<-sort(rlnorm(n,a,b))#sort x generated from standard lognormal
#    mu<-mean(x)#?? is replaced by mean(x)
#    z<-numeric(m)
#      for(i in 1:n){
#        z<-(2*i-n-1)*x[i]
#      }
#      G[j]<- sum(z)/(n^2*mu)#set the function of Gini ratio
#      c=0#Count parameter starts from 0
#        if(G[j]<LCL)c=c
#          else{
#          if(G[j]>UCL)c=c
#            else{
#            c=c+1
#      }#set conditions if G in each simulation is between confidence interval,then c=c+1
#    }
#    }
#  coverage_rate<-c/m#calculate the coverage rate=times of G falling in the interval/numbers of simulation
#  percent_rate<-paste0(format(100*coverage_rate, format = "fg", digits = 3), "%")#set format type by function format

## ----eval=FALSE----------------------------------------------------------
#  
#  library(MASS)#for mvrnorm
#  N <- 1e3 # Number of random samples
#  alpha<-0.05#set the significant level to 0.05
#  p1<-p2<-p3<-numeric(N)
#  set.seed(123)
#  rho <- 0#set the correlation coefficient to be 0 due to the  null hypothesis  of the test is that the correlation coefficient to be 0
#  mu1 <- 0; s1 <- 1#set the mean and standard deviation of variable x
#  mu2 <- 0; s2 <- 1#set the mean and standard deviation of variable x
#  # Parameters for bivariate normal distribution
#  mu <- c(mu1,mu2) # Mean
#  sigma <- matrix(c(s1^2, s1*s2*rho, s1*s2*rho, s2^2),2,2) # Covariance matrix
#  for(i in 1:N){
#  bvn<- mvrnorm(N, mu = mu, Sigma = sigma ) # generate bivariate normal variables from MASS package
#  x<-bvn[,1]#variable x from bivariate normal distribution
#  y<-bvn[,2]#variable y from bivariate normal distribution
#  p1[i]<-cor.test(x,y,method="spearman")$p.value#cauculate the p_value under the nonparametric tests based on Spearman???s rank correlation coefficient ??s
#  p2[i]<-cor.test(x,y,method="kendall")$p.value#cauculate the p_value under the nonparametric tests based on Kendall???s coefficient ??
#  p3[i]<-cor.test(x,y,method="pearson")$p.value#cauculate the p_value under the correlation test
#  }
#  power<-c(mean(p1<=alpha),mean(p2<=alpha),mean(p3<=alpha))#obtain the power(average rate of test that is a significant ) under three tests
#  print(power)

## ----eval=FALSE----------------------------------------------------------
#  library(MASS)#for mvrnorm
#  N <- 1e3 # Number of random samples
#  alpha<-0.05#set the significant level to 0.05
#  p1<-p2<-p3<-numeric(N)
#  set.seed(123)
#  rho <- 0.15#set different correlation coefficient,and after many trials,0.15 is the especial example
#  mu1 <- 0; s1 <-1#set the mean and standard deviation of variable x
#  mu2 <- 0; s2 <-1#set the mean and standard deviation of variable x
#  # Parameters for bivariate normal distribution
#  mu <- c(mu1,mu2) # Mean
#  sigma <- matrix(c(s1^2, s1*s2*rho, s1*s2*rho, s2^2),2,2) # Covariance matrix
#  for(i in 1:N){
#  bvn<- mvrnorm(N, mu = mu, Sigma = sigma ) # generate bivariate normal variables from MASS package
#  x<-bvn[,1]#variable x from bivariate normal distribution
#  y<-bvn[,2]#variable y from bivariate normal distribution
#  p1[i]<-cor.test(x,y,method="spearman")$p.value#cauculate the p_value under the nonparametric tests based on Spearman???s rank correlation coefficient ??s
#  p2[i]<-cor.test(x,y,method="kendall")$p.value#cauculate the p_value under the nonparametric tests based on Kendall???s coefficient ??
#  p3[i]<-cor.test(x,y,method="pearson")$p.value#cauculate the p_value under the correlation test
#  }
#  power<-c(mean(p1<=alpha),mean(p2<=alpha),mean(p3<=alpha))#obtain the power(average rate of test that is a significant ) under three tests
#  print(power)

## ----eval=FALSE----------------------------------------------------------
#  set.seed(1)
#  library(bootstrap)#for law dataset
#  data(law,package = "bootstrap")
#  LSAT <- law$LSAT #the variable LAST of dataset law
#  GPA <- law$GPA#the variable GPA of dataset law
#  n<-length(LSAT)#n means the number of observed sample
#  theta.hat <- cor(LSAT, GPA)#the correlation statistic in Example 7.2
#  theta.jack <- numeric(n)
#  for(i in 1:n){
#    theta.jack[i] <-cor(LSAT[-i], GPA[-i])#compute the jackknife replicates, leave out the ith observation xi
#  }
#  bias.jack <- (n-1)*(mean(theta.jack)-theta.hat)#jackknife estimate of bias
#  se.jack <- sqrt((n-1)*mean((theta.jack-mean(theta.hat))^2))#Jackknife estimate of standard error
#  round(c(original=theta.hat,bias=bias.jack,se=se.jack),5)#display the results of theta.hat,and the estimator of the bias and the standard error

## ----eval=FALSE----------------------------------------------------------
#  set.seed(1)
#  m<-1e3#number of cycles
#  B<-1e3#The number of bootstrap replicates
#  library(boot)#for boot and boot.ci function
#  data(aircondit,package = "boot")#for aircondi dataset
#  for(i in 1:m){
#    boot.obj <- boot(data=aircondit,statistic=function(x,i){mean(x[i,])}, R=B )#generate R bootstrap replicates of a statistic applied to data,statistic is the mean time
#    ci <- boot.ci(boot.obj,type=c("norm","basic","perc","bca"))#generates 4 different types of equi-tailed two-sided nonparametric confidence intervals
#  }
#  print(ci)
#  

## ----eval=FALSE----------------------------------------------------------
#  library(bootstrap)#for scor dataset
#  n <-length( scor[,1] )#number of calculations
#  x<-as.matrix(scor)
#  theta.jack <- numeric(n)
#  data(scor,package = "bootstrap")#attach the data scor
#  theta<-function(x){
#    eigen(cov(x))$values[1]/sum(eigen(cov(x))$values)
#  }#construct funtion theta=lambda1/sum(lambdai)
#  theta.hat <- theta(scor)#compute the statistic to be estimated
#  for(i in 1:n){
#    theta.jack[i] <-theta(x[-i,])#compute the jackknife replicates, leave-one-out estimates
#  }
#  bias.jack <- (n-1)*(mean(theta.jack)-theta.hat)#jackknife estimate of bias
#  se.jack <- sqrt((n-1)*mean((theta.jack-mean(theta.jack))^2))#Jackknife estimate of standard error
#  round(c(original=theta.hat,bias=bias.jack,se=se.jack),6)#display the results of theta.hat,and the estimator of the bias and the standard error

## ----eval=FALSE----------------------------------------------------------
#  data(ironslag, package = "DAAG")#for data ironslag
#  magnetic<-ironslag$magnetic#extract variable magnetic
#  chemical<-ironslag$chemical#extract variable chemical
#  n <- length(magnetic)
#  p<-n#set number of outer circulation
#  q<-n-1#set number of inner circulation
#  a<-array(dim=c(p,q))
#  e1 <- e2 <- e3 <- e4<-a
#  # for n-fold cross validation
#  #fit models on leave-two-out samples
#  for (k in 1:p) {#outer circulations from  1 to n
#    y1 <- magnetic[-k]#y1 is magnetic leave yk out
#    x1<- chemical[-k]#x1 is chemical leave xk out
#  for(l in 1:q){#inner circulations from 1 to n-1
#    y <- y1[-l]#y is y1 leave yl out
#    x <- x1[-l]#x is x1 leave xl out
#  #model selection:cross validation
#  #1.linear model and estiamtion of prediction error
#    J1 <- lm(y ~ x)
#    yhat1 <- J1$coef[1] + J1$coef[2] * x1[l]#mutatis mutandis u for chemical
#    e1[k,l] <- y1[l] - yhat1#mutatis mutandis z for magnetic
#  #2.Quadratic model and estiamtion of prediction error
#    J2 <- lm(y ~ x + I(x^2))
#    yhat2 <- J2$coef[1] + J2$coef[2] * x1[l] +J2$coef[3] *x1[l]^2
#    e2[k,l] <- y1[l] - yhat2
#  #3.Exponential model and estiamtion of prediction error
#    J3 <- lm(log(y) ~ x)
#    logyhat3 <- J3$coef[1] + J3$coef[2] * x1[l]
#    yhat3 <- exp(logyhat3)
#    e3[k,l] <- y1[l] - yhat3
#  #4.log-log model and estiamtion of prediction error
#    J4 <- lm(log(y) ~ log(x))
#    logyhat4 <- J4$coef[1] + J4$coef[2] * log(x1[l])
#    yhat4 <- exp(logyhat4)
#    e4[k,l] <- y1[l] - yhat4
#  }
#  }
#  c(mean(e1^2), mean(e2^2), mean(e3^2), mean(e4^2))#estimates for prediction error

## ----eval=FALSE----------------------------------------------------------
#  set.seed(1)
#  attach(chickwts)
#  x <- sort(as.vector(weight[feed == "soybean"]))
#  y <- sort(as.vector(weight[feed == "linseed"]))
#  detach(chickwts)
#  
#  
#  R <- 999 #number of replicates
#  
#  z <- c(x, y) #pooled sample
#  
#  W <-k<- numeric(R) #storage for replicates
#  m<-length(x)
#  n<-length(y)
#  N<-length(z)
#  u<-numeric(N)
#  
#  cvm<-function(x1,y1){#set the statistic function
#    Fn<-ecdf(x1)#Empirical cumulative distribution function of x
#    Gm<-ecdf(y1)#Empirical cumulative distribution function of y
#    z1<-c(x1,y1)
#  for(i in 1:N){
#    u[i]<-(Fn(z1[i])-Gm(z1[i]))^2
#  }
#  m*n/(m+n)^2*sum(u)
#  }
#  W0 <- cvm(x, y)#statistic
#  for (i in 1:R) {
#  #generate indices k for the first sample
#  k <- sample(1:N, size = m, replace = FALSE)
#  x1 <- z[k]
#  y1 <- z[-k] #complement of x1
#  W[i] <- cvm(x1, y1)
#  }
#  p <- mean(c(W0, W) >= W0)#calculate p value
#  p

## ----eval=FALSE----------------------------------------------------------
#  set.seed(1)
#  library(RANN) # implementing a fast algorithm
#                # for locating nearest neighbors
#                # (alternative R package: "yaImpute")
#  library(boot)
#  library(Ball)
#  library(energy)#for enenrgy test
#  library(devtools)#for Ball statistic
#  attach(chickwts) # chicken weights for various feed methods
#  x <- as.vector(weight[feed == "sunflower"])
#  y <- as.vector(weight[feed == "linseed"])
#  detach(chickwts)
#  z <- c(x, y)
#  m <- 1e2; k<-3; p<-2;n1 <- n2 <- 20#size for sample 1 and 2
#  R<-999; N = c(n1,n2)
#  Tn <- function(z, ix, sizes,k) {
#    n1 <- sizes[1]
#    n2 <- sizes[2]
#    n <- n1 + n2
#    if(is.vector(z)) z <- data.frame(z,0);
#    z <- z[ix, ];
#    NN <- nn2(data=z, k=k+1) # what's the first column?
#    block1 <- NN$nn.idx[1:n1,-1]
#    block2 <- NN$nn.idx[(n1+1):n,-1]
#    i1 <- sum(block1 < n1 + .5); i2 <- sum(block2 > n1+.5)
#    (i1 + i2) / (k * n)
#  }
#  
#  eqdist.nn <- function(z,sizes,k){
#    boot.obj <- boot(data=z,statistic=Tn,R=R,
#    sim = "permutation", sizes = sizes,k=k)
#    ts <- c(boot.obj$t0,boot.obj$t)
#    p.value <- mean(ts>=ts[1])
#    list(statistic=ts[1],p.value=p.value)
#  }
#  p.values <- matrix(NA,m,3)
#  
#  #Unequal variances and equal expectations
#  
#  for(i in 1:m){
#    x <- matrix(rnorm(n1*p,0,1),ncol=p);
#    y <- matrix(rnorm(n2*p,0,2),ncol=p);
#    z <- rbind(x,y)
#    p.values[i,1] <- eqdist.nn(z,N,k)$p.value#performance of NN method
#    p.values[i,2] <- eqdist.etest(z,sizes=N,R=R)$p.value#performance of energy method
#    p.values[i,3] <- bd.test(x=x,y=y,R=999,seed=i*12345)$p.value#performance of ball method
#  }
#  alpha <- 0.05                        #confidence level
#  pow <- colMeans(p.values<alpha)
#  print(pow)
#  
#  
#  
#  #Unequal variances and unequal expectations
#  for(i in 1:m){
#    x <- matrix(rnorm(n1*p),ncol=p);
#    y <-cbind(rnorm(n2,0,2),rnorm(n2,1,1))
#    z <- rbind(x,y)
#    p.values[i,1] <- eqdist.nn(z,N,k)$p.value#performance of NN method
#    p.values[i,2] <- eqdist.etest(z,sizes=N,R=R)$p.value#performance of energy method
#    p.values[i,3] <- bd.test(x=x,y=y,R=999,seed=i*12345)$p.value#performance of ball method
#  }
#  alpha <- 0.05                        #confidence level
#  pow <- colMeans(p.values<alpha)
#  print(pow)
#  
#  #Non-normal distributions: t distribution with 1 df (heavy-taileddistribution), bimodel distribution (mixture of two normal distributions)
#  m <- 1e2
#  n1 <- n2 <- 20
#  p<-2
#  for(i in 1:m){
#    x <- matrix(rt(n1*p,1),ncol=p);
#    y <-cbind(rnorm(n2,0,1),rnorm(n2,0,1))
#    z <- rbind(x,y)
#    p.values[i,1] <- eqdist.nn(z,N,k)$p.value#performance of NN method
#    p.values[i,2] <- eqdist.etest(z,sizes=N,R=R)$p.value#performance of energy method
#    p.values[i,3] <- bd.test(x=x,y=y,R=999,seed=i*12345)$p.value#performance of ball method
#  }
#  alpha <- 0.05                        #confidence level
#  pow <- colMeans(p.values<alpha)
#  print(pow)
#  
#  
#  
#  #Unbalanced samples (say, 1 case versus 10 controls)
#  n1<-10;n2<-100;N=c(n1,n2)
#  for(i in 1:m){
#    x <- matrix(rt(n1*p,1),ncol=p);
#    y <- matrix(rnorm(n2*p,1,2),ncol=p);
#    z <- rbind(x,y)
#    p.values[i,1] <- eqdist.nn(z,N,k)$p.value#performance of NN method
#    p.values[i,2] <- eqdist.etest(z,sizes=N,R=R)$p.value#performance of energy method
#    p.values[i,3] <- bd.test(x=x,y=y,R=999,seed=i*12345)$p.value#performance of ball method
#  }
#  alpha <- 0.05                        #confidence level
#  pow <- colMeans(p.values<alpha)
#  print(pow)

## ----eval=FALSE----------------------------------------------------------
#  m <- 10000
#  x <- numeric(m)
#  x[1] <- rnorm(1)
#  k <- 0
#  b<-1000
#  #generates random deviates
#  u <- runif(m)
#  #Use the Metropolis-Hastings sampler to generate random variables from a standard Cauchy distribution.
#  for (i in 2:m)
#  {
#      xt <- x[i-1]
#      y <- rnorm(1, mean = xt)
#      #Density Cauchy distribution
#      num <- dcauchy(y)*dnorm(xt, mean = y)
#      den <- dcauchy(xt)*dnorm(y, mean = xt)
#  
#      if(u[i] <= num/den)
#      {
#          x[i] <- y
#      }
#      else
#      {
#          x[i] <- xt
#         k <- k+1#y is rejected
#      }
#  }
#  
#  plot(x,type = "l",main="",ylab="x")
#  y <- x[b:m]
#  b <- x[1000:m]
#  sequence <- seq(0,1,0.01)
#  standardCauchy <- qcauchy(sequence)
#  standardCauchy <- standardCauchy[(standardCauchy> -Inf) & (standardCauchy< Inf)]
#  hist(b, freq = FALSE,main = "")
#  lines(standardCauchy, dcauchy(standardCauchy), lty = 2)
#  qqplot(standardCauchy, quantile(x,sequence), main="",xlab="Cauchy Quantiles", ylab="Sample Quantiles")#compare the deciles of the generated observations with the deciles of the standard Cauchy distribution
#  

## ----eval=FALSE----------------------------------------------------------
#  set.seed(1)
#  t <- c(125,18,20,34) #actual value of theta
#  w <- 0.25 #width of the uniform support set
#  m <- 5000 #length of the chain
#  burn <- 1000 #burn-in time
#  animals <- 197
#  x <- numeric(m) #the chain
#  
#  
#  
#  prob <- function(y, t) {
#  # computes (without the constant) the target density
#  if (y < 0 || y >1)
#  return (0)
#  return((1/2+y/4)^t[1] *
#  ((1-y)/4)^t[2] * ((1-y)/4)^t[3] *
#  (y/4)^t[4])
#  }
#  u <- runif(m) #for accept/reject step
#  v <- runif(m, -w, w) #proposal distribution
#  x[1] <- .25
#  for (i in 2:m) {
#  y <- x[i-1] + v[i]
#  if (u[i] <= prob(y, t) / prob(x[i-1], t))
#  x[i] <- y else
#  x[i] <- x[i-1]
#  }
#  xt <- x[(burn+1):m]
#  theta.hat<-mean(xt)
#  print(theta.hat)
#  that <- sum(t) * c((2+theta.hat)/4, (1-theta.hat)/4, (1-theta.hat)/4, theta.hat/4)
#  print(round(that))

## ----eval=FALSE----------------------------------------------------------
#  set.seed(1)
#  Gelman.Rubin <- function(psi) {
#    # psi[i,j] is the statistic psi(X[i,1:j])
#    # for chain in i-th row of X
#    psi <- as.matrix(psi)
#    n <- ncol(psi)
#    k <- nrow(psi)
#  
#    psi.means <- rowMeans(psi)     #row means
#    B <- n * var(psi.means)        #between variance est.
#    psi.w <- apply(psi, 1, "var")  #within variances
#    W <- mean(psi.w)               #within est.
#    v.hat <- W*(n-1)/n + (B/n)     #upper variance est.
#    r.hat <- v.hat / W             #G-R statistic
#    return(r.hat)
#  }
#  
#  t <- c(125,18,20,34) #actual value
#  b<- 1000 #burn-in time
#  prob<- function(y, t) {
#    # computes (without the constant) the target density
#    if (y < 0 || y >1)
#      return (0)
#    else
#      return((1/2+y/4)^t[1] *((1-y)/4)^t[2] * ((1-y)/4)^t[3] *(y/4)^t[4])
#  }
#  
#  #generate chains
#  chain<-function(w,N){
#    x<-rep(0,N)
#    u <- runif(N) #for accept/reject step
#    v <- runif(N, -w, w) #proposal distribution
#    x[1] <-w
#    for (i in 2:N) {
#      y <- x[i-1] + v[i]
#      if (u[i] <= prob(y, t) / prob(x[i-1], t))
#        x[i] <- y
#      else
#          x[i] <- x[i-1]
#    }
#  return(x)
#  }
#  w <- c(0.05,0.2,0.4,0.8)#set the paremeter of proposal distribution
#  n<- 15000 #length of the chain
#  k<-4 #number of chains to generate
#  
#  #generate the chains
#  X <- matrix(0, nrow=k, ncol=n)
#  for (i in 1:k)
#    X[i, ] <- chain(w[i],n)
#  
#  #compute diagnostic statistics
#  psi <- t(apply(X, 1, cumsum))
#  for (i in 1:nrow(psi))
#    psi[i,] <- psi[i,] / (1:ncol(psi))
#    print(Gelman.Rubin(psi))
#  #plot psi for the four chains
#  par(mfrow=c(2,2))
#  for (i in 1:k)
#    plot(psi[i, (b+1):n], type="l",
#         xlab=i, ylab=bquote(psi))
#  par(mfrow=c(1,1)) #restore default
#  
#  
#  #plot the sequence of R-hat statistics
#  rhat <- rep(0, n)
#  for (j in (b+1):n)
#    rhat[j] <- Gelman.Rubin(psi[,1:j])
#  plot(rhat[(b+1):n], type="l", xlab="", ylab="R",ylim=c(1,1.3))
#  abline(h=1.1, lty=2)
#  abline(h=1.2, lty=2)

## ----eval=FALSE----------------------------------------------------------
#  findIntersection = function (k) {
#    s.k.minus.one = function (a) {#set function S_{k-1}(a)
#      1-pt(sqrt(a^2 * (k - 1) / (k - a^2)), df = k-1)
#    }
#    s.k = function (a) {#set function S_{k}(a)
#      1-pt(sqrt(a^2 * k / (k + 1 - a^2)), df = k)
#    }
#  
#    f = function (a) {#set function S_{k}(a)-S_{k-1}(a)
#      s.k(a) - s.k.minus.one(a)
#    }#Find the intersection points A(k) in (0,sqrt(k)) of the curves
#    eps = .Machine$double.eps^0.5
#    return(uniroot(f, interval = c(eps, sqrt(k)-eps))$root)
#  }
#  k<-c(4:25, 100, 500, 1000)
#  rs = sapply(k, function (k) {findIntersection(k)})
#  s<-cbind(rs,k)
#  s

## ----eval=FALSE----------------------------------------------------------
#  dcauchy <- function(t, eta, theta) {#compute the desity function of cauchy distribution
#    if (theta<=0) return (0)#let theta > 0
#    else
#      return (1/(theta*pi*(1+((t-eta)/theta)^2)))
#    }
#  cdf.cauchy <- function(x, eta, theta){#compute the cdf of cauchy distribution
#  res <- integrate(dcauchy, lower=-Inf, upper=x,
#                     rel.tol=.Machine$double.eps^0.25,
#                     eta=eta, theta=theta)$value
#  
#  return(res)
#  }
#  pcauchy(0,0,1)#Compare your results to the results from the R function pcauchy.
#  cdf.cauchy (0,0,1)
#  pcauchy(1,0,1)
#  cdf.cauchy (1,0,1)
#  pcauchy(-0.1,0,1)
#  cdf.cauchy (-0.1,0,1)

## ----eval=FALSE----------------------------------------------------------
#  #set the likelihood function
#  lnL <- function(p, q, nA =28, nB = 24, nAB = 70, nO = 41) {
#    r = 1.0 - p - q
#    nA * log(p^2 + 2*p*r) + nB * log(p^2 + 2 * q * r) +
#      nAB * log(2 * p * q) + 2 * nO * log(r)
#  }
#  
#  EM <- function (p,q,nA =28, nB = 24, nAB = 70, nO = 41, debug = FALSE) {
#  
#    # Evaluate the likelihood using initial estimates
#    llk <- lnL(p, q, nA, nB, nAB, nO)
#  
#    # Count the number of iterations so far
#    iter <- 1
#  
#    # Loop until convergence ...
#    while (TRUE)
#    {
#      # Estimate the frequency for allele O
#      r= 1.0 - p - q
#  
#      # First we carry out the E-step
#  
#      # The counts for genotypes O/O and A/B are effectively observed
#      # Estimate the counts for the other genotypes
#      nAA <- nA * p / (p + 2*r)
#      nAO <- nA - nAA
#      nBB <- nB * q / (q + 2*r)
#      nBO <- nB - nBB
#  
#      # Print debugging information
#      if (debug)
#      {
#        cat("Round #", iter, "lnLikelihood = ", llk, "\n")
#        cat("    Allele frequencies: p = ", p, ", q = ", q, ",r = ", r, "\n")
#        cat("    Genotype counts:    nAA = ", nAA, ", nAO = ", nAO, ", nBB = ", nBB,
#            ", nBO = ", nBO, "\n")
#      }
#  
#      # Then the M-step
#      p <- (2 * nAA + nAO + nAB) / (2 * (nA + nB + nO + nAB))
#      q <- (2 * nBB + nBO + nAB) / (2 * (nA + nB + nO + nAB))
#  
#  
#      # Then check for convergence ...
#      llk1 <- lnL(p, q, nA, nB, nAB, nO)
#      if (abs(llk1 - llk) < (abs(llk) + abs(llk1)) * 1e-6) break
#  
#      # Otherwise keep going
#      llk <- llk1
#      iter <- iter + 1
#    }
#  
#   list(p = p, q = q)
#  }
#  EM(0.3,0.2,nA =28, nB = 24, nAB = 70, nO = 41, debug = TRUE)

## ----eval=FALSE----------------------------------------------------------
#  formulas <- list(
#    mpg ~ disp,
#    mpg ~ I(1 / disp),
#    mpg ~ disp + wt,
#    mpg ~ I(1 / disp) + wt
#  )
#  #With lapply()
#  model_mtcars1<-lapply(formulas,lm,data=mtcars)
#  model_mtcars1
#  #With a for loop
#  model_mtcars2<- vector("list", length(formulas))
#          for (i in seq_along(formulas)) {
#            model_mtcars2[[i]] <- lm( formulas[[i]],mtcars)
#          }
#  model_mtcars2

## ----eval=FALSE----------------------------------------------------------
#  set.seed(1)
#  bootstraps <- lapply(1:10, function(i) {
#  rows <- sample(1:nrow(mtcars), rep = TRUE)
#  mtcars[rows, ]
#  })
#  #With lapply()
#  model_bootstrap1<-lapply(bootstraps,lm,formula=mpg ~ disp)
#  model_bootstrap1
#  #With a for loop
#  model_bootstrap2 <- vector("list", length(bootstraps))
#          for (i in seq_along(bootstraps)) {
#            model_bootstrap2[[i]] <- lm( mpg ~ disp,bootstraps[[i]])
#          }
#  model_bootstrap2

## ----eval=FALSE----------------------------------------------------------
#  #model in question 1
#  formulas <- list(
#    mpg ~ disp,
#    mpg ~ I(1 / disp),
#    mpg ~ disp + wt,
#    mpg ~ I(1 / disp) + wt
#  )
#  #With lapply()
#  model_mtcars1<-lapply(formulas,lm,data=mtcars)
#  
#  #With a for loop
#  model_mtcars2<- vector("list", length(formulas))
#          for (i in seq_along(formulas)) {
#            model_mtcars2[[i]] <- lm( formulas[[i]],mtcars)
#          }
#  
#  
#  #model in question 2
#  bootstraps <- lapply(1:10, function(i) {
#  rows <- sample(1:nrow(mtcars), rep = TRUE)
#  mtcars[rows, ]
#  })
#  #With lapply()
#  model_bootstrap1<-lapply(bootstraps,lm,formula=mpg ~ disp)
#  
#  #With a for loop
#  model_bootstrap2 <- vector("list", length(bootstraps))
#          for (i in seq_along(bootstraps)) {
#            model_bootstrap2[[i]] <- lm( mpg ~ disp,bootstraps[[i]])
#          }
#  
#  
#  
#  #extract R2
#  rsq <- function(mod) summary(mod)$r.squared
#  #For model in question 1
#  model1_rsq1<-sapply(model_mtcars1, rsq)
#  model1_rsq2<-sapply(model_mtcars2, rsq)
#  #For model in question 2
#  model2_rsq1<-sapply(model_bootstrap1, rsq)
#  model2_rsq2<-sapply(model_bootstrap1, rsq)
#  rbind(model1_rsq1,model1_rsq2)
#  rbind(model2_rsq1,model2_rsq2)

## ----eval=FALSE----------------------------------------------------------
#  set.seed(1)
#  # Create some random data
#  trials <- replicate(
#  100,
#  t.test(rpois(10, 10), rpois(7, 10)),
#  simplify = FALSE
#  )
#  #Use sapply() and an anonymous function to extract the p-value from every trial
#  p1<-sapply(trials, function(test) test$p.value)
#  #Get rid of the anonymous function by using [[ directly.
#  p2<-sapply(trials, '[[', i = "p.value")
#  cbind(p1,p2)

## ----eval=FALSE----------------------------------------------------------
#  #construct function to compute  the expected (theoretical) count
#  expected <- function(colsum, rowsum, total) {
#    (colsum / total) * (rowsum / total) * total
#  }
#  #construct function to compute the value of the test-statistic
#  chi_stat <- function(observed, expected) {
#    ((observed - expected) ^ 2) / expected
#  }
#  
#  
#  chisq_test2 <- function(x, y) {
#    total <- sum(x) + sum(y)
#    rowsum_x <- sum(x)
#    rowsum_y <- sum(y)
#    chistat <- 0
#  # computes the chi-square test statistic which is apparently different from chisq.test function
#    for (i in seq_along(x)) {
#      colsum <- x[i] + y[i]
#      expected_x <- expected(colsum, rowsum_x, total)
#      expected_y <- expected(colsum, rowsum_y, total)
#      chistat <- chistat + chi_stat(x[i], expected_x)
#      chistat <- chistat + chi_stat(y[i], expected_y)
#    }
#    chistat
#  }
#  o <- as.table(rbind(c(56, 67, 57,34), c(135, 125, 45,65)))
#  #compare the results of chi-square test
#  print(chisq_test2(c(56, 67, 57,34), c(135, 125, 45,65)))
#  print(chisq.test(as.table(rbind(c(56, 67, 57,34), c(135, 125, 45,65)))))
#  #compare the computing time
#  print(microbenchmark::microbenchmark(
#    chisq_test2(c(56, 67, 57,34), c(135, 125, 45,65)),
#    chisq.test(as.table(rbind(c(56, 67, 57,34), c(135, 125, 45,65))))
#  ))
#  
#  

## ----eval=FALSE----------------------------------------------------------
#  #With table
#  subject1<-c(60,70,60,80)
#  subject2<-c(60,60,60,90)
#  data<-data.frame(subject1,subject2)
#  result_table<-table(data)
#  result_table
#  
#  #with loop
#  x<-c(60,70,80,90)
#  f<-function(x){
#  i<-numeric(4)
#  j<-numeric(4)
#  q<-numeric(4)
#  for (i in 1:4){
#    for(j in 1:4){
#      for(q in 1:4){
#  result_2<-ifelse(subject1[i]==x[j]&subject2[i]==x[q],"1","0")
#      }}}}
#  
#  print(microbenchmark::microbenchmark(
#    table(data),
#    f(x)))
#  

