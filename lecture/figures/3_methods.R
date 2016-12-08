
T = 1:20 + 1993
Y = 0.5*t + rnorm(length(T))

b1 <- function(t){
  return(rep(1,length(t)))
}

b2 <- function(t){
  return(t)
}

B <- c(b1,b2)
linear_regression <- function(t,T,Y,B){
  n <- length(T)
  np <- length(t)
  m <- length(B)
  X <- matrix(0,n,m)
  for(i in 1:n){
    for(j in 1:m){
      X[i,j] = B[[j]](T[i])
    }
  }
  Xp <- matrix(0,np,m)
  for(i in 1:np){
    for(j in 1:m){
      Xp[i,j] = B[[j]](t[i])
    }
  }
  return(Xp%*% solve(t(X)%*%X) %*% t(X) %*% Y)
}

Tp = 1:27 + 1993
Yp <- linear_regression(Tp,T,Y,B) 

## data
plot(T,Y,xlab="Year",ylab="Sales in thousands",xlim=c(1992,2020),ylim=c(0,15))

## basis functions
plot(Tp,b1(Tp),xlab="Year",ylab="basis functions",xlim=c(1992,2020),ylim=c(0,15),type="l")
lines(Tp,(b2(Tp)-1995)/2,col='red')
legend("topleft", legend = c("b1", "b2"),lty=1,col=c('black','red'))
       
## prediction
plot(T,Y,xlab="Year",ylab="Sales in thousands",xlim=c(1992,2020),ylim=c(0,15))
lines(Tp,Yp,col='red')
#points(Tp,Yp,col='red')

# residuals
res <- Y - linear_regression(T,T,Y,B) 
plot(T,res,xlab="Year",ylab="Residuals",xlim=c(1992,2014))
