
#############################################
## Moving average
moving_average <- function(T,Y,l){
  w <- rep(1,l) 
  return(weighted_moving_average(T,Y,w))
}

#############################################
## Weighted Moving average
weighted_moving_average <- function(T,Y,w){
  l <- length(w)
  n <- length(T)
  Ypred = rep(0,n-l+1)
  for(i in 1:(n-l+1)){
      Ypred[i] <- sum(w*Y[1:l+i-1])/sum(w)
  } 
  Tpred <- c(T[(l+1):n],2*T[n]-T[n-1])
  return(list(T=Tpred,Y=Ypred))
}

#############################################
## Exponential smoothing
exponential_smoothing <- function(T,Y,alpha){
  n <- length(T)
  dt <- T[2] - T[1]
  Ypred <- rep(Y[1],n+1)
  for(i in 1:n+1){
    Ypred[i] <- alpha* Y[i-1] + (1-alpha) * Ypred[i-1]
  } 
  Tpred <- c(T,2*T[n]-T[n-1])
  return(list(T=Tpred,Y=Ypred))
}

#############################################
## regression
b0 <- function(t){
  return(0*t + 1)
}

b1 <- function(t){
  return(t)
}

b2 <- function(t){
  return(t^2)
}

linear_regression <- function(Tpred,T,Y,B){
  n <- length(T)
  np <- length(Tpred)
  m <- length(B)
  X <- matrix(0,n,m)
  for(i in 1:n){
    for(j in 1:m){
      X[i,j] <- B[[j]](T[i])
    }
  }
  Xp <- matrix(0,np,m)
  for(i in 1:np){
    for(j in 1:m){
      Xp[i,j] <- B[[j]](Tpred[i])
    }
  }
  Ypred <-  Xp%*% solve(t(X)%*%X) %*% t(X) %*% Y
  return(list(T=Tpred,Y=Ypred))
}

