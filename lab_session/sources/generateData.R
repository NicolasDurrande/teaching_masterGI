setwd('/home/durrande/Desktop/TP_DemandForecast/TP1/R')
library(MASS)

###########################################
# Data1 trend + exp
kern <- function(x,y){
	d <- matrix(x,length(x),length(y)) - matrix(y,length(x),length(y),byrow=T)
	return(exp(-abs(d/5)))
}

T <- 1950:2014
K <- kern(T,T)
Z <- mvrnorm(1,0*T,K)

Y = 10*sin(T/25-4.8)+ Z + 500

plot(T,Y)

save(T,Y,file="data1.RData")

###########################################
# Data2 exp + missing + outlier
kern <- function(x,y){
	d <- matrix(x,length(x),length(y)) - matrix(y,length(x),length(y),byrow=TRUE)
	return(exp(-abs(d/25)))
}

T <- 0:100
K <- kern(T,T)
Z <- mvrnorm(1,0*T,K)

Y <- Z 
Y[43] <- NA
Y[22] <- 289

plot(T,Z)

save(T,Y,file="data2.RData")

###########################################
# Data3 cst + noise

T <- 0:1000/1000
Z <- rnorm(length(T),.5,.5)

Y <- pmax(Z,0)

plot(T,Y)

save(T,Y,file="data3.RData")

###########################################
# Data4 trend + seasonality + noise

T <- 1:(12*5)/12
Y <- T/10 + sin(T*2*pi) + rnorm(length(T),0,.15)

plot(T,Y)

save(T,Y,file="data4.RData")

