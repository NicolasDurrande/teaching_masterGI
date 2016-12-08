setwd('/home/durrande/Desktop/DemandForecasting/lecture2/figures/R')
library(tseries)

######################################
######################################
# exponential smoothing no trend in data
unemploiment <- read.ts(file="unemp.txt", start=1961, frequency=12,skip=0)

xlisse <- HoltWinters(unemploiment, beta=FALSE, gamma=FALSE) 
p <- predict(xlisse, n.ahead=60)

# prediction
pdf("SESpred.pdf",width=9, height=5)
plot(unemploiment, type="b",xlim = c(1980,1989),ylim=c(500,1000),ylab='unemployment')
lines(xlisse$fitted[,1], col="red")
lines(p, col="red")
dev.off()

######################################
######################################
# exponential smoothing with trend in data
unemploiment <- read.ts(file="unemp.txt", start=1961, frequency=12,skip=0) + 5*1:288

xlisse <- HoltWinters(unemploiment, beta=FALSE, gamma=FALSE) 
p <- predict(xlisse, n.ahead=60)

# prediction
pdf("SESpredtrend.pdf",width=9, height=5)
plot(unemploiment,xlim = c(1960,1990), type="b",ylab='unemployment')
lines(xlisse$fitted[,1], col="red")
lines(p, col="red")
dev.off()

######################################
######################################
# Holt
unemploiment <- read.ts(file="unemp.txt", start=1961, frequency=12,skip=0) + 5*1:288

xlisse <- HoltWinters(unemploiment,gamma=FALSE) 
p <- predict(xlisse, n.ahead=60)

# prediction
pdf("Holtpredtrend.pdf",width=9, height=5)
plot(unemploiment,xlim = c(1960,1990),ylim = c(400,2500), type="b",ylab='unemployment')
lines(xlisse$fitted[,1], col="red")
lines(p, col="red")
dev.off()



######################################
######################################
# Data with seasonality
data(AirPassengers)
airlines <- ts(AirPassengers, start=1949, frequency=12)
pdf("airlines.pdf",width=9, height=5) 
plot(airlines, type="o")
dev.off()


#############
# additive seasonal Holt-Winters
xlisse <- HoltWinters(airlines,seasonal = "additive") 
p <- predict(xlisse, n.ahead=36)

# prediction
pdf("HoltWintersseasonadd.pdf",width=9, height=5)
plot(airlines, xlim = c(1949,1963),ylim=range(airlines,p),type="b",ylab='airlines')
lines(xlisse$fitted[,1], col="red")
lines(p, col="red")
dev.off()

#############
# multiplicative seasonal Holt-Winters
xlisse <- HoltWinters(airlines,seasonal = "mult") 
p <- predict(xlisse, n.ahead=36)

# prediction
pdf("HoltWintersseasonmult.pdf",width=9, height=5)
plot(airlines, xlim = c(1949,1963),ylim=range(airlines,p),type="b",ylab='airlines')
lines(xlisse$fitted[,1], col="red")
lines(p, col="red")
dev.off()



######################################
######################################
# autres commandes utiles

# decomposer les modeles
data(AirPassengers)
airlines <- ts(AirPassengers, start=1949, frequency=12)
xlisse <- HoltWinters(airlines,seasonal = "mult") 
pdf("dec.pdf",width=12, height=7) 
plot(xlisse$fitted)
dev.off()

## recuperer les composantes du modele
fitted <- data.frame(xlisse$fitted)
pred <- (fitted$trend + fitted$level)*fitted$season
