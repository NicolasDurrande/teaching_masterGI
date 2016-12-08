load('data1.mat')
plot(Tpred,Ypred)

scatter(T,Y)
hold on

%% Moving average
l = 2;
[Tpred,Ypred] = moving_average(T,Y,l);

plot(Tpred,Ypred,'*');

%% Weighted Moving average
w = [0.1,0.5,1,2];
[Tpred,Ypred] = weighted_moving_average(T,Y,w);

plot(Tpred,Ypred,'o');

%% Exponential Smoothing
alpha = 0.7;
[Tpred,Ypred] = exponential_smoothing(T,Y,alpha);
plot(Tpred,Ypred,'x');


%% linear regression
n = length(T);
Tpred = vertcat(T,2*T(n)-T(n-1));
[Tpred,Ypred] = linear_regression(Tpred,T,Y);

plot(Tpred,Ypred,'.');


