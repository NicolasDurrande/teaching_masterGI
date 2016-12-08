function [Tpred,Ypred] = linear_regression(Tpred,T,Y)
  T = double(T);
  Tpred = double(Tpred);
  X = [ones(length(T),1) T];
  Xp = [ones(length(Tpred),1) Tpred];
  beta = X\Y;
  Ypred = Xp * beta ;
end
