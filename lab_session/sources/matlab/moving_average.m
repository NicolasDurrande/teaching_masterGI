function [Tpred,Ypred] = moving_average(T,Y,l)
  w = ones(1,l);
  [Tpred,Ypred] = weighted_moving_average(T,Y,w);
end
