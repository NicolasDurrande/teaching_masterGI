function [Tpred,Ypred] = exponential_smoothing(T,Y,alpha)
  n = length(T);
  Ypred = Y(1) + zeros(n+1,1);
  for i = 2:n+1
      Ypred(i) = alpha * Y(i-1) + (1-alpha) * Ypred(i-1) ;
  end
  Tpred = vertcat(T,2*T(n)-T(n-1));
end
