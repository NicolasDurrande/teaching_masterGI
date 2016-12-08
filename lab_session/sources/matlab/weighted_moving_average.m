function [Tpred,Ypred] = weighted_moving_average(T,Y,w)
  l = length(w);
  n = length(T);
  Ypred = zeros(n-l+1,1);
  for i = 1:(n-l+1)
      Ypred(i) = sum(w*Y(i:l+i-1))/sum(w);
  end
  Tpred = vertcat(T(l+1:n),2*T(n)-T(n-1));
end
