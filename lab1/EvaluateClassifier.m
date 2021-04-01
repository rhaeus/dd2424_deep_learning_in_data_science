function [P] = EvaluateClassifier(X, W, b)
% • each column of X corresponds to an image and it has size d×n.
% • W and b are the parameters of the network.
% • each column of P contains the probability for each label

s = W * X + b;
P = exp(s) ./ sum(exp(s));
end

