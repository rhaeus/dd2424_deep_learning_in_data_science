function [P, Xs] = EvaluateClassifier(X, Ws, bs)
[k, ~] = size(Ws);
Xs = cell(k, 1);
Xs{1} = X;

for i=1:k-1
    s1 = Ws{i} * Xs{i} + bs{i}; % mxn
    h = max(0, s1); % mxn
    Xs{i+1} = h;
end

s = Ws{k} * Xs{k} + bs{k}; %Kxn
P = exp(s) ./ sum(exp(s)); %Kxn
end