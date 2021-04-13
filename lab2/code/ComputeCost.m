function J = ComputeCost(X, Y, Ws, bs, lambda)
[d,n] = size(X);
P = EvaluateClassifier(X, Ws, bs); % Kxn

reg = 0;
[k, ~] = size(Ws);
for i=1:k
    reg = reg + sum(sum(Ws{i} .* Ws{i}));
end


l1 = -log(sum(Y .* P)); % Y = Kxn, l = 1xn
l1 = sum(l1)/n;

% J = sum(l)/n + lambda * reg;
bla = mean(-mean(sum(Y .* log(P)), 1));
if bla ~= l1
    disp('not the same')
end
% size(bla)
J = bla + lambda * reg;
end