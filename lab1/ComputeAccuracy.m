function acc = ComputeAccuracy(X, y, W, b)
% • each column of X corresponds to an image and X has size d×n.
% • y is the vector of ground truth labels of length n. 1xn
% • acc is a scalar value containing the accuracy.
P = EvaluateClassifier(X, W, b); % Kxn

[K,n] = size(P); % 10x10000
[~, p] = max(P); % p is index of max, 1xn

acc = sum(p==y) / n;
end

