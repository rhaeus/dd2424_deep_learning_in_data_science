function J = ComputeCost(X, Y, W, b, lambda)
% • each column of X corresponds to an image and X has size d×n.
% • each column of Y (K×n) is the one-hot ground truth label for the corre-
% sponding column of X or Y is the (1×n) vector of ground truth labels.
% • J is a scalar corresponding to the sum of the loss of the network’s
% predictions for the images in X relative to the ground truth labels and
% the regularization term on W.
[d,n] = size(X);

P = EvaluateClassifier(X, W, b); % Kxn

% S = sum(X) is the sum of the elements of the vector X. If X is a matrix,
% S is a row vector with the sum over each column. 
l = -log(sum(Y .* P)); % Y = Kxn, l = 1xn

J = sum(l)/n + lambda * sum(sum(W .* W));

end
