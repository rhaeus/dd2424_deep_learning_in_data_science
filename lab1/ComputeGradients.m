function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)

% P = EvaluateClassifier(X, W, b); % Kxn
[K,n] = size(P);
% J = ComputeCost(X, Y, W, b, lambda);

G_batch = -(Y - P); %Kxn
% Kxn * nxd = Kxd
grad_W = (G_batch * X') / n + 2 * lambda * W;
grad_b = G_batch / n;

