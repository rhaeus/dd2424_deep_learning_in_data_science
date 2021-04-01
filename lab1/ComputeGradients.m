function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)

[K,n] = size(P);

G_batch = -(Y - P); %Kxn
% Kxn * nxd = Kxd
grad_W = (G_batch * X') / n + 2 * lambda * W;

One = ones(n,1);
grad_b = (G_batch * One) / n;

