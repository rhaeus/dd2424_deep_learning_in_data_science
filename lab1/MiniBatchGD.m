function [Wstar, bstar] = MiniBatchGD(X, Y, GDparams, W, b, lambda)

P = EvaluateClassifier(X, W, b);
[grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda);

Wstar = W - GDparams.eta * grad_W;
bstar = b - (GDparams.eta * grad_b);

end

