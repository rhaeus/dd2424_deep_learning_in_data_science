addpath datasets/cifar-10-batches-mat/;

% load data
[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
mean_X = mean(X_train, 2); %dx1
std_X = std(X_train, 0, 2); %dx1

X_train = X_train - repmat(mean_X, [1, size(X_train,2)]);
X_train = X_train ./ repmat(std_X, [1, size(X_train,2)]);

[X_valid, Y_valid, y_valid] = LoadBatch('data_batch_2.mat');
X_valid = X_valid - repmat(mean_X, [1, size(X_valid,2)]);
X_valid = X_valid ./ repmat(std_X, [1, size(X_valid,2)]);

[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');
X_test = X_test - repmat(mean_X, [1, size(X_test,2)]);
X_test = X_test ./ repmat(std_X, [1, size(X_test,2)]);

% init model
[d, n] = size(X_train);
K = 10;
rng(400);
W = 0.01*randn(K,d);
b = 0.01*randn(K,1);

% P = EvaluateClassifier(X_train(1:20, 1), W(:, 1:20), b);
% J = ComputeCost(X_train, Y_train, W, b, 0.01)
% acc = ComputeAccuracy(X_train, y_train, W, b)

lambda = 0;
% P = EvaluateClassifier(X_train(1:20, 1), W(:, 1:20), b);
% disp('computing gradsnum')
% [ngrad_b, ngrad_W] = ComputeGradsNumSlow(X_train(1:20, 1), Y_train(:, 1), W(:, 1:20), b, lambda, 1e-6);
% disp('computing grads')
% [grad_W, grad_b] = ComputeGradients(X_train(1:20, 1), Y_train(:, 1), P, W(:, 1:20), lambda);

P = EvaluateClassifier(X_train(:, 1), W, b);
disp('computing gradsnum')
[ngrad_b, ngrad_W] = ComputeGradsNumSlow(X_train(:, 1), Y_train(:, 1), W, b, lambda, 1e-6);
disp('computing grads')
[grad_W, grad_b] = ComputeGradients(X_train(:, 1), Y_train(:, 1), P, W, lambda);

eps = 1e-6;
diff_W = abs(ngrad_W - grad_W)./max(eps, abs(grad_W) + abs(ngrad_W));
diff_b = abs(ngrad_b - grad_b)./max(eps, abs(grad_b) + abs(ngrad_b));


if all(diff_W < 1e-6)
    disp('W ok')
else 
    disp('W not ok, max diff: ')
    max(max(diff_W))
end

if all(diff_b < 1e-6)
    disp('b ok')
else 
    disp('b not ok, max diff: ')
    max(max(diff_b))
end
