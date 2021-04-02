addpath datasets/cifar-10-batches-mat/;

% load data
disp('load data')
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
disp('init model')
[d, n] = size(X_train);
K = 10;
rng(400);
W = 0.01*randn(K,d);
b = 0.01*randn(K,1);

% % Experiment 1
% lambda = 0;
% n_batch = 100;
% eta = 0.1;
% n_epochs = 40;
% 
% % Experiment 2
% lambda = 0;
% n_batch = 100;
% eta = 0.001;
% n_epochs = 40;
% 
% % Experiment 3
% lambda = 0.1;
% n_batch = 100;
% eta = 0.001;
% n_epochs = 40;
% 
% Experiment 4
lambda = 1;
n_batch = 100;
eta = 0.001;
n_epochs = 40;

gd_params = GDparams(n_batch, eta, n_epochs);

fprintf('lambda=%0.5f\nn_batch=%d\neta=%0.5f\nn_epochs=%d\n', lambda, n_batch, eta, n_epochs);

loss_training = zeros(gd_params.n_epochs, 1);
loss_validation = zeros(gd_params.n_epochs, 1);
accuracy = zeros(gd_params.n_epochs, 1);

% train
disp('begin training')
for i=1:gd_params.n_epochs
    fprintf('epoch %d of %d.\n',i,gd_params.n_epochs);
    for j=randperm(n/gd_params.n_batch)
        j_start = (j-1)*gd_params.n_batch + 1;
        j_end = j*gd_params.n_batch;
        inds = j_start:j_end;
        Xbatch = X_train(:, j_start:j_end);
        Ybatch = Y_train(:, j_start:j_end);
        
        [W, b] = MiniBatchGD(Xbatch, Ybatch, gd_params, W, b, lambda);
        
    end
    
    loss_training(i) = ComputeCost(X_train, Y_train, W, b, lambda);
    loss_validation(i) = ComputeCost(X_valid, Y_valid, W, b, lambda);
    accuracy(i) = ComputeAccuracy(X_test, y_test, W, b);
end
disp('training done')

fprintf('final training loss %0.3f\n', loss_training(gd_params.n_epochs));
fprintf('final validation loss %0.3f\n', loss_validation(gd_params.n_epochs));
fprintf('final accuracy %0.4f\n', accuracy(gd_params.n_epochs));

% Plots the weights
figure(1);
for i=1:10
    im = reshape(W(i, :), 32, 32, 3);
    s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
    s_im{i} = permute(s_im{i}, [2, 1, 3]);
end
montage(s_im, 'Size', [2,5]);

% evolution of the loss as diagram
figure(2);
x = 1:gd_params.n_epochs;
plot(x, loss_training, x, loss_validation);
title('Loss')
legend('Training', 'Validation')

% evolution of the accuracy as diagram
figure(3);
x = 1:gd_params.n_epochs;
plot(x, accuracy);
title('Accuracy')
legend('Test')


% functions
function acc = ComputeAccuracy(X, y, W, b)
P = EvaluateClassifier(X, W, b); % Kxn
[K,n] = size(P); % 10x10000
[~, p] = max(P); % p is index of max, 1xn
acc = sum(p==y) / n;
end

function J = ComputeCost(X, Y, W, b, lambda)
[d,n] = size(X);
P = EvaluateClassifier(X, W, b); % Kxn
l = -log(sum(Y .* P)); % Y = Kxn, l = 1xn
J = sum(l)/n + lambda * sum(sum(W .* W));
end

function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
[K,n] = size(P);
G_batch = -(Y - P); %Kxn
% Kxn * nxd = Kxd
grad_W = (G_batch * X') / n + 2 * lambda * W;
One = ones(n,1);
grad_b = (G_batch * One) / n;
end

function [P] = EvaluateClassifier(X, W, b)
s = W * X + b; % Kxn
P = exp(s) ./ sum(exp(s));
end

function [X, Y, y] = LoadBatch(filename)
A = load(filename);
[n, d] = size(A.data); % nxd
X = double(A.data') / 255; % dxn
y = A.labels'; % 1xn

% CIFAR-10 encodes the labels as integers between 0-9 but
% Matlab indexes matrices and vectors starting at 1. Therefore it may be
% easier to encode the labels between 1-10.
y = y + uint8(ones(1, n)); % 1xn

% Create image label matrix
K = 10;
Y = zeros(K, n); % Kxn
for i = 1:n
    Y(y(i),i) = 1;
end
end

function [Wstar, bstar] = MiniBatchGD(X, Y, GDparams, W, b, lambda)
P = EvaluateClassifier(X, W, b);
[grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda);
Wstar = W - GDparams.eta * grad_W;
bstar = b - (GDparams.eta * grad_b);
end