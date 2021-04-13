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
[K, n] = size(Y_train);
m = 50; % number of nodes in hidden layer

[Ws, bs] = InitModel(m,d,K);
% Ws

%d=20, n = 2;

disp('compute grad')
lambda = 0;
[ngrad_b, ngrad_W] = ComputeGradsNumSlow(X_train(:, 1:2), Y_train(:, 1:2), Ws, bs, lambda, 1e-5);
disp('slow done')

[P, Xs] = EvaluateClassifier(X_train(:, 1:2), Ws, bs);
disp('evaluate done')
[grad_W, grad_b] = ComputeGradients(Xs, Y_train(:, 1:2), P, Ws, lambda);
disp('my done')

[k, ~] = size(Xs);

for i = 1:k
%     ngrad_W{i}(:,1:8) 
%     grad_W{i}(:,1:8) 
    diff_W = abs(ngrad_W{i} - grad_W{i});
    diff_b = abs(ngrad_b{i} - grad_b{i});
    
    disp(i)

    if all(diff_W < 1e-5)
        disp('W ok')
    else
        disp('W not ok')
        disp(max(max(diff_W)))
    end

    if all(diff_b < 1e-5)
        disp('b ok')
    else
        disp('b not ok')
        disp(max(max(diff_b)))
    end
end


% functions
function [Ws, bs] = InitModel(m, d, K)
rng(400);

W1 = randn(m,d)/sqrt(d);
b1 = zeros(m,1);

W2 = randn(K,m)/sqrt(m);
b2 = zeros(K,1);

Ws = {W1; W2};
bs = {b1; b2};
end


function acc = ComputeAccuracy(X, y, W, b)
P = EvaluateClassifier(X, W, b); % Kxn
[K,n] = size(P); % 10x10000
[~, p] = max(P); % p is index of max, 1xn
acc = sum(p==y) / n;
end



function [grad_Ws, grad_bs] = ComputeGradients(Xs, Y, P, Ws, lambda)
[K,n] = size(P);
[k, ~] = size(Ws);

grad_Ws = cell(k, 1);
grad_bs = cell(k, 1);

G_batch = -(Y - P); %Kxn
One = ones(n,1);

for l=k:-1:2
    % Kxn * nxd = Kxd
    grad_Ws{l} = (G_batch * Xs{l}') / n + 2 * lambda * Ws{l};
    grad_bs{l} = (G_batch * One) / n;
    
    G_batch = Ws{l}' * G_batch;
    G_batch = G_batch .* (Xs{l} > 0);
end

grad_Ws{1} = (G_batch * Xs{1}') / n + 2 * lambda * Ws{1};
grad_bs{1} = (G_batch * One) / n;
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