addpath datasets/cifar-10-batches-mat/;

% load data
disp('load data')
d = 3072;
n = 10000;
K = 10;

validation_amount = 5000;

X_train = zeros(d,n*5);
Y_train = zeros(K,n*5);
y_train = zeros(1,n*5);

for i=0:4
    filename = sprintf('data_batch_%d.mat', i+1);
    [X, Y, y] = LoadBatch(filename);
    X_train(:,i*n+1:(i+1)*n) = X;
    Y_train(:,i*n+1:(i+1)*n) = Y;
    y_train(:,i*n+1:(i+1)*n) = y;
end

% use last validation_amount columns as validation set
X_valid = X_train(:, end-validation_amount+1:end);
Y_valid = Y_train(:, end-validation_amount+1:end);
y_valid = y_train(:, end-validation_amount+1:end);

% remove last validation_amount columns from training set
X_train(:, end-validation_amount+1:end) = [];
Y_train(:, end-validation_amount+1:end) = [];
y_train(:, end-validation_amount+1:end) = [];

mean_X = mean(X_train, 2); %dx1
std_X = std(X_train, 0, 2); %dx1

% pre process data
X_train = X_train - repmat(mean_X, [1, size(X_train,2)]);
X_train = X_train ./ repmat(std_X, [1, size(X_train,2)]);

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
% lambda = 0.01;
n_batch = 100;

eta_min = 1e-5; 
eta_max = 1e-1;
n_s = 2*floor(n / n_batch);
cycles = 5;

l_min=-5;
l_max=-2;

lambda_count = 20;
lambdas = zeros(lambda_count,1);
% random
for i=1:lambda_count
    l = l_min + (l_max - l_min)*rand(1, 1);
    lambdas(i) = 10^l;
end

% l_min = 10^-5;
% l_max = 10^-1;
% step = (l_max - l_min)/lambda_count;
% for i=0:lambda_count-1
%     lambdas(i+1) = l_min + i*step;
% end

ex=4;
name = sprintf('result_pics/ex%d_lambda_fine_search_random_1.csv', ex);
delete(name);
fid = fopen(name, 'a+');
fprintf(fid, 'l_min=%0.5f;l_max=%0.5f;cycles=%d;n_s=%d\n', l_min, l_max, cycles, n_s);
fprintf(fid, 'lambda;accuracy_validation;accuracy_test\n');

for i=1:lambda_count
    fprintf('%d/%d: lambda=%0.5f\n', i, lambda_count,lambdas(i));
    [Ws, bs] = InitModel(m,d,K);
    [acc_valid, acc_test] = train(X_train, Y_train, X_valid, y_valid,X_test, y_test, Ws, bs, cycles, n_s, eta_max, eta_min, n_batch, lambdas(i));
    fprintf('accuracy_validation = %0.5f\n', acc_valid);
    fprintf('accuracy_test = %0.5f\n', acc_test);
    fprintf(fid, '%0.5f;%0.5f;%0.5f\n', lambdas(i), acc_valid, acc_test);
end

fclose(fid);
disp('search done')


function [acc_valid, acc_test] = train(X_train, Y_train, X_valid, y_valid,X_test, y_test, Ws, bs, cycles, n_s, eta_max, eta_min, n_batch, lambda)
[d, n] = size(X_train);
j=1;
for l=0:cycles-1
    fprintf('cycle %d of %d.\n',l+1,cycles);
    
    for t = 2*l*n_s:(2*l+1)*n_s-1
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        Xbatch = X_train(:, j_start:j_end);
        Ybatch = Y_train(:, j_start:j_end);
        eta = eta_min + (t-2*l*n_s)/n_s*(eta_max-eta_min);
        
        j = j + 1;
        if j > n/n_batch
            j = 1;
        end

        [Ws, bs] = MiniBatchGD(Xbatch, Ybatch, eta, Ws, bs, lambda);
    end
    
    for t = (2*l+1)*n_s:2*(l+1)*n_s-1
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        Xbatch = X_train(:, j_start:j_end);
        Ybatch = Y_train(:, j_start:j_end);
        eta = eta_max - (t-(2*l+1)*n_s)/n_s*(eta_max-eta_min);
        
        j = j + 1;
        if j > n/n_batch
            j = 1;
        end

        [Ws, bs] = MiniBatchGD(Xbatch, Ybatch, eta, Ws, bs, lambda);
        
    end
end

acc_valid = ComputeAccuracy(X_valid, y_valid, Ws, bs);
acc_test = ComputeAccuracy(X_test, y_test, Ws, bs);

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


function acc = ComputeAccuracy(X, y, Ws, bs)
[P, Xs] = EvaluateClassifier(X, Ws, bs); % Kxn
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

function [Ws_star, bs_star] = MiniBatchGD(X, Y, eta, Ws, bs, lambda)
[P, Xs] = EvaluateClassifier(X, Ws, bs); % Kxn
[grad_Ws, grad_bs] = ComputeGradients(Xs, Y, P, Ws, lambda);

[k, ~] = size(Ws);
Ws_star = cell(k, 1);
bs_star = cell(k, 1);

for i=1:k
    Ws_star{i} = Ws{i} - eta * grad_Ws{i};
    bs_star{i} = bs{i} - eta * grad_bs{i};
end
end