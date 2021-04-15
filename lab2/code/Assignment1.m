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
lambda = 0;
n_batch = 100;
eta = 0.01;
n_epochs = 200;
snapshot_step = 100;

gd_params = GDparams(n_batch, eta, n_epochs);

fprintf('lambda=%0.5f\nn_batch=%d\neta=%0.5f\nn_epochs=%d\n', lambda, n_batch, eta, n_epochs);

loss_training = zeros(gd_params.n_epochs, 1);
loss_validation = zeros(gd_params.n_epochs, 1);
accuracy = zeros(gd_params.n_epochs, 1);
difference = zeros(gd_params.n_epochs, 1);


% train
disp('begin training')
final_epoch = gd_params.n_epochs;

delete 'result_pics/values.csv';
fid = fopen('result_pics/values.csv', 'a+');
fprintf(fid, 'epoch;training_loss;validation_loss;difference;accuracy\n');


for i=1:gd_params.n_epochs
    fprintf('epoch %d of %d.\n',i,gd_params.n_epochs);
%     for j=randperm(n/gd_params.n_batch)
%     for j=1:n/gd_params.n_batch
%         j_start = (j-1)*gd_params.n_batch + 1;
%         j_end = j*gd_params.n_batch;
%         inds = j_start:j_end;
%         Xbatch = X_train(:, j_start:j_end);
%         Ybatch = Y_train(:, j_start:j_end);
        X = X_train(:,1:100);
        Y = Y_train(:,1:100);
        
        [Ws, bs] = MiniBatchGD(X, Y, gd_params, Ws, bs, lambda);
        
%     end
    
    loss_training(i) = ComputeCost(X_train(:,1:100), Y_train(:,1:100), Ws, bs, lambda);
    loss_validation(i) = ComputeCost(X_valid(:,1:100), Y_valid(:,1:100), Ws, bs, lambda);
    accuracy(i) = ComputeAccuracy(X_test, y_test, Ws, bs);
    
    diff = abs(loss_training(i) - loss_validation(i));
    difference(i) = diff;
    
    fprintf(fid, '%d;%0.5f;%0.5f;%0.5f;%0.5f\n', i, loss_training(i), loss_validation(i), difference(i), accuracy(i));
    
    fprintf('training loss: %0.3f\n', loss_training(i))
    fprintf('validation loss: %0.3f\n', loss_validation(i))
    fprintf('loss diff: %0.3f\n', diff)
    
end
fclose(fid);
disp('training done')



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

function [Ws_star, bs_star] = MiniBatchGD(X, Y, GDparams, Ws, bs, lambda)
[P, Xs] = EvaluateClassifier(X, Ws, bs); % Kxn
[grad_Ws, grad_bs] = ComputeGradients(Xs, Y, P, Ws, lambda);

[k, ~] = size(Ws);
Ws_star = cell(k, 1);
bs_star = cell(k, 1);

for i=1:k
    Ws_star{i} = Ws{i} - GDparams.eta * grad_Ws{i};
    bs_star{i} = bs{i} - GDparams.eta * grad_bs{i};
end
end