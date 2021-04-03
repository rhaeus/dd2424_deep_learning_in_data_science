addpath datasets/cifar-10-batches-mat/;

% load data
disp('load data')
d = 3072;
n = 10000;
K = 10;

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

% use last 1000 columns as validation set
X_valid = X_train(:, end-1000+1:end);
Y_valid = Y_train(:, end-1000+1:end);
y_valid = y_train(:, end-1000+1:end);

% remove last 1000 columns from training set
X_train(:, end-1000+1:end) = [];
Y_train(:, end-1000+1:end) = [];
y_train(:, end-1000+1:end) = [];

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
n_epochs = 500;
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
    for j=randperm(n/gd_params.n_batch)
%     for j=1:n/gd_params.n_batch
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
    
    diff = abs(loss_training(i) - loss_validation(i));
    difference(i) = diff;
    
    fprintf(fid, '%d;%0.5f;%0.5f;%0.5f;%0.5f\n', i, loss_training(i), loss_validation(i), difference(i), accuracy(i));
    
    if mod(i, snapshot_step) == 0
        % take snapshot
        % Plots the weights
        figure(1);
        for j=1:10
            im = reshape(W(j, :), 32, 32, 3);
            s_im{j} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
            s_im{j} = permute(s_im{j}, [2, 1, 3]);
        end
        montage(s_im, 'Size', [2,5]);
        name = sprintf('result_pics/%d_weights.png', i);
        saveas(gcf,name);

        % evolution of the loss as diagram
        figure(2);
        x = 1:i;
        plot(x, loss_training(1:i, :), x, loss_validation(1:i,:));
        title('Loss')
        legend('Training', 'Validation')
        name = sprintf('result_pics/%d_loss.png', i);
        saveas(gcf,name);

        % evolution of the accuracy as diagram
        figure(3);
        x = 1:i;
        plot(x, accuracy(1:i,:));
        title('Accuracy')
        legend('Test')
        name = sprintf('result_pics/%d_accuracy.png', i);
        saveas(gcf,name);
        
        % evolution of the difference as diagram
        figure(4);
        x = 1:i;
        plot(x, difference(1:i,:));
        title('Difference')
        name = sprintf('result_pics/%d_difference.png', i);
        saveas(gcf,name);
    end
    
    fprintf('training loss: %0.3f\n', loss_training(i))
    fprintf('validation loss: %0.3f\n', loss_validation(i))
    fprintf('loss diff: %0.3f\n', diff)
    
    if diff > 0.5
        final_epoch = i;
        fprintf('overfit start at epoch %d. Abort.\n', i)
        break
    end
end
fclose(fid);
disp('training done')

for i=1:gd_params.n_epochs/snapshot_step
    index = i*snapshot_step;
    if index > final_epoch
        fprintf('epoch %d\n', final_epoch)
        fprintf('final training loss %0.3f\n', loss_training(final_epoch));
        fprintf('final validation loss %0.3f\n', loss_validation(final_epoch));
        fprintf('final accuracy %0.4f\n', accuracy(final_epoch));
        fprintf('final difference %0.4f\n', difference(final_epoch));
        break
    end
    
    fprintf('epoch %d\n', index)
    fprintf('final training loss %0.3f\n', loss_training(index));
    fprintf('final validation loss %0.3f\n', loss_validation(index));
    fprintf('final accuracy %0.4f\n', accuracy(index));
    fprintf('final difference %0.4f\n', difference(index));
end




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