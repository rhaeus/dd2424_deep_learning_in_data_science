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
gd_params = GDparams(100, 0.001, 40);

loss_training = zeros(gd_params.n_epochs, 1);
loss_validation = zeros(gd_params.n_epochs, 1);
accuracy = zeros(gd_params.n_epochs, 1);

for i=1:gd_params.n_epochs
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

