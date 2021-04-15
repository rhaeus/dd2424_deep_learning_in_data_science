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
lambda = 0.01;
n_batch = 100;
% eta = 0.01;
% n_epochs = 200;
eta_min = 1e-5; 
eta_max = 1e-1;
n_s = 800;
cycles = 3;


updates_per_cycle = 2*n_s
snapshots_per_cycle = 10;
snapshot_step = updates_per_cycle/snapshots_per_cycle

% gd_params = GDparams(n_batch, eta, n_epochs);

% fprintf('lambda=%0.5f\nn_batch=%d\neta=%0.5f\nn_epochs=%d\n', lambda, n_batch, eta, n_epochs);

amount = snapshots_per_cycle*cycles + 1;
loss_training = zeros(amount, 1);
loss_validation = zeros(amount, 1);

cost_training = zeros(amount, 1);
cost_validation = zeros(amount, 1);

accuracy_training = zeros(amount, 1);
accuracy_validation = zeros(amount, 1);
accuracy_test = zeros(amount, 1);

difference = zeros(amount, 1);
x_axis = zeros(amount, 1);
etas = zeros(cycles*updates_per_cycle, 1);

% loss_training = [];
% loss_validation = [];
% accuracy = [];
% difference = [];
% x_axis = [];


% train
disp('begin training')
ex = 4;

name = sprintf('result_pics/ex%d_values.csv', ex);
delete(name);
fid = fopen(name, 'a+');
fprintf(fid, 'cycle;count;loss_training;loss_validation;cost_training;cost_validation;accuracy_training;accuracy_validation;accuracy_test\n');


j = 1;
count = 0;
for l=0:cycles-1
    fprintf('cycle %d of %d.\n',l+1,cycles);
    
    for t = 2*l*n_s:(2*l+1)*n_s-1
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        Xbatch = X_train(:, j_start:j_end);
        Ybatch = Y_train(:, j_start:j_end);
        eta = eta_min + (t-2*l*n_s)/n_s*(eta_max-eta_min);
        etas(count+1) = eta;
        
        
        j = j + 1;
        if j > n/n_batch
            j = 1;
        end

        [Ws, bs] = MiniBatchGD(Xbatch, Ybatch, eta, Ws, bs, lambda);
        

        count = count + 1;
        if count == 1 || mod(count, snapshot_step) == 0 
            if count == 1
                index = 1;
            else
                index = count/snapshot_step+1;
            end
            
            % A loss function/error function is for a single training example/input. 
            % A cost function, on the other hand, is the average loss over the entire training dataset.
            loss_training(index) = ComputeCost(X_train(:,1), Y_train(:,1), Ws, bs, lambda);
            loss_validation(index) = ComputeCost(X_valid(:,1), Y_valid(:,1), Ws, bs, lambda);
            
            cost_training(index) = ComputeCost(X_train, Y_train, Ws, bs, lambda);
            cost_validation(index) = ComputeCost(X_valid, Y_valid, Ws, bs, lambda);
            
            accuracy_training(index) = ComputeAccuracy(X_train, y_train, Ws, bs);
            accuracy_validation(index) = ComputeAccuracy(X_valid, y_valid, Ws, bs);
            accuracy_test(index) = ComputeAccuracy(X_test, y_test, Ws, bs);
            
            x_axis(index) = count;

            diff = abs(loss_training(index) - loss_validation(index));
            difference(index) = diff;

            fprintf(fid, '%d;%d;%0.5f;%0.5f;%0.5f;%0.5f;%0.5f;%0.5f;%0.5f\n', l+1, x_axis(index), loss_training(index), loss_validation(index), cost_training(index), cost_validation(index), accuracy_training(index), accuracy_validation(index), accuracy_test(index));
        end     
    end
    
    for t = (2*l+1)*n_s:2*(l+1)*n_s-1
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        Xbatch = X_train(:, j_start:j_end);
        Ybatch = Y_train(:, j_start:j_end);
        eta = eta_max - (t-(2*l+1)*n_s)/n_s*(eta_max-eta_min);
        etas(count+1) = eta;
        
        j = j + 1;
        if j > n/n_batch
            j = 1;
        end

        [Ws, bs] = MiniBatchGD(Xbatch, Ybatch, eta, Ws, bs, lambda);
        
        
        count = count + 1;
        if count == 1 || mod(count, snapshot_step) == 0 
            if count == 1
                index = 1;
            else
                index = count/snapshot_step+1;
            end
            % A loss function/error function is for a single training example/input. 
            % A cost function, on the other hand, is the average loss over the entire training dataset.
            loss_training(index) = ComputeCost(X_train(:,1), Y_train(:,1), Ws, bs, lambda);
            loss_validation(index) = ComputeCost(X_valid(:,1), Y_valid(:,1), Ws, bs, lambda);
            
            cost_training(index) = ComputeCost(X_train, Y_train, Ws, bs, lambda);
            cost_validation(index) = ComputeCost(X_valid, Y_valid, Ws, bs, lambda);
            
            accuracy_training(index) = ComputeAccuracy(X_train, y_train, Ws, bs);
            accuracy_validation(index) = ComputeAccuracy(X_valid, y_valid, Ws, bs);
            accuracy_test(index) = ComputeAccuracy(X_test, y_test, Ws, bs);
            
            x_axis(index) = count;

            diff = abs(loss_training(index) - loss_validation(index));
            difference(index) = diff;

            fprintf(fid, '%d;%d;%0.5f;%0.5f;%0.5f;%0.5f;%0.5f;%0.5f;%0.5f\n', l+1, x_axis(index), loss_training(index), loss_validation(index), cost_training(index), cost_validation(index), accuracy_training(index), accuracy_validation(index), accuracy_test(index));
        end

    end
end

fclose(fid);
disp('training done')

 % evolution of the loss as diagram
figure(2);
plot(x_axis, loss_training, x_axis, loss_validation);
title('Loss')
legend('Training', 'Validation')
name = sprintf('result_pics/ex%d_loss.png', ex);
saveas(gcf,name);

 % evolution of the cost as diagram
figure(3);
plot(x_axis, cost_training, x_axis, cost_validation);
title('Cost')
legend('Training', 'Validation')
name = sprintf('result_pics/ex%d_cost.png', ex);
saveas(gcf,name);

% evolution of the accuracy as diagram
figure(4);
plot(x_axis, accuracy_training, x_axis, accuracy_validation, x_axis, accuracy_test);
title('Accuracy')
legend('Training', 'Validation','Test')
name = sprintf('result_pics/ex%d_accuracy.png', ex);
saveas(gcf,name);

%evolution of the eta as diagram
figure(5);
x=1:count;
plot(x, etas);
title('eta')
name = sprintf('result_pics/ex%d_eta.png', ex);
saveas(gcf,name);

% evolution of the difference as diagram
% figure(4);
% plot(x_axis, difference);
% title('Difference')
% name = sprintf('result_pics/%d_difference.png', i);
% saveas(gcf,name);



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