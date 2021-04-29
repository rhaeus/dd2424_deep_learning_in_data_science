addpath datasets/cifar-10-batches-mat/;

% load data
disp('load data')
d = 3072;
n = 10000;
K = 10;

validation_amount = 1000;

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
ms = [50; 30; 20; 20; 10; 10; 10; 10]; % number of nodes in hidden layer

[Ws, bs] = InitModel(ms,d,K);

n_batch = 100;
eta_min = 1e-5; 
eta_max = 1e-1;
% n_s = 2*floor(n / n_batch);
n_s = 5*45000/n_batch;
cycles = 2;

%%%%%%%%%%%%%%%%%%%%%%%
% random lambda search
%%%%%%%%%%%%%%%%%%%%%%%

% l_min=-5;
% l_max=-2;
% 
% lambda_count = 20;
% lambdas = zeros(lambda_count,1);

% for i=1:lambda_count
%     l = l_min + (l_max - l_min)*rand(1, 1);
%     lambdas(i) = 10^l;
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% uniform lambda search
%%%%%%%%%%%%%%%%%%%%%%%%%%%

% l_min = 10^-5;
% l_max = 10^-1;
% step = (l_max - l_min)/lambda_count;
% for i=0:lambda_count-1
%     lambdas(i+1) = l_min + i*step;
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% perform lambda search
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ex=4;
% name = sprintf('result_pics/ex%d_lambda_fine_search_random_1.csv', ex);
% delete(name);
% fid = fopen(name, 'a+');
% fprintf(fid, 'l_min=%0.5f;l_max=%0.5f;cycles=%d;n_s=%d\n', l_min, l_max, cycles, n_s);
% fprintf(fid, 'lambda;accuracy_validation;accuracy_test\n');
% 
% for i=1:lambda_count
%     fprintf('%d/%d: lambda=%0.5f\n', i, lambda_count,lambdas(i));
%     [Ws, bs] = InitModel(m,d,K);
%     [acc_valid, acc_test] = train(X_train, Y_train, X_valid, y_valid,X_test, y_test, Ws, bs, cycles, n_s, eta_max, eta_min, n_batch, lambdas(i));
%     fprintf('accuracy_validation = %0.5f\n', acc_valid);
%     fprintf('accuracy_test = %0.5f\n', acc_test);
%     fprintf(fid, '%0.5f;%0.5f;%0.5f\n', lambdas(i), acc_valid, acc_test);
% end
% fclose(fid);

%%%%%%%%%%%%%%%%%%%%
% final training
%%%%%%%%%%%%%%%%%%%%
lambda = 0.005;
[acc_valid, acc_test] = train(X_train, Y_train, y_train,X_valid, Y_valid, y_valid,X_test, Y_test, y_test, Ws, bs, cycles, n_s, eta_max, eta_min, n_batch, lambda);


disp('done')


function [acc_valid, acc_test] = train(X_train, Y_train,y_train, X_valid,Y_valid, y_valid, X_test, Y_test, y_test, Ws, bs, cycles, n_s, eta_max, eta_min, n_batch, lambda)
[d, n] = size(X_train);
[k, ~] = size(Ws);

updates_per_cycle = 2*n_s;
snapshots_per_cycle = 10;
snapshot_step = updates_per_cycle/snapshots_per_cycle;


amount = snapshots_per_cycle*cycles + 1;
loss_training = zeros(amount, 1);
loss_validation = zeros(amount, 1);

cost_training = zeros(amount, 1);
cost_validation = zeros(amount, 1);

accuracy_training = zeros(amount, 1);
accuracy_validation = zeros(amount, 1);
accuracy_test = zeros(amount, 1);

x_axis = zeros(amount, 1);
etas = zeros(cycles*updates_per_cycle, 1);

name = sprintf('result_pics/values_lambda=%0.5f_ns=%d_cycles=%d_k=%d.csv', lambda, n_s,cycles,k);
delete(name);
fid = fopen(name, 'a+');
fprintf(fid, 'cycle;count;loss_training;loss_validation;cost_training;cost_validation;accuracy_training;accuracy_validation;accuracy_test\n');


shuffled_batch = randperm(n/n_batch);
j = 1;
count = 0;
for l=0:cycles-1
    fprintf('cycle %d of %d.\n',l+1,cycles);
    
    for t = 2*l*n_s:(2*l+1)*n_s-1
        j_start = (shuffled_batch(j)-1)*n_batch + 1;
        j_end = shuffled_batch(j)*n_batch;
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
            loss_training(index) = ComputeCost(X_train(:,1:n_batch), Y_train(:,1:n_batch), Ws, bs, lambda);
            loss_validation(index) = ComputeCost(X_valid(:,1:n_batch), Y_valid(:,1:n_batch), Ws, bs, lambda);
            
            cost_training(index) = ComputeCost(X_train, Y_train, Ws, bs, lambda);
            cost_validation(index) = ComputeCost(X_valid, Y_valid, Ws, bs, lambda);
            
            accuracy_training(index) = ComputeAccuracy(X_train, y_train, Ws, bs);
            accuracy_validation(index) = ComputeAccuracy(X_valid, y_valid, Ws, bs);
            accuracy_test(index) = ComputeAccuracy(X_test, y_test, Ws, bs);
            
            x_axis(index) = count;

            fprintf(fid, '%d;%d;%0.5f;%0.5f;%0.5f;%0.5f;%0.5f;%0.5f;%0.5f\n', l+1, x_axis(index), loss_training(index), loss_validation(index), cost_training(index), cost_validation(index), accuracy_training(index), accuracy_validation(index), accuracy_test(index));
        end     
    end
    
    for t = (2*l+1)*n_s:2*(l+1)*n_s-1
        j_start = (shuffled_batch(j)-1)*n_batch + 1;
        j_end = shuffled_batch(j)*n_batch;
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
            loss_training(index) = ComputeCost(X_train(:,1:n_batch), Y_train(:,1:n_batch), Ws, bs, lambda);
            loss_validation(index) = ComputeCost(X_valid(:,1:n_batch), Y_valid(:,1:n_batch), Ws, bs, lambda);
            
            cost_training(index) = ComputeCost(X_train, Y_train, Ws, bs, lambda);
            cost_validation(index) = ComputeCost(X_valid, Y_valid, Ws, bs, lambda);
            
            accuracy_training(index) = ComputeAccuracy(X_train, y_train, Ws, bs);
            accuracy_validation(index) = ComputeAccuracy(X_valid, y_valid, Ws, bs);
            accuracy_test(index) = ComputeAccuracy(X_test, y_test, Ws, bs);
            
            x_axis(index) = count;

            fprintf(fid, '%d;%d;%0.5f;%0.5f;%0.5f;%0.5f;%0.5f;%0.5f;%0.5f\n', l+1, x_axis(index), loss_training(index), loss_validation(index), cost_training(index), cost_validation(index), accuracy_training(index), accuracy_validation(index), accuracy_test(index));
        end     
        
    end
end

acc_valid = ComputeAccuracy(X_valid, y_valid, Ws, bs);
acc_test = ComputeAccuracy(X_test, y_test, Ws, bs);

cost_train = ComputeCost(X_train, Y_train, Ws, bs, lambda);
cost_valid = ComputeCost(X_valid, Y_valid, Ws, bs, lambda);
cost_test = ComputeCost(X_test, Y_test, Ws, bs, lambda);

% terminal output
fprintf('accuracy_validation = %0.5f\n', acc_valid);
fprintf('accuracy_test = %0.5f\n', acc_test);
fprintf('cost_train = %0.5f\n', cost_train);
fprintf('cost_valid = %0.5f\n', cost_valid);
fprintf('cost_test = %0.5f\n', cost_test);

%plot
% evolution of the loss as diagram
figure(2);
plot(x_axis, loss_training, x_axis, loss_validation);
title('Loss')
legend('Training', 'Validation')
name = sprintf('result_pics/loss_lambda=%0.5f_ns=%d_cycles=%d_k=%d.png', lambda, n_s, cycles,k);
saveas(gcf,name);

 % evolution of the cost as diagram
figure(3);
plot(x_axis, cost_training, x_axis, cost_validation);
title('Cost')
legend('Training', 'Validation')
name = sprintf('result_pics/cost_lambda=%0.5f_ns=%d_cycles=%d_k=%d.png', lambda, n_s, cycles,k);
saveas(gcf,name);

% evolution of the accuracy as diagram
figure(4);
plot(x_axis, accuracy_training, x_axis, accuracy_validation, x_axis, accuracy_test);
title('Accuracy')
legend('Training', 'Validation','Test')
name = sprintf('result_pics/accuracy_lambda=%0.5f_ns=%d_cycles=%d_k=%d.png', lambda, n_s, cycles,k);
saveas(gcf,name);

%evolution of the eta as diagram
figure(5);
x=1:count;
plot(x, etas);
title('eta')
name = sprintf('result_pics/eta_lambda=%0.5f_ns=%d_cycles=%d_k=%d.png', lambda, n_s, cycles,k);
saveas(gcf,name);

end

% functions
function [Ws, bs] = InitModel(ms, d, K)
rng(400);
%Xavier initialization
[k, ~] = size(ms); %given are k-1 hidden layers
k = k + 1; %k is number of layers

Ws = cell(k, 1);
bs = cell(k, 1);

Ws{1} = randn(ms(1),d)/sqrt(d);
bs{1} = zeros(ms(1),1);

for i=2:k-1
    Ws{i} = randn(ms(i),ms(i-1))/sqrt(ms(i-1));
    bs{i} = zeros(ms(i),1);
end

Ws{k} = randn(K,ms(k-1))/sqrt(ms(k-1));
bs{k} = zeros(K,1);

% W2 = randn(m,m)/sqrt(m);
% b2 = zeros(m,1);
% 
% W3 = randn(K,m)/sqrt(m);
% b3 = zeros(K,1);

% Ws = {W1; W2; W3};
% bs = {b1; b2;b3};
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

function J = ComputeCost(X, Y, Ws, bs, lambda)
[d,n] = size(X);
P = EvaluateClassifier(X, Ws, bs); % Kxn

reg = 0;
[k, ~] = size(Ws);
for i=1:k
    reg = reg + sum(sum(Ws{i} .* Ws{i}));
end

l = mean(-mean(sum(Y .* log(P)), 1));
J = l + lambda * reg;
end

function [P, Xs] = EvaluateClassifier(X, Ws, bs)
[k, ~] = size(Ws);
Xs = cell(k, 1);
Xs{1} = X;

for i=1:k-1
    s1 = Ws{i} * Xs{i} + bs{i}; % mxn
    h = max(0, s1); % mxn
    Xs{i+1} = h;
end

s = Ws{k} * Xs{k} + bs{k}; %Kxn
P = exp(s) ./ sum(exp(s)); %Kxn
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