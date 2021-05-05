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
% ms = [50; 30; 20; 20; 10; 10; 10; 10]; % number of nodes in hidden layer
ms = [50;50]; % number of nodes in hidden layer


[Ws, bs] = InitModel(ms,d,K);
% size(ms)
% size(Ws)

[n_hidden, ~] = size(ms);

gammas = cell(n_hidden, 1);
betas = cell(n_hidden, 1);

for i=1:n_hidden
    gammas{i} = ones(ms(i),1);
    betas{i} = zeros(ms(i),1);
end

NetParams.W = Ws;
NetParams.b = bs;
NetParams.use_bn = true;
NetParams.gammas = gammas;
NetParams.betas = betas;

n_batch = 100;
eta_min = 1e-5; 
eta_max = 1e-1;
% n_s = 2*floor(n / n_batch);
n_s = 5*45000/n_batch;
cycles = 2;

testGradients(X_train, Y_train, NetParams, 0);

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
% lambda = 0.005;
% [acc_valid, acc_test] = train(X_train, Y_train, y_train,X_valid, Y_valid, y_valid,X_test, Y_test, y_test, Ws, bs, cycles, n_s, eta_max, eta_min, n_batch, lambda);


disp('done')

function testGradients(X_train, Y_train, NetParams, lambda)

disp('compute grads num')

grads = ComputeGradsNumSlow(X_train(:, 1:2), Y_train(:, 1:2), NetParams, lambda, 1e-5);
save numgrads3layer.mat grads
% load numgrads2layer.mat

disp('compute grads analytic')
[P, Xs, Ss, Shats, vs, mus] = EvaluateClassifier(X_train(:, 1:2), NetParams);
disp('evaluate done')

[grad_W, grad_b, grad_gammas, grad_betas] = ComputeGradients(Xs, Y_train(:, 1:2), P, NetParams.W, Ss, Shats, mus, vs, NetParams.gammas, NetParams.betas, lambda);
disp('done')

[k, ~] = size(Xs);
% grad_betas{1}

ngrad_W = grads.W;
ngrad_b = grads.b;
ngrad_gammas = grads.gammas;
ngrad_betas = grads.betas;

disp('size W, b, gamma, beta')
size(ngrad_W)
size(ngrad_b)
size(ngrad_gammas)
size(ngrad_betas)

for i = 1:k
%     ngrad_W{i}(:,1:8) 
%     grad_W{i}(:,1:8) 
    diff_W = abs(ngrad_W{i} - grad_W{i});
    diff_b = abs(ngrad_b{i} - grad_b{i});
    
    if i < k
        diff_gamma = abs(ngrad_gammas{i} - grad_gammas{i});
        diff_beta = abs(ngrad_betas{i} - grad_betas{i});
    end

    fprintf('layer %d:\n', i);

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
    
    if i < k
        if all(diff_gamma < 1e-5)
            disp('gamma ok')
        else
            disp('gamma not ok')
            disp(max(max(diff_gamma)))
        end

        if all(diff_beta < 1e-5)
            disp('beta ok')
        else
            disp('beta not ok')
            disp(max(max(diff_beta)))
        end
    end
end

end


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
            shuffled_batch = randperm(n/n_batch); % new epoch - new shuffle
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
            shuffled_batch = randperm(n/n_batch); % new epoch - new shuffle
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

function [G_batch] = BatchNormBackPass(G_batch, s, mu, v)
[m, ~] = size(v);
% epsilon = 1e-10;
[~,n] = size(G_batch);
One = ones(n,1);


% sigma1 = transpose((v + eps).^(-0.5));
% sigma2 = transpose((v + eps).^(-1.5));



sigma1 = ((v + eps).^(-0.5));
sigma2 = ((v + eps).^(-1.5));

% sigma1 = zeros(m,1);
% sigma2 = zeros(m,1);
% 
% for i=1:m
%     sigma1(i,1) = (v(i) + eps) .^ (-0.5);
%     sigma2(i,1) = (v(i) + eps) .^ (-1.5);
% end
% sigma1 = sigma1';
% sigma2 = sigma2';

% size(One')
% size(sigma1)

G1 = G_batch .* (sigma1 * One');
G2 = G_batch .* (sigma2 * One');

D = s - mu * One';
c = (G2 .* D) * One;

G_batch = G1 - (1/n) *(G1 * One)* One' - (1/n) * D .* (c*One');
end

function [grad_Ws, grad_bs, grad_gammas, grad_betas] = ComputeGradients(Xs, Y, P, Ws, Ss, shats, mus, vs, gammas, betas, lambda)
[K,n] = size(P);
[k, ~] = size(Ws);

grad_Ws = cell(k, 1);
grad_bs = cell(k, 1);
grad_gammas = cell(k-1, 1);
grad_betas = cell(k-1, 1);

% Propagate the gradient through the loss and softmax operations
G_batch = -(Y - P); %Kxn
One = ones(n,1);

% The gradients of J w.r.t. bias vector b k and W k
grad_Ws{k} = (G_batch * Xs{k}') / n + 2 * lambda * Ws{k};
grad_bs{k} = (G_batch * One) / n;

% Propagate G batch to the previous layer
G_batch = Ws{k}' * G_batch;
G_batch = G_batch .* (Xs{k} > 0);


for l=k-1:-1:1
    % 1. Compute gradient for the scale and offset parameters for layer l:
    grad_gammas{l} = ((G_batch .* shats{l}) * One) / n;
    grad_betas{l} = (G_batch * One) / n;
    
    % 2. Propagate the gradients through the scale and shift
    G_batch = G_batch .* (gammas{l} * One');
    
    % 3. Propagate G batch through the batch normalization
    G_batch =  BatchNormBackPass(G_batch, Ss{l}, mus{l}, vs{l});
    
    % 4. The gradients of J w.r.t. bias vector b l and W l
    % Kxn * nxd = Kxd
    grad_Ws{l} = (G_batch * Xs{l}') / n + 2 * lambda * Ws{l};
    grad_bs{l} = (G_batch * One) / n;
    
    % 5. If l > 1 propagate G batch to the previous layer
    if l > 1
        % Propagate G batch to the previous layer
        G_batch = Ws{l}' * G_batch;
        G_batch = G_batch .* (Xs{l} > 0);
    end
end

end

function J = ComputeCost(X, Y, NetParams, lambda)
Ws = NetParams.W;
bs = NetParams.b;

[d,n] = size(X);
[P, Xs, Ss, Shats, vs, mus] = EvaluateClassifier(X, NetParams); % Kxn

reg = 0;
[k, ~] = size(Ws);
for i=1:k
    reg = reg + sum(sum(Ws{i} .* Ws{i}));
end

l = mean(-mean(sum(Y .* log(P)), 1));
J = l + lambda * reg;
end

function [shat] = BatchNormalize(s, mu, var)
d = diag(var+eps) ^ (-1/2);
shat = d * (s-mu);
end

function [P, Xs, S, S_hat, variance, mu] = EvaluateClassifier(X, NetParams, varargin)
n = size(X, 2);

[k, ~] = size(NetParams.W);
Xs = cell(k, 1);
S = cell(k, 1);

S_hat = cell(k-1, 1);

if nargin == 3
    mu = varargin{1};
    variance = varargin{2};
else
    variance = cell(k-1, 1);
    mu = cell(k-1, 1);
end


Xs{1} = X;


for i = 1:k-1
    s_i = NetParams.W{i} * Xs{i} + NetParams.b{i};
   
    if NetParams.use_bn
        
        if nargin == 3
            s_hat_i = BatchNormalize(s_i, mu{i}, variance{i});
        else
            mu_i = mean(s_i, 2);
            var_i = var(s_i, 0, 2);
            var_i = var_i * (n-1) / n;
            variance{i} = var_i;
            mu{i} = mu_i;
            
            s_hat_i = BatchNormalize(s_i, mu_i, var_i);
        end
        
        s_tilde_i = NetParams.gammas{i} .* s_hat_i + NetParams.betas{i};
        next_x = max(0, s_tilde_i);
        
        S_hat{i} = s_hat_i;
        
%         var_i
%         mu_i
%         var{i} = transpose(var_i);
        
        
    else
        next_x = max(0, s_i);
    end
    S{i} = s_i;
    Xs{i+1} = next_x;
end

s_k = NetParams.W{k} * Xs{k} + NetParams.b{k};
S{k} = s_k;

P = exp(s_k) ./ sum(exp(s_k)); %Kxn

end

% function [P, Xs, S, S_hat, vari, mu] = EvaluateClassifier(X, NetParams)
% n = size(X, 2);
% 
% [k, ~] = size(NetParams.W);
% Xs = cell(k, 1);
% S = cell(k, 1);
% 
% S_hat = cell(k-1, 1);
% vari = cell(k-1, 1);
% mu = cell(k-1, 1);
% 
% Xs{1} = X;
% 
% for i = 1:k-1
%     s_i = NetParams.W{i}*X + NetParams.b{i};
%    
%     if NetParams.use_bn
%         mu_s_i = mean(s_i, 2);
%         var_s_i = var(s_i, 0, 2) * (n-1) / n;
%         
%         s_hat_i = BatchNormalize(s_i, mu_s_i, var_s_i);
%         
%         % If using validation or test data
% %         if nargin > 4
% %             s_hat_i = BatchNormailize(s_i, varargin{1}{1}{1}{i}, varargin{1}{1}{2}{i});
% %         else
% %             s_hat_i = BatchNormailize(s_i, mu_s_i, var_s_i);
% %         end
%         
%         s_tilde_i = NetParams.gammas{i} .* s_hat_i + NetParams.betas{i};
%         X = max(0, s_tilde_i);
%         
%         S_hat{i} = s_hat_i;
%         mu{i} = mu_s_i;
% %         vari{i} = transpose(var_s_i);
%         vari{i} = var_s_i;
%     else
%         X = max(0, s_i);
%     end
%     S{i} = s_i;
%     Xs{i+1} = X;
% end
% 
% s_k = NetParams.W{k}*X + NetParams.b{k};
% S{k} = s_k;
% P = exp(s_k) ./ sum(exp(s_k)); %Kxn
% % P = zeros(K, n);
% % for col = 1:n
% %     P(:,col) = Softmax(s_k(:,col));
% % end
% % BNData.S = S;
% % BNData.S_hat = S_hat;
% % BNData.mu = mu;
% % BNData.vari = vari;
% end

% function [P, Xs, Ss, Shats, vs, mus] = EvaluateClassifier(X, NetParams)
% Ws = NetParams.W;
% bs = NetParams.b;
% gammas = NetParams.gammas;
% betas = NetParams.betas;
%     
% [k, ~] = size(Ws);
% 
% Xs = cell(k, 1);
% Ss = cell(k, 1);
% 
% Shats = cell(k-1, 1);
% vs = cell(k-1, 1);
% mus = cell(k-1, 1);
% 
% Xs{1} = X;
% 
% % alpha = 0.9;
% 
% for i=1:k-1
%     s = Ws{i} * Xs{i} + bs{i}; % mxn
%     
%     Ss{i} = s;
%     [shat, mu, v] = BatchNormalize(s);
%     Shats{i} = shat;
%     mus{i} = mu;
%     vs{i} = v;
%     
% %     mu_avgs{i} = alpha * mu_avgs{i} + (1 - alpha) * mu;
% %     v_avgs{i} = alpha * v_avgs{i} + (1 - alpha) * v;
%     
%     stilde = gammas{i} .* shat + betas{i};
%     
%     xnext = max(0, stilde); % mxn
%     Xs{i+1} = xnext;
% end
% 
% s = Ws{k} * Xs{k} + bs{k}; %Kxn
% Ss{k} = s;
% P = exp(s) ./ sum(exp(s)); %Kxn
% end

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


function Grads = ComputeGradsNumSlow(X, Y, NetParams, lambda, h)

Grads.W = cell(numel(NetParams.W), 1);
Grads.b = cell(numel(NetParams.b), 1);
if NetParams.use_bn
    Grads.gammas = cell(numel(NetParams.gammas), 1);
    Grads.betas = cell(numel(NetParams.betas), 1);
end

for j=1:length(NetParams.b)
    Grads.b{j} = zeros(size(NetParams.b{j}));
    NetTry = NetParams;
    for i=1:length(NetParams.b{j})
        b_try = NetParams.b;
        b_try{j}(i) = b_try{j}(i) - h;
        NetTry.b = b_try;
        c1 = ComputeCost(X, Y, NetTry, lambda);        
        
        b_try = NetParams.b;
        b_try{j}(i) = b_try{j}(i) + h;
        NetTry.b = b_try;        
        c2 = ComputeCost(X, Y, NetTry, lambda);
        
        Grads.b{j}(i) = (c2-c1) / (2*h);
    end
end

for j=1:length(NetParams.W)
    Grads.W{j} = zeros(size(NetParams.W{j}));
        NetTry = NetParams;
    for i=1:numel(NetParams.W{j})
        
        W_try = NetParams.W;
        W_try{j}(i) = W_try{j}(i) - h;
        NetTry.W = W_try;        
        c1 = ComputeCost(X, Y, NetTry, lambda);
    
        W_try = NetParams.W;
        W_try{j}(i) = W_try{j}(i) + h;
        NetTry.W = W_try;        
        c2 = ComputeCost(X, Y, NetTry, lambda);
    
        Grads.W{j}(i) = (c2-c1) / (2*h);
    end
end

if NetParams.use_bn
    for j=1:length(NetParams.gammas)
        Grads.gammas{j} = zeros(size(NetParams.gammas{j}));
        NetTry = NetParams;
        for i=1:numel(NetParams.gammas{j})
            
            gammas_try = NetParams.gammas;
            gammas_try{j}(i) = gammas_try{j}(i) - h;
            NetTry.gammas = gammas_try;        
            c1 = ComputeCost(X, Y, NetTry, lambda);
            
            gammas_try = NetParams.gammas;
            gammas_try{j}(i) = gammas_try{j}(i) + h;
            NetTry.gammas = gammas_try;        
            c2 = ComputeCost(X, Y, NetTry, lambda);
            
            Grads.gammas{j}(i) = (c2-c1) / (2*h);
        end
    end
    
    for j=1:length(NetParams.betas)
        Grads.betas{j} = zeros(size(NetParams.betas{j}));
        NetTry = NetParams;
        for i=1:numel(NetParams.betas{j})
            
            betas_try = NetParams.betas;
            betas_try{j}(i) = betas_try{j}(i) - h;
            NetTry.betas = betas_try;        
            c1 = ComputeCost(X, Y, NetTry, lambda);
            
            betas_try = NetParams.betas;
            betas_try{j}(i) = betas_try{j}(i) + h;
            NetTry.betas = betas_try;        
            c2 = ComputeCost(X, Y, NetTry, lambda);
            
            Grads.betas{j}(i) = (c2-c1) / (2*h);
        end
    end    
end
end