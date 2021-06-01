clear all;
addpath dataset/;
format longg

%load data
book_fname = 'dataset/goblet_book.txt';
fid = fopen(book_fname,'r');
book_data = fscanf(fid,'%c');
fclose(fid);

book_chars = unique(book_data);
[~,K] = size(book_chars);

char_to_ind = containers.Map('KeyType','char','ValueType','int32');
ind_to_char = containers.Map('KeyType','int32','ValueType','char');

for i=1:K
    char_to_ind(book_chars(1,i)) = i;
    ind_to_char(i) = book_chars(1,i);
end

% hyper-parameters

m = 100; % dimensionality of hidden state
eta = 0.1; % learning rate
seq_length = 25; % length of input sequence
% rng(42);
RNN.b = zeros(m, 1);
RNN.c = zeros(K, 1);
sig = 0.01;
RNN.U = randn(m, K)*sig;
RNN.W = randn(m, m)*sig;
RNN.V = randn(K, m)*sig;

ada.b = zeros(size(RNN.b));
ada.c = zeros(size(RNN.c));
ada.U = zeros(size(RNN.U));
ada.W = zeros(size(RNN.W));
ada.V = zeros(size(RNN.V));


% % test gradients
% h0 = zeros(m, 1);
% 
% % make one hot encoding of input sequence
% X_chars = book_data(1:seq_length);
% Y_chars = book_data(2:seq_length + 1);
% 
% X = zeros(K, seq_length);
% Y = zeros(K, seq_length);
% 
% for i = 1 : seq_length
%     % X
%     onehot = zeros(K, 1);
%     index = char_to_ind(X_chars(i));
%     onehot(index) = 1;
%     X(:,i) = onehot;
%     % Y
%     onehot = zeros(K, 1);
%     index = char_to_ind(Y_chars(i));
%     onehot(index) = 1;
%     Y(:,i) = onehot;
% end
% 
% testGradients(RNN, X, Y, h0)

% training loop
name_loss = sprintf('results/loss.csv');
delete(name_loss);
fid_loss = fopen(name_loss, 'a+');
fprintf(fid_loss, 'step;smooth_loss;loss\n');

name_text = sprintf('results/synthesized.txt');
delete(name_text);
fid_text = fopen(name_text, 'a+');
% fprintf(fid_text, 'step;text\n');

update_step = 0;

all_losses = [];

lowest_loss = -1;
smooth_loss = -1;
best_RNN = RNN;
best_step = 0;

epochs = 10;
book_len = length(book_data);
for epoch = 1 : epochs
    e = 1;
    hprev = zeros(m, 1);
    while e < book_len-1
        % make one hot encoding of input sequence
        start = e;
        stop = min(e+seq_length-1, book_len-1);
        
        e = e + seq_length;
        
        X_chars = book_data(start:stop);
        Y_chars = book_data(start+1:stop+1);
        
        len = stop - start + 1;
        if len == 0
            break
        end
        
        X = zeros(K, len);
        Y = zeros(K, len);

        for i = 1 : len
            X(char_to_ind(X_chars(i)),i) = 1;
            Y(char_to_ind(Y_chars(i)),i) = 1;
        end
        
        % synthesize before update
        % synthesize every 10 000th step
        if mod(update_step, 10000) == 0
            fprintf('step: %d, Smooth Loss: %f\n', update_step, smooth_loss);
            onehot = synthesize(RNN, hprev, X(:, 1), 200);
            [K, n] = size(onehot);
            text = [];
            for j = 1 : n
                index = find(onehot(:,j)==1);
                text = [text ind_to_char(index)];
            end
            fprintf(fid_text, 'Step: %d, Smooth Loss: %f\n;%s\n\n', update_step, smooth_loss, text);
            disp(text);
        end
        
        
        % forward and backward pass
        [loss, a, h, p] = forwardPass(RNN, X, Y, hprev);
        [RNN, ada] = backwardPass(RNN, ada, X, Y, a, h, p, eta);
        hprev = h(:,end);
        update_step = update_step + 1;
        
        if update_step == 1
            % init loss
            smooth_loss = loss;
            lowest_loss = loss;
        else
            smooth_loss = 0.999 * smooth_loss + 0.001 * loss;
        end

        
        % store best RNN
        if smooth_loss < lowest_loss
            lowest_loss = smooth_loss;
            best_RNN = RNN;
            best_step = update_step;
        end
       
        % print smooth loss every 100th update step
        if mod(update_step, 100) == 1
            fprintf('epoch: %d, update_step: %d, smooth_loss: %f, loss: %f\n',epoch, update_step, smooth_loss, loss);
            all_losses = [all_losses smooth_loss];
            fprintf(fid_loss, '%d; %f;%f\n', update_step, smooth_loss, loss);
        end
        
        
         
    end
end

fprintf('training done\n');
fprintf('synthesizing text with best model. Step: %d, Lowest loss: %f\n', best_step, lowest_loss);

% synthesize text with best model
x0 = zeros(K, 1);
x0(char_to_ind('.')) = 1; % input
h0 = zeros(m, 1); % initial hidden state

Y = synthesize(RNN, h0, x0, 1000);
[K, n] = size(Y);
text = [];
for j = 1 : n
    index = find(Y(:,j)==1);
    text = [text ind_to_char(index)];
end
fprintf(fid_text, 'Best Model (step=%d,smooth_loss=%f)\n;%s\n\n', best_step, lowest_loss, text);
disp(text);

% Plot smooth_loss
n = length(all_losses);
figure(1);
x = 1:100:100*n;
plot(x,all_losses,'b');
title('Smooth loss')
xlabel('update step') 
ylabel('smooth loss') 
saveas(gcf,'results/smooth_loss.png');

function [RNN, ada] = backwardPass(RNN, ada, X, Y, a, h, p, eta)
grads = computeGradients(RNN, X, Y, a, h, p);
% clip gradients
for f = fieldnames(grads)'
    grads.(f{1}) = max(min(grads.(f{1}), 5), -5);
end

for f = fieldnames(RNN)'
    ada.(f{1}) = ada.(f{1}) + grads.(f{1}).^2;
    RNN.(f{1}) = RNN.(f{1}) - eta * (grads.(f{1}) ./ sqrt(ada.(f{1}) + eps));
end
end

function [loss, a, h, p] = forwardPass(RNN, X, Y, h0)
[K, n] = size(X);
m = length(h0);

p = zeros(K, n);    % p_1, p_2, .., p_n
h = zeros(m, n+1);  % h_0, h_1, .., h_n
a = zeros(m, n);    % a_1, a_2, .., a_n
h(:, 1) = h0;
loss = 0;

for t = 1:n
   a(:, t) = RNN.W * h(:, t) + RNN.U * X(:, t) + RNN.b;
   h(:, t+1) = tanh(a(:, t));
   ot = RNN.V * h(:, t+1) + RNN.c;
   p(:,t) = softmax(ot);
    
   loss = loss - log(Y(:,t)' * p(:,t));
end
end

function testGradients(RNN, X, Y, h0)
[loss, a, h, p] = forwardPass(RNN, X, Y, h0);

% my grads
grads = computeGradients(RNN, X, Y, a, h, p);

% numerical grads
num_grads = ComputeGradsNum(X, Y, RNN, 1e-4);

for f = fieldnames(RNN)'
    num = num_grads.(f{1});
    ana = grads.(f{1});

    rel_err = abs(num - ana)./abs(num + ana);
    max_rel_err = max(rel_err(:));
    
    fprintf('grads.%s: max relative error: %d\n', f{1}, max_rel_err);
end
end

function grads = computeGradients(RNN, X, Y, a, h, p)
[m, ~] = size(h);
[K, n] = size(X);

grad_o = (p - Y)';
grads.c = (sum(grad_o))';
grads.V = grad_o' * h(:,2 : end)';

grad_h = grad_o(n, :) * RNN.V;

grad_a = zeros(n, m);
grad_a(n, :) = grad_h * diag(1 - (tanh(a(:, n))).^2);

for t = n-1 : -1 : 1
   grad_h = grad_o(t, :) * RNN.V + grad_a(t+1, :) * RNN.W;
   grad_a(t, :) = grad_h * diag(1 - (tanh(a(:, t))).^2);
end

grads.b = (sum(grad_a))';

grads.U = grad_a' * X';
grads.W = grad_a' * h(:,1 : end-1)';

end

function res = softmax(s)
res = exp(s) ./ sum(exp(s)); 
end

function Y = synthesize(RNN, h0, x0, n)
% RNN: network parameter
% h0: hidden state at time 0
% x0: first (dummy) input vector (e.g. fullstop)
% n: length of output sequence
% Y: one-hot encoding of each sampled character (Kxn)

K = length(x0);

Y = zeros(K, n);

xt = x0;
ht = h0;

for i=1:n
    at = RNN.W * ht + RNN.U * xt + RNN.b;
    ht = tanh(at);
    ot = RNN.V * ht + RNN.c;
    pt = softmax(ot);
    
    % calculate xnext
    cp = cumsum(pt);
    a = rand;
    ixs = find(cp-a >0);
    ii = ixs(1);
    
    % make one hot encoding
    xt = zeros(K, 1);
    xt(ii) = 1;
    
    % add to Y
    Y(:, i) = xt;
end

end



