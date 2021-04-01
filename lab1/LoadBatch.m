function [X, Y, y] = LoadBatch(filename)
% X = image pixel data, dxn, double or single between 0 and 1
% n = number of images, d image dimension

A = load(filename);
% disp('data')
[n, d] = size(A.data); % nxd
% disp('labels')
% size(A.labels) % nx1
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
