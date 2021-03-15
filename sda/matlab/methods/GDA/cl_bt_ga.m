function [model,K,m,n] = cl_bt_ga(Xs,Xt,Ys,options)
%PD_CL_NSO Summary of this function goes here
%   Detailed explanation goes here
addpath(genpath('NSO/'));
if exist('options', 'var')
    if ~isfield(options,"ker")
        options.ker = 'rbf';
    end
    if ~isfield(options, 'gamma')
        options.gamma = 1.0;
    end
    if ~isfield(options, "svmc")
        options.svmc = 10;
    end
else
    options.ker = 'rbf';
    options.gamma = 1.0;
    options.svmc = 10;
end

[Xs,Ys] = augmentation(Xs,size(Xt,1),Ys);
[~,ZS,~] = svd(Xs,'econ');
[U,S,V] = svd(Xt,'econ');
Xs = U*ZS*V';
m = size(Xs, 1);
n = size(Xt, 1);
K = kernel(options.ker, [Xs', Xt'], [],options.gamma);

model = svmtrain(full(Ys), [(1:m)', K(1:m, 1:m)], ['-c ', num2str(options.svmc), ' -t 4 -q 1']);
end

