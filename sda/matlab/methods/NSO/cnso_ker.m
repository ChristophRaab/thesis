function [model,Kt] = cnso_ker(Xs,Xt,Ys,options)
%KER_CL_NSO: Subspace Override via kernelized classwise approximation.

if exist('options', 'var')
    if ~isfield(options,"ker")
        options.ker = 'rbf';
    end
    if ~isfield(options, 'gamma')
        options.gamma = 1.0;
    end
    if ~isfield(options,'landmarks')
        options.landmarks = 500;
    end
    if ~isfield(options, "svmc")
        options.svmc = 10;
    end
else
    options.ker = 'rbf';
    options.gamma = 1.0;
    options.landmarks = 500;
    options.svmc = 10;
end
[Xs,Ys] = augmentation(Xs,size(Xt,1),Ys);
Ks = kernel(options.ker, [Xs'], [],options.gamma);
Kt = kernel(options.ker, [Xt'], [],options.gamma);
[Xt,Xs]=c_ny_svd(Kt,Ks,Ys,options.landmarks);
Ks = zscore(Ks,1);
Kt = zscore(Kt,1);
m = size(Ks, 1);
n = size(Kt, 1);
% K = kernel(options.ker, [Xs', Xt'], [],options.gamma);

model = svmtrain(full(Ys),  [(1:size(Ks,1))', Ks], ['-c ', num2str(options.svmc), ' -t 4 -q 1']);

end

