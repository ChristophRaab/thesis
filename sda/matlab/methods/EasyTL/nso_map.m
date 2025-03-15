function [Xs,Xt,Ys] = nso_map(Xs,Xt,Ys)
%Nystroem Subspace Override. Approximation of Subpsace via Nystroem
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
[Xt,Xs]=pd_ny_svd(Xt,Xs,Ys,options.landmarks);
Xs = zscore(Xs,1);
Xt = zscore(Xt,1);
end

