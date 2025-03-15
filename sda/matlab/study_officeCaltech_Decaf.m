clear all;

addpath(genpath('../matlab'));
database_path


%% OFFICE vs CALLTECH-256 Dataset
options.ker = 'rbf';        % kernel: 'linear' | 'rbf' | 'lap'
options.gamma = 1.0;         % width of gaussian kernel
options.k = 100;              % JDA: subspaces bases
options.lambda = 1.0;        % JDA: regularization parameter
options.svmc = 10.0;         % SVM: complexity regularizer in LibSVM
options.g = 40;              % GFK: subspace dimension
options.tcaNv = 60;          % TCA: numbers of Vectors after reduction
options.subspace_dim_d = 10;  %SA: Subspace Dimensions
options.landmarks = 125;     %NSO: Number of Landmarks
% JGSA
options.k = 30;             % subspace base dimension
options.T = 10;             % #iterations, default=10
options.alpha= 1;           % the parameter for subspace divergence ||A-B||
options.mu = 1;             % the parameter for target variance
options.beta = 0.1;         % the parameter for P and Q (source discriminaiton)
options.gamma = 2;          % the parameter for kernel

% MEDA: Parameters
options.d = 20;
options.rho = 1.0;
options.p = 10;
options.lambda = 10.0;
options.eta = 0.1;
options.T = 10;
accuracy = [];
stdd = [];
time = [];
n_svds = [];
testsize = 5;
srcStr = {'caltech', 'caltech', 'caltech', 'amazon', 'amazon', 'amazon', 'webcam', 'webcam', 'webcam', 'dslr', 'dslr', 'dslr'};
tgtStr = {'amazon', 'webcam', 'dslr', 'caltech', 'webcam', 'dslr', 'caltech', 'amazon', 'dslr', 'caltech', 'amazon', 'webcam'};
for iData = 1:12
    acc_data = [];
    time_data = [];
    nsvds_data = [];
    for i=1:testsize
        src = char(srcStr{iData});
        tgt = char(tgtStr{iData});
        data = strcat(src, '_vs_', tgt);
        fprintf('data=%s\n', data);
        load([oc_decaf_path src '_decaf.mat']);
        feas = feas ./ repmat(sum(feas, 2), 1, size(feas, 2));
        Xs = zscore(feas, 1);
%         Xs = normr(Xs);
        Ys = labels;

        load([oc_decaf_path tgt '_decaf.mat']);
        feas = feas ./ repmat(sum(feas, 2), 1, size(feas,2));
        Xt = zscore(feas, 1);
        Yt = labels;
%         Xt = normr(Xt);
        [acc_data,time_data,nsvds_data] = approaches(Xs,Ys,Xt,Yt,options,acc_data,time_data,nsvds_data);

    end
    acc_data = reshape(acc_data, [size(acc_data,2) / testsize, testsize])';
    time_data = reshape(time_data, [size(time_data,2) / testsize, testsize])';
    nsvds_data = reshape(nsvds_data, [size(nsvds_data,2) / testsize,testsize])';
    accuracy = [accuracy; mean(acc_data)];
    time = [time; mean(time_data)];
    stdd = [stdd; std(acc_data)];
    n_svds = [n_svds; mean(nsvds_data)];
    save('study_offficeCaltech_Decaf_SVM_SA.mat','accuracy',"stdd","time","n_svds")
end
save('study_offficeCaltech_Decaf_SVM_SA.mat','accuracy',"stdd","time","n_svds")