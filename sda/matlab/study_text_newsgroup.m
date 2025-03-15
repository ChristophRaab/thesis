clear all;
addpath(genpath('lib/'));
addpath(genpath('methods/'));
addpath(genpath('../matlab'));
database_path
    
%% 20 Newsgroup
options.ker = 'rbf';        % kernel: 'linear' | 'rbf' | 'lap'
options.gamma = 1.0;         % width of gaussian kernel
options.k = 100;              % JDA: subspaces bases
options.lambda = 1.0;        % JDA: regularization parameter
options.svmc = 10.0;         % SVM: complexity regularizer in LibSVM
options.g = 40;              % GFK: subspace dimension
options.tcaNv = 60;          % TCA: numbers of Vectors after reduction
options.subspace_dim_d = 5;  %SA: Subspace Dimensions
options.landmarks = 500;    %NTVM: Number of Landmarks
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
testsize = 1;
accuracy = [];
stdd = [];
time = [];
n_svds = [];
for name = {'comp_vs_sci','comp_vs_sci','comp_vs_talk','rec_vs_sci','rec_vs_talk','sci_vs_talk'}%
    acc_data = [];
    time_data = [];
    nsvds_data = [];
    n = 0;
    for j=[1:1:36]
        n = n+1;
            data = char(name);
            data = strcat(data, '_', num2str(j));
            load(strcat(newsgroup_path, data));
            fprintf('data=%s\n', data);
            Xs=zscore(Xs');
            Xt=zscore(Xt');

            soureIndx = crossvalind('Kfold', Ys, 2);
            targetIndx = crossvalind('Kfold', Yt,2);
            Xs = Xs(find(soureIndx==1),:);
            Ys = Ys(find(soureIndx==1),:);

            [acc_data,time_data,nsvds_data] = approaches(Xs,Ys,Xt,Yt,options,acc_data,time_data,nsvds_data);
    end
    s = size(time_data,2) / n; k = size(time_data,2) / s;
    accuracy = [accuracy; mean(reshape(acc_data, [s  k])')];
    n_svds = [n_svds; mean(reshape(nsvds_data, [s  k])')];
    time = [time; mean(reshape(time_data, [s  k])')];
    stdd = [stdd; std(reshape(acc_data, [s  k])')];
    save('study_text_newsgroup_BTLGA.mat','accuracy',"time","stdd","n_svds")
end
    save('study_text_newsgroup_BTLGA.mat','accuracy',"time","stdd","n_svds")