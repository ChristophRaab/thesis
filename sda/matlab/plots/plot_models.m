%% Skript to plot the model of stvm and tkl-svm
% Load orgs vs people dataset
addpath(genpath('../../matlab'));
database_path
%% Reuters Dataset
options.ker = 'rbf';        % kernel: 'linear' | 'rbf' | 'lap'
options.gamma = 1.0;         % width of gaussian kernel
options.k = 100;              % JDA: subspaces bases
options.lambda = 1.0;        % JDA: regularization parameter
options.svmc = 10.0;         % SVM: complexity regularizer in LibSVM
options.g = 40;              % GFK: subspace dimension
options.tcaNv = 60;          % TCA: numbers of Vectors after reduction
options.subspace_dim_d = 10;  %SA: Subspace Dimensions
options.ntvm_ker = 'rbf';     %NTVM: Kernel Type
options.ker = 'rbf';      % TKL: kernel: | 'rbf' |'srbf | 'lap'
options.eta = 2.0;           % TKL: eigenspectrum damping factor
options.gamma = 1;         % TKL: width of gaussian kernel
options.landmarks = 500;

load(strcat("../",reuters_path, 'org_vs_people_1'));
Xs =full(Xs);
Xt = full(Xt);
Xs = zscore(Xs,1);
Xt = zscore(Xt,1);
Xs = Xs';Xt = Xt';

% Find subspace embedding with minimal loss
i = 100; j = 30;
[Xs,Ys] = augmentation(Xs,size(Xt,1),Ys);
[Xt,Xs]=c_ny_svd(Xt,Xs,Ys,options.landmarks);

m = size(Xs, 1);
n = size(Xt, 1);
K = kernel(options.ker, [Xs', Xt'], [],options.gamma);
model = svmtrain(full(Ys), [(1:m)', K(1:m, 1:m)], ['-c ', num2str(options.svmc), ' -t 4 -q 1']);
X = [Xs;Xt];
[X,loss]  = tsne(full(X),'Exaggeration',2,"Distance","cosine",'NumDimensions',2,'Perplexity',40);
Xs = X(1:size(Xs,1),:);
Xt = X(size(Xs,1)+1:end,:);

plotf(Xs,Ys,Xt,Yt,'#67001F','#053061',"reuters__cnso_class_labels");
plotf(Xs,-1*ones(size(Ys,1),1),Xt,1*ones(size(Yt,1),1),'#4276B0','#AFD8E4',"reuters__cnso_domain_labels");
plotm(Xs,Ys,Xt,Yt,'#67001F','#053061',"reuters_cnso_model",model);
%==========================================================================
load(strcat("../",reuters_path, 'org_vs_people_1'));
Xs =full(Xs);
Xt = full(Xt);
Xs = zscore(Xs,1);
Xt = zscore(Xt,1);
Xs = Xs';Xt = Xt';


%Train and predict
% K = TKL(Xs', Xt', options);
% modeltkm = svmtrain(full(Ys), [(1:m)', K(1:m, 1:m)], ['-s 0 -c ', num2str(10), ' -t 4 -q 1']);
% [labels, acc,scores] = svmpredict(full(Yt), [(1:n)', K(m+1:end, 1:m)], modeltkm);

% Find subspace embedding with minimal loss
i = 100; j = 30;
[Xs,Ys] = augmentation(Xs,size(Xt,1),Ys);
[U,S,V] = svd(Xs);
[L,E,R] = svd(Xt);

landmarks =min(min(size(Xs,1),size(Xt,2)),options.landmarks);
Xs = L(:,1:landmarks) * S(1:landmarks,1:landmarks);
Xt = L(:,1:landmarks) * E(1:landmarks,1:landmarks);

m = size(Xs, 1);
n = size(Xt, 1);
K = kernel(options.ker, [Xs', Xt'], [],options.gamma);
model = svmtrain(full(Ys), [(1:m)', K(1:m, 1:m)], ['-c ', num2str(options.svmc), ' -t 4 -q 1']);
X = [Xs;Xt];
[X,loss]  = tsne(full(X),'Exaggeration',2,"Distance","cosine",'NumDimensions',2,'Perplexity',40);
Xs = X(1:size(Xs,1),:);
Xt = X(size(Xs,1)+1:end,:);

plotf(Xs,Ys,Xt,Yt,'#67001F','#053061',"reuters_so_class_labels");
plotf(Xs,-1*ones(size(Ys,1),1),Xt,1*ones(size(Yt,1),1),'#4276B0','#AFD8E4',"reuters_so_domain_labels");
plotm(Xs,Ys,Xt,Yt,'#67001F','#053061',"reuters_so_model",model);

%==========================================================================

load(strcat("../",reuters_path, 'org_vs_people_1'));
Xs =full(Xs);
Xt = full(Xt);
Xs = zscore(Xs,1);
Xt = zscore(Xt,1);
Xs = Xs';Xt = Xt';


%Train and predict
% K = TKL(Xs', Xt', options);
% modeltkm = svmtrain(full(Ys), [(1:m)', K(1:m, 1:m)], ['-s 0 -c ', num2str(10), ' -t 4 -q 1']);
% [labels, acc,scores] = svmpredict(full(Yt), [(1:n)', K(m+1:end, 1:m)], modeltkm);

% Find subspace embedding with minimal loss
i = 100; j = 30;
[Xs,Ys] = augmentation(Xs,size(Xt,1),Ys);
[Xt,Xs]=ny_svd(Xt,Xs,options.landmarks);

m = size(Xs, 1);
n = size(Xt, 1);
K = kernel(options.ker, [Xs', Xt'], [],options.gamma);
model = svmtrain(full(Ys), [(1:m)', K(1:m, 1:m)], ['-c ', num2str(options.svmc), ' -t 4 -q 1']);
X = [Xs;Xt];
[X,loss]  = tsne(full(X),'Exaggeration',2,"Distance","cosine",'NumDimensions',2,'Perplexity',40);
Xs = X(1:size(Xs,1),:);
Xt = X(size(Xs,1)+1:end,:);

plotf(Xs,Ys,Xt,Yt,'#67001F','#053061',"reuters_nso_class_labels");
plotf(Xs,-1*ones(size(Ys,1),1),Xt,1*ones(size(Yt,1),1),'#4276B0','#AFD8E4',"reuters_nso_domain_labels");
plotm(Xs,Ys,Xt,Yt,'#67001F','#053061',"reuters_nso_model",model);

%============================================================================


%==========================================================================

load(strcat("../",reuters_path, 'org_vs_people_1'));
Xs =full(Xs);
Xt = full(Xt);
Xs = zscore(Xs,1);
Xt = zscore(Xt,1);
Xs = Xs';Xt = Xt';


%Train and predict
% K = TKL(Xs', Xt', options);
% modeltkm = svmtrain(full(Ys), [(1:m)', K(1:m, 1:m)], ['-s 0 -c ', num2str(10), ' -t 4 -q 1']);
% [labels, acc,scores] = svmpredict(full(Yt), [(1:n)', K(m+1:end, 1:m)], modeltkm);

% Find subspace embedding with minimal loss
i = 100; j = 30;

C = unique(Ys,'stable');
sizeC = size(C,1);
    %    Align of feature space examples
if size(Xs,1) > size(Xt,1)
            
            indxYs1 = find(Ys==1);
            indxYs2 = find(Ys==-1);
            
            s1 = size(indxYs1,1);
            s2 = size(indxYs2,1);
            
            if (s1 >= round(size(Xt,1)/2)) &&(s2 >= round(size(Xt,1)/2))
                s1 = round(size(Xt,1)/2); s2 = round(size(Xt,1)/2);
            elseif s1 < round(size(Xt,1)/2)
                labelDiff = abs(size(Xt,1)/2-s1);
                s2 =s1+2*labelDiff;
            elseif s2 < round(size(Xt,1)/2)
                labelDiff = abs(size(Xt,1)/2-s2);
                s1 =s2+2*labelDiff;
            end
            
            
            Xs1 = Xs(indxYs1,:);
            C1 = cov(Xs1');
            [v,e] = eigs(C1,s1);
            Xs1 = (Xs1' * v)';
            
            Xs2 = Xs(indxYs2,:);
            C2 = cov(Xs2');
            [v,e] = eigs(C2,s2);
            Xs2 = (Xs2' * v)';
            
            Xs = [Xs1;Xs2];
            Ys = [ones(size(Xs1,1),1);ones(size(Xs2,1),1)*-1];
            
            if(size(Xs,1) > size(Xt,1))
                Xs = Xs(1:size(Xt,1),:);
                Ys = Ys(1:size(Xs,1),:);
            end
end
if size(Xs,1) <  size(Xt,1)
        data = [];
        label = [];
        diff = size(Xt,1) - size(Xs,1);
        sampleSize = floor(diff / sizeC);
        for c = C'
            idxs = find(Ys == c);
            classData= Xs(idxs,:);
            m = mean(classData); sd = std(classData);
            augmentationData = mvnrnd(m,sd,sampleSize);
            data = [data; classData;augmentationData];
            label = [label;ones(size(classData,1),1)*c;ones(sampleSize,1)*c];
        end
        sampleSize = mod(diff,sizeC);
        c = C(end);
        idxs = find(Ys == c);
        classData= Xs(idxs,:);
        m = mean(classData); sd = std(classData);
        augmentationData = mvnrnd(m,sd,sampleSize);
        data = [data;augmentationData];
        label = [label;ones(size(augmentationData,1),1)*c];
        Xs = data;Ys = label;
end
[~,ZS,~] = svd(Xs,'econ');
[U,S,V] = svd(Xt,'econ');
Xs = U*ZS*V';

m = size(Xs, 1);
n = size(Xt, 1);
K = kernel(options.ker, [Xs', Xt'], [],options.gamma);
model = svmtrain(full(Ys), [(1:m)', K(1:m, 1:m)], ['-c ', num2str(options.svmc), ' -t 4 -q 1']);
X = [Xs;Xt];
[X,loss]  = tsne(full(X),'Exaggeration',2,"Distance","cosine",'NumDimensions',2,'Perplexity',40);
Xs = X(1:size(Xs,1),:);
Xt = X(size(Xs,1)+1:end,:);

plotf(Xs,Ys,Xt,Yt,'#67001F','#053061',"reuters_gtlga_class_labels");
plotf(Xs,-1*ones(size(Ys,1),1),Xt,1*ones(size(Yt,1),1),'#4276B0','#AFD8E4',"reuters_gtlga_domain_labels");
plotm(Xs,Ys,Xt,Yt,'#67001F','#053061',"reuters_gtlga_model",model);

%==========================================================================
load(strcat("../",reuters_path, 'org_vs_people_1'));
Xs =full(Xs);
Xt = full(Xt);
Xs = zscore(Xs,1);
Xt = zscore(Xt,1);
Xs = Xs';Xt = Xt';


%Train and predict


% Find subspace embedding with minimal loss
i = 100; j = 30;
K = TKL(Xs', Xt', options);
m = size(Xs, 1);
n = size(Xt, 1);
model = svmtrain(full(Ys), [(1:m)', K(1:m, 1:m)], ['-c ', num2str(options.svmc), ' -t 4 -q 1']);
X = [Xs;Xt];
[X,loss]  = tsne(full(X),'Exaggeration',2,"Distance","cosine",'NumDimensions',2,'Perplexity',40);
Xs = X(1:size(Xs,1),:);
Xt = X(size(Xs,1)+1:end,:);

plotf(Xs,Ys,Xt,Yt,'#67001F','#053061',"reuters_tkl_class_labels");
plotf(Xs,-1*ones(size(Ys,1),1),Xt,1*ones(size(Yt,1),1),'#4276B0','#AFD8E4',"reuters_tkl_domain_labels");
plotm(Xs,Ys,Xt,Yt,'#67001F','#053061',"reuters_tkl_model",model)


function [] = plotf(Xs,Ys,Xt,Yt,c1,c2,name)
X = [Xs;Xt];
Y = [Ys;Yt];
figure
hold on
plot(X(Y==-1,1),X(Y==-1,2),'.','MarkerFaceColor',c1,'MarkerEdgeColor',c1);
plot(X(Y==1,1),X(Y==1,2),'x','MarkerFaceColor',c2,'MarkerEdgeColor',c2);
print(name,"-depsc","-r400")
hold off
end

function [] = plotm(Xs,Ys,Xt,Yt,c1,c2,name,model)
X = [Xs;Xt];
Y = [Ys;Yt];
figure
hold on
plot(X(Y==-1,1),X(Y==-1,2),'.','MarkerFaceColor',c1,'MarkerEdgeColor',c1);
plot(X(Y==1,1),X(Y==1,2),'x','MarkerFaceColor',c2,'MarkerEdgeColor',c2);
 plot(Xs(model.SVs,1),Xs(model.SVs,2),'ko','LineWidth',1);
print(name,"-depsc","-r400")
hold off
end

% figure
% hold on
% plot(X(Y==-1,1),X(Y==-1,2),'.','MarkerFaceColor','#67001F','MarkerEdgeColor','#67001F');
% plot(X(Y==1,1),X(Y==1,2),'x','MarkerFaceColor','#053061','MarkerEdgeColor','#053061');
% plot(Xs(modeltkm.SVs,1),Xs(modeltkm.SVs,2),'ko','LineWidth',1);
% title(strcat('TKL-SVM, Number Vectors',{' '},num2str(modeltkm.totalSV),{', '},'Error',{' '},num2str(100-acc(1))));
% xlabel(strcat('Kullback-Leibler divergence original vs reduced space',{' '},num2str(loss)));
% print("TKLSVM_Model","-depsc","-r400")
% hold off
