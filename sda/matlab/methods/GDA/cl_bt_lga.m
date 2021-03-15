function [model,K,m,n] = cl_bt_lga(Xs,Xt,Ys,options)
%PD_CL_NSO Summary of this function goes here
%   Detailed explanation goes here
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
end

