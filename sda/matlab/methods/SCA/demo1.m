clear all
clc

src = char("amazon");
load([oc_decaf_path src '_SURF_L10.mat']);
Xs = double(zscore(fts, 1));
Ys = labels;


tgt = char("webcam");
load([oc_decaf_path tgt '_SURF_L10.mat']);
Xt = double(zscore(fts, 1));
Yt = labels;


% soureIndx = crossvalind('Kfold', Ys, 2);
% Xss = Xs(find(soureIndx==1),:);
% Yss = Ys(find(soureIndx==1),:);
% 
% params.X_v = Xs(find(soureIndx==2),:);
% params.Y_v = Ys(find(soureIndx==2),:);
% 
% soureIndx = crossvalind('Kfold', Yss, 2);
% 
% 
% X_s_cell = cell(1,2);
% Y_s_cell = cell(1,2);
% X_s_cell{1} = Xss(find(soureIndx==1),:);
% X_s_cell{2} = Xss(find(soureIndx==2),:);
% 
% Y_s_cell{1} = Yss(find(soureIndx==1),:);
% Y_s_cell{2} = Yss(find(soureIndx==2),:);

params.verbose = false;
[test_accuracy, predicted_labels, Zs, Zt] = SCA(Xs,Ys, Xt, Yt, params);
