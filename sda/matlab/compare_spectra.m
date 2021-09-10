close all;
clear all; 

src = char("amazon");
tgt = char("webcam");
data = strcat(src, '_vs_', tgt);
fprintf('data=%s\n', data);
load([oc_decaf_path src '_SURF_L10.mat']);
fts = fts ./ repmat(sum(fts, 2), 1, size(fts, 2));
Xs = zscore(fts, 1);
Ys = labels;

figure;
plot(svd(Xs));

load([oc_decaf_path tgt '_SURF_L10.mat']);
fts = fts ./ repmat(sum(fts, 2), 1, size(fts,2));
Xt = zscore(fts, 1);
Yt = labels;

figure;
plot(svd(Xt));
% [Xs,Ys] = augmentation(Xs,Xt,Ys);
[Xt,Xs]=pd_ny_svd(Xt,Xs,Ys,125);
figure;
plot(svd(Xs))
figure;
plot(svd(Xt))

load([oc_decaf_path src '_decaf_fc8.mat']);
Xs = double(zscore(X, 1));
Ys = double(Y)';
figure;
plot(svd(Xs));

load([oc_decaf_path tgt '_decaf_fc8.mat']);
Xt = double(zscore(X, 1));
Yt = double(Y)';

figure;
plot(svd(Xt));
% 
% [Xs,Ys] = augmentation(Xs,Xt,Ys);
[Xt,Xs]=pd_ny_svd(Xt,Xs,Ys,125);

figure;
plot(svd(Xs))
figure;
plot(svd(Xt))
