clear all;

srcStr = {'Caltech10', 'Caltech10', 'Caltech10', 'amazon', 'amazon', 'amazon', 'webcam', 'webcam', 'webcam', 'dslr', 'dslr', 'dslr'};
tgtStr = {'amazon', 'webcam', 'dslr', 'Caltech10', 'webcam', 'dslr', 'Caltech10', 'amazon', 'dslr', 'Caltech10', 'amazon', 'webcam'};
testsize =10;
values = []
for i=1:testsize
    for iData = 1:12
            src = char(srcStr{iData});
            tgt = char(tgtStr{iData});
            data = strcat(src, '_vs_', tgt);
            fprintf('data=%s\n', data);
            load([oc_decaf_path src '_SURF_L10.mat']);
            Xs = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
            Ys = labels;
            Xs = zscore(Xs);


            load([oc_decaf_path tgt '_SURF_L10.mat']);
            Xt = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
            Yt = labels;
            Xt = zscore(Xt);

            [Xs,Ys] = augmentation(Xs,size(Xt,1),Ys);
%              diff = approx_original_norm(Xs,Xt);
            [diff,diff1,normB1,normL,normB2,normN] = calc_bound(Xs,Xt,3,125);
            values = [values;  [diff,diff1,normB1,normL,normB2,normN]];
    end
end
figure;
hold on;
plot(values(1:10,2));
plot(values(1:10,3));
plot(values(1:10,4));
plot(values(1:10,5));
hold off;

function [values]=approx_original_norm(Xs,Xt)
    
[U,S,V] = svd(Xs);
[L,E,R] = svd(Xt);
values = []
    for i=1:min(size(Xs))
        Sl = S(1:i,1:i);
        El = E(1:i,1:i);
        dl = sqrt(trace(Sl^2))^2+sqrt(trace(El^2))^2-2*trace(Sl'*El);
        d = sqrt(sum(diag(S).^2))^2+sqrt(sum(diag(E).^2))^2-2*trace(S'*E);
        values = [values abs(d-dl)];
    end
end


function [diff,diff1,normB1,normL,normB2,normN] = calc_bound(Xs,Xt,mu,landmarks)


[U,S,V] = svd(Xs);
[L,E,R] = svd(Xt);

Smu = diag(S(S>mu));
Emu = diag(E(E>mu));

Sl = Smu(1:landmarks+1,1:landmarks+1);
El = Emu(1:landmarks+1,1:landmarks+1);

Xsl = L(:,1:landmarks) * Smu(1:landmarks,1:landmarks+1);
Xtl = L(:,1:landmarks) * Emu(1:landmarks,1:landmarks+1);


normB1 = sqrt(trace((Sl - El).^2))^2; % sqrt(trace(Sl^2))^2+sqrt(trace(El^2))^2-2*trace(Sl'*El);
normB2 = sqrt(sum(diag((S - E).^2)))^2; % sqrt(sum(diag(S).^2))^2+sqrt(sum(diag(E).^2))^2-2*trace(S'*E);%norm(S,"Fro")^2 + norm(E,"Fro")^2-2*norm(S'*E,"Fro")
normL = norm(Xsl-Xtl,'Fro')^2;
normN = norm(Xs-Xt,'Fro')^2;
if normN < normB2
    "testW"
end
diff = normB1<normB2;
diff1 = normL < normB1;
end
