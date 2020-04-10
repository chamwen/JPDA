% Joint Probability Distribution Adaptation (JPDA)
% Author: Wen Zhang
% Date: Dec. 8, 2019
% E-mail: wenz@hust.edu.cn

clc; clear all;

srcStr = {'caltech','caltech','caltech','amazon','amazon','amazon','webcam','webcam','webcam','dslr','dslr','dslr'};
tgtStr = {'amazon','webcam','dslr','caltech','webcam','dslr','caltech','amazon','dslr','caltech','amazon','webcam'};

T = 10;
for i = 1:12
    src = char(srcStr{i});
    tgt = char(tgtStr{i});
    fprintf('%d: %s_vs_%s\n',i,src,tgt);

    % Preprocess surf features using z-score
    load(['./data/Office/' src '_SURF_L10.mat']);
    fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
    Xs = zscore(fts,1); Xs = Xs';
    Ys = labels;
    load(['./data/Office/' tgt '_SURF_L10.mat']);
    fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
    Xt = zscore(fts,1); Xt = Xt';
    Yt = labels;

    % JPDA evaluation
    options.p = 100;
    options.lambda = 1.0;
    options.ker = 'linear';
    options.mu = 0.1;
    options.gamma = 1.0;
    Cls = []; Acc = [];
    for t = 1:T
        [Zs,Zt] = JPDA(Xs,Xt,Ys,Cls,options);
        mdl = fitcknn(Zs',Ys);
        Cls = predict(mdl,Zt');
        acc = length(find(Cls==Yt))/length(Yt);
        Acc = [Acc;acc];
    end
    fprintf('JPDA=%0.4f\n\n',Acc(end));
end
