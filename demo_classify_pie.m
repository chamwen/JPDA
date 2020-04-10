% Joint Probability Distribution Adaptation (JPDA)
% Author: Wen Zhang
% Date: Dec. 8, 2019
% E-mail: wenz@hust.edu.cn

clc; clear all;

srcStr = {'PIE05','PIE05','PIE05','PIE05','PIE07','PIE07','PIE07','PIE07','PIE09','PIE09','PIE09','PIE09','PIE27','PIE27','PIE27','PIE27','PIE29','PIE29','PIE29','PIE29'};
tgtStr = {'PIE07','PIE09','PIE27','PIE29','PIE05','PIE09','PIE27','PIE29','PIE05','PIE07','PIE27','PIE29','PIE05','PIE07','PIE09','PIE29','PIE05','PIE07','PIE09','PIE27'};

T = 10;
for i = 1:20
    src = char(srcStr{i});
    tgt = char(tgtStr{i});
    fprintf('%d: %s_vs_%s\n',i,src,tgt);
    
    % Preprocess data using L2-norm
    load(strcat('./data/',src));
    Xs = fea'; Xs = Xs*diag(sparse(1./sqrt(sum(Xs.^2))));
    Ys = gnd;
    load(strcat('./data/',tgt));
    Xt = fea'; Xt = Xt*diag(sparse(1./sqrt(sum(Xt.^2))));
    Yt = gnd;

    % JPDA evaluation
    options.p = 100;
    options.lambda = 0.1;
    options.ker = 'primal';
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



