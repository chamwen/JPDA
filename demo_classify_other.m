% Joint Probability Distribution Adaptation (JPDA)
% Author: Wen Zhang
% Date: Dec. 8, 2019
% E-mail: wenz@hust.edu.cn

clc; clear all;

T = 10;
for i=1:2
    dataStr = {'COIL1_vs_COIL2','COIL2_vs_COIL1'};
    fname = dataStr{i};
    fprintf('%d: %s\n',i,fname);
    
    % Preprocess data using L2-norm
    data = strcat(char(fname));
    options.data = data;
    load(strcat('./data/',data));
    Xs = Xs*diag(sparse(1./sqrt(sum(Xs.^2))));
    Xt = Xt*diag(sparse(1./sqrt(sum(Xt.^2))));

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
