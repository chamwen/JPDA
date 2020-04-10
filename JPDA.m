function [Zs,Zt] = JPDA(Xs,Xt,Ys,Yt0,options)
    % Joint Probability Distribution Adaptation (JPDA)
    % Author: Wen Zhang
    % Date: Dec. 8, 2019
    % E-mail: wenz@hust.edu.cn

    % Input:
    %  Xs and Xt: d*n, source and target features
    %  Ys and Yt0: y*1, source labels and target pseudo-labels
	%  options: optional parameters
	%     options.p = 100;
	%     options.lambda = 0.1;
	%     options.ker = 'primal';
	%     options.mu = 0.001~0.2;
	%     options.gamma = 1.0;
    
    % Output:
	%  Embeddings Zs,Zt;
	
    % Load algorithm options
    p = options.p;
    lambda = options.lambda;
    ker = options.ker;
    mu = options.mu;
    gamma = options.gamma;

    % Set variables
    X = [Xs,Xt];
    X = X*diag(sparse(1./sqrt(sum(X.^2))));
    [m,n] = size(X);
    ns = size(Xs,2);
    nt = size(Xt,2);
    C = length(unique(Ys));

    % Joint probability MMD by onehot encoding
    Ns=1/ns*onehot(Ys,unique(Ys)); Nt=zeros(nt,C);
    if ~isempty(Yt0); Nt=1/nt*onehot(Yt0,unique(Ys)); end
	
    % For transferability
    Rmin=[Ns*Ns',-Ns*Nt';-Nt*Ns',Nt*Nt'];
    Rmin = Rmin / norm(Rmin,'fro');
    
    % For discriminability
    Ms=[]; Mt=[];
    for i=1:C
        Ms=[Ms,repmat(Ns(:,i),1,C-1)];
        idx=1:C; idx(i)=[];
        Mt=[Mt,Nt(:,idx)];
    end
    Rmax=[Ms*Ms',-Ms*Mt';-Mt*Ms',Mt*Mt'];
    Rmax = Rmax / norm(Rmax,'fro');
    
    % Construct centering matrix
    H = eye(n)-1/(n)*ones(n,n);

    % Joint Probability Distribution Adaptation: JPDA
    if strcmp(ker,'primal')
        [A,~] = eigs(X*(Rmin-mu*Rmax)*X'+lambda*eye(m),X*H*X',p,'SM');
        Z = A'*X;
    else
        K = kernel(ker,X,[],gamma);
        [A,~] = eigs(K*(Rmin-mu*Rmax)*K'+lambda*eye(n),K*H*K',p,'SM');
        Z = A'*K;
    end

    % Embeddings
    Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
    Zs = Z(:,1:size(Xs,2));
    Zt = Z(:,size(Xs,2)+1:end);
end

function y_onehot=onehot(y,class)
    % Encode label to onehot form
    % Input:
    % y: label vector, N*1
    % Output:
    % y_onehot: onehot label matrix, N*C

    nc=length(class);
    y_onehot=zeros(length(y), nc);
    for i=1:length(y)
        y_onehot(i, class==y(i))=1;
    end
end

function K = kernel(ker,X,X2,gamma)
    % With Fast Computation of the RBF kernel matrix
    % Inputs:
    %       ker:    'linear','rbf','sam'
    %       X:      data matrix (features * samples)
    %       gamma:  bandwidth of the RBF/SAM kernel
    % Output:
    %       K: kernel matrix
    %
    % Gustavo Camps-Valls
    % 2006(c)
    % Jordi (jordi@uv.es), 2007

    switch ker
        case 'linear'

            if isempty(X2)
                K = X'*X;
            else
                K = X'*X2;
            end

        case 'rbf'

            n1sq = sum(X.^2,1);
            n1 = size(X,2);

            if isempty(X2)
                D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*(X'*X);
            else
                n2sq = sum(X2.^2,1);
                n2 = size(X2,2);
                D = (ones(n2,1)*n1sq)' + ones(n1,1)*n2sq -2*X'*X2;
            end
            K = exp(-gamma*D); 

        case 'sam'

            if isempty(X2)
                D = X'*X;
            else
                D = X'*X2;
            end
            K = exp(-gamma*acos(D).^2);

        otherwise
            error(['Unsupported kernel ' ker])
    end
end
