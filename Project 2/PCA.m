function  [mu, Q, B, errors, D, V, F] = PCA(returns, factRet, lambda, K)
    
    % Use this function to perform PCA on the asset returns. Be sure to 
    % comment on your code to (briefly) explain your procedure.

    % Find the total number of time periods and assets, respectively.
    [T,n] = size(returns); 

    % *************** WRITE YOUR CODE HERE ***************
    %----------------------------------------------------------------------
    
    p = 3;  % number of principal components to select
    
    X = returns.';  % dim = [N, T] - transpose the input to match the dimensions throughout the calculation
    mu = mean(X,2);  % dim = [N, 1] - vector of asset expected returns
    Sigma = cov(X.');  % dim = [N, N] - covariance matrix of the returns
    [Gamma, Lambda] = eig(Sigma);  % dim = [N, N], [N, N] - matrix of eigenvectors as columns, matrix of corresponding eigenvalues as diagonal elements
    
    % flip Gamma and Lambda such that they are in decreasing order of eigenvalues
    Gamma = flip(Gamma,2);
    Lambda = flip(flip(Lambda,1),2);
    
    pcs = Gamma.' * (X - mu);  % dim = [N, T] - matrix of principal components (factors) in increasing order of eigenvalues
    
    V = Gamma(:,1:p).';  % dim = [P, N] - matrix of factor loadings corresponding to the first p principal components
    % take the first p eigenvectors since the columns of Gamma are in decreasing order of eigenvalues
    
    F = Lambda(1:p,1:p);  % dim = [P, P] - matrix of factor covariances
    % take the first p eigenvalues since the Lambda is in decreasing order of eigenvalues
    
    p_selected = pcs(1:p,:);  % dim = [P, T] - take the first p principal components (factors) since they are in decreasing order of eigenvalues
    errors = X - mu - (V.' * p_selected);  % dim = [N, T] - matrix of residuals between true X and estimated X from p PCA factors
    D = diag((vecnorm(errors,2,2).^2)/(T-p-1));  % dim = [N, N] - matrix of asset idiosyncratic variances as diagonal elements
    
    % set flag as 1 (default value) for skewness and kurtosis
    S = skewness(p_selected,1,2);  % dim = [P, 1] - vector of factor skewness
    K = kurtosis(p_selected,1,2);  % dim = [P, 1] - vector of factor kurtosis
    
    Q = V.' * F * V + D;  % dim = [N, N] - matrix of covariance of asset returns, derived from the p PCA factors

    % default outputs
    B = [];
    errors = [];

    % Sometimes quadprog shows a warning if the covariance matrix is not
    % perfectly symmetric.
    Q = (Q + Q')/2;
    %----------------------------------------------------------------------
    
end