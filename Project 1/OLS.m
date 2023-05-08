function  [mu, Q, B, errors] = OLS(returns, factRet, lambda, K)
    
    % Use this function to perform an OLS regression. Note that you will 
    % not use lambda or K in this model (lambda is for LASSO, and K is for
    % BSS).
 
    % *************** WRITE YOUR CODE HERE ***************
    %----------------------------------------------------------------------
    [T,p] = size(factRet);
    
    X = [ones(size(factRet,1),1), factRet];
    B = (X.'*X)\X.'*returns;

    errors = returns - X*B;
    res_var = (vecnorm(errors).^2)/(T-p-1);
    D = diag(res_var);
    
    alpha = B(1,:).';
    V = B(2:end,:);
    F = cov(factRet);
    
    mu = alpha + V.'*(geomean(factRet+1)-1).';  % n x 1 vector of asset exp. returns
    Q = V.'*F*V + D;    % n x n asset covariance matrix
    %----------------------------------------------------------------------
    
end