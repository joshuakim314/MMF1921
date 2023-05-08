function  [mu, Q, B, errors] = FF(returns, factRet, lambda, K)
    
    % Use this function to calibrate the Fama-French 3-factor model. Note 
    % that you will not use lambda or K in this model (lambda is for LASSO, 
    % and K is for BSS).
 
    % *************** WRITE YOUR CODE HERE ***************
    %----------------------------------------------------------------------
    factRet_FF = factRet(:,1:3);
    % mu = n x 1 vector of asset exp. returns
    % Q = n x n asset covariance matrix
    [mu, Q, B, errors] = OLS(returns, factRet_FF, lambda, K);
    %----------------------------------------------------------------------
    
end