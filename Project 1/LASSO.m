function  [mu, Q, B, errors] = LASSO(returns, factRet, lambda, K)
    
    % Use this function for the LASSO model. Note that you will not use K 
    % in this model (K is for BSS).
    %
    % You should use an optimizer to solve this problem. Be sure to comment 
    % on your code to (briefly) explain your procedure.
    
    % *************** WRITE YOUR CODE HERE ***************
    %----------------------------------------------------------------------
    error_tol = 1e-15;
    
    [T,p] = size(factRet);
    N = size(returns,2);
    B_abs = zeros(2*(p+1),N);
    
    X = [ones(size(factRet,1),1), factRet];
    Q = [X.'*X   -X.'*X;
         -X.'*X  X.'*X];
    
    lb = zeros(1,2*(p+1));

    % it might be useful to increase the tolerance of 'quadprog'
    options = optimoptions('quadprog', 'TolFun', error_tol);
    
    for i = 1:N
        c = [-2*(X.'*returns(:,i)); 2*(X.'*returns(:,i))] + lambda*ones(2*(p+1),1); 
        % solve this problem using 'quadprog'
        B_abs(:,i) = quadprog( 2 * Q, c, [], [], [], [], lb, [], [], options );
    end 
    
    B_pos = B_abs(1:p+1,:);
    B_neg = B_abs(p+2:end,:);
    B = B_pos - B_neg;
    
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