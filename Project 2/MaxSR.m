function  x = MaxSR(mu, Q)
    
    % Use this function to construct your maximum Sharpe ratio portfolio.
    %
    % You can use quadprog to solve this problem. Just be sure to comment 
    % on your code to (briefly) explain your steps. 
    
    % Find the total number of assets
    n = size(mu, 1); 
    
    % *************** WRITE YOUR CODE HERE ***************
    %----------------------------------------------------------------------
    
    H = [2*Q  zeros(n,1); zeros(1,n)  0];  % dim = [N+1, N+1] - Hessian matrix of the quadratic optimization problem
    
    lb = zeros(n+1,1);  % dim = [N+1, 1] - lower bound on y's and k
    Aeq = [mu.'  0; ones(1,n)  -1];  % dim = [2, N+1] - coefficient matrix for equality constraints
    beq = [1; 0];  % dim = [2, 1] - constant vector for equality constraints
    
    % It might be useful to increase the tolerance of 'quadprog'
    options = optimoptions( 'quadprog', 'TolFun', 1e-9 );
    
    ynk = quadprog(H, [], [], [], Aeq, beq, lb, [], [], options);  % solve the quadratic programming given the inputs
    
    y = ynk(1:n,:);  % dim = [N, 1] - y's vector is spliced from the variable vector
    k = ynk(end,:);  % dim = [1, 1] - k scalar is extracted from the variable vector

    x = y/k;  % dim = [N, 1] - optimal asset weight vector is calculated by normalizing y's by k 
    
    %----------------------------------------------------------------------
    
end