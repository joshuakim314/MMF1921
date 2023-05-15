function  x = CVaR(scenarios, mu, targetRet, alpha)
    
    % Use this function to construct your CVaR portfolios.
    %
    % You can use linprog to solve this problem. Just be sure to comment on 
    % your code to (briefly) explain your steps. 
 
    % Find the total number of assets
    [n,S] = size(scenarios);
    
    % *************** WRITE YOUR CODE HERE ***************
    %----------------------------------------------------------------------
    
    % We can model CVaR Optimization as a Linear Program. 
    % 
    %   min     gamma + (1 / [(1 - alpha) * S]) * sum( z_s )
    %   s.t.    z_s   >= 0,                 for s = 1, ..., S
    %           z_s   >= -r_s' x - gamma,   for s = 1, ..., S  
    %           1' x  =  1,
    %           mu' x >= R
    % 
    % Therefore, we will use MATLAB's 'linprog' in this example. In this 
    % section of the code we will construct our inequality constraint matrix 
    % 'A' and 'b' for
    % 
    %   A x <= b
    
    % Define the lower and upper bounds to our portfolio
    lb = [ zeros(n,1); zeros(S,1); -inf ];
    
    % Define the inequality constraint matrices A and b
    A = [ -scenarios.' -eye(S) -ones(S,1); -mu' zeros(1,S) 0 ];
    b = [ zeros(S, 1); -targetRet ];

    % Define the equality constraint matrices A_eq and b_eq
    Aeq = [ ones(1,n) zeros(1,S) 0 ];
    beq = 1;

    % Define our objective linear cost function c
    k = (1 / ( (1 - alpha) * S) );
    c = [ zeros(n,1); k * ones(S,1); 1 ];
    
    % Set the linprog options to increase the solver tolerance
    options = optimoptions('linprog','TolFun',1e-9);

    % Use 'linprog' to find the optimal portfolio
    y = linprog(c, A, b, Aeq, beq, lb, [], options);

    % Retrieve the optimal portfolio weights
    x = y(1:n);  % Optimal asset weights
    
    %----------------------------------------------------------------------
    
end