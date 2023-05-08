function  x = MVO(mu, Q, targetRet)
    
    % Use this function to construct your MVO portfolio subject to the
    % target return, with short sales disallowed. 
    %
    % You may use quadprog, Gurobi, or any other optimizer you are familiar
    % with. Just be sure to comment on your code to (briefly) explain your
    % procedure.

    % Find the total number of assets
    n = size(Q,1); 

    % *************** WRITE YOUR CODE HERE ***************
    %----------------------------------------------------------------------
    % Disallow shortselling
    lb = zeros(1,n);
    
    % Add the expected return constraint
    A = -mu.';
    b = -targetRet;
    
    %constrain weights to sum to 1
    Aeq = ones(1,n);
    beq = 1;
    
    % It might be useful to increase the tolerance of 'quadprog'
    options = optimoptions('quadprog', 'TolFun', 1e-9);

    % Solve this problem using 'quadprog'
    x = quadprog(2*Q, [], A, b, Aeq, beq, lb, [], [], options);   % Optimal asset weights
    % Use this to allow shortselling
    % x = quadprog(2*Q, [], A, b, Aeq, beq, [], [], [], options);  % Optimal asset weights
    %----------------------------------------------------------------------
    
end