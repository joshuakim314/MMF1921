function  [mu, Q, B, errors] = BSS(returns, factRet, lambda, K)
    
    % Use this function for the BSS model. Note that you will not use 
    % lambda in this model (lambda is for LASSO).
    %
    % You should use an optimizer to solve this problem. Be sure to comment 
    % on your code to (briefly) explain your procedure.
    
 
    % *************** WRITE YOUR CODE HERE ***************
    %----------------------------------------------------------------------
    % for sufficiently large bound (subject to change) 
    lb = -100; 
    ub = 100; 
    
    [T,p] = size(factRet);
    N = size(returns,2);
    B_y = zeros(2*(p+1),N);
    
    X = [ones(size(factRet,1),1), factRet];
    Q = [X.'*X zeros(p+1); zeros(p+1,2*(p+1))];
    
    % Define the variable types:'C' defines a continuous variable, 'B' defines
    % a binary variable
    varTypes = [repmat('C', p+1, 1); repmat('B', p+1, 1)];
    
    A = [-eye(p+1)    lb*eye(p+1);
         eye(p+1)     -ub*eye(p+1);
         zeros(1,p+1) ones(1,p+1)];
    b = [zeros(2*(p+1),1); K];
    
    
    for i = 1:N
        clear model;

        % Gurobi accepts an objective function of the following form:
        % f(x) = (1/2) x' H x + c' x 

        % Define the Q matrix in the objective 
        model.Q = sparse(Q);

        % define the c vector in the objective (which is a vector of zeros since
        % there is no linear term in our objective)
        c = [-2*(X.'*returns(:,i)); zeros(p+1,1)];
        model.obj = c;

        % Gurobi only accepts a single A matrix, with both inequality and equality
        % constraints
        model.A = sparse(A);

        % Define the right-hand side vector b
        model.rhs = full(b);

        % Indicate whether the constraints are ">=", "<=", or "="
        model.sense = repmat('<', (2*(p+1) + 1), 1);
        
        % bounds
        model.lb = [(ones(1, p+1) * lb) zeros(1, p+1)];
        model.ub = [(ones(1, p+1) * ub) ones(1, p+1)];

        % Define the variable type (continuous, integer, or binary)
        model.vtype = varTypes;

        % Set some Gurobi parameters to limit the runtime and to avoid printing the
        % output to the console. 
        clear params;
        params.TimeLimit = 100;
        params.OutputFlag = 0;

        results = gurobi(model,params);
        B_y(:,i) = results.x;

    end
    B = B_y(1:p+1,:);
    
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