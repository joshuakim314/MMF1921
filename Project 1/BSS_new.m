function  [mu, Q, Adjusted_R2] = BSS(returns, factRet, lambda, K)
    
    % Use this function for the BSS model. Note that you will not use 
    % lambda in this model (lambda is for LASSO).
    %
    % You should use an optimizer to solve this problem. Be sure to comment 
    % on your code to (briefly) explain your procedure.
    
 
    % *************** WRITE YOUR CODE HERE ***************
    %----------------------------------------------------------------------
    % returns [r_1 r_2 ... r_n]: asset returns (size: T*N)
    % factRet [f_1 f_2 ... f_p]: factor returns (size: T*P)
    
    % T: over time t = 1,...,T
    T = size(returns,1);
    % N: total # of assets
    N = size(returns,2);
    % P: total # of factors
    P = size(factRet,2);
    P_1 = P + 1; % # of factors + 1 intercept
    
    % X = [1 f]: data matrix (size: T*(P+1))
    X = [ones(T, 1) factRet];

    % objective: min (r_i - X B_i)'(r_i - X B_i) 
    %            => min (r_i' r_i - 2 r_i' X B_i + B_i' X' X B_i)
    %            => min (- 2 r_i' X B_i + B_i' X' X B_i)
    %            => Q = X' X
    Q = X.' * X;
    
    % include variable y_i to manage the # of non-zero b_i => re-size Q
    Q = [Q zeros(P_1); zeros(P_1, 2 * P_1)];
    
	% Matrix to store expected returns (size: T*N)
	expectedReturns = zeros(T, N);
    % Matrix to store optimized betas (size: (P+1)*N)
    betas = zeros(P_1, N);
	% Vector to store residual variances (size: N*1)
    D = zeros(N, 1);
    
    for i = 1:N % loop through each asset
        r_i = returns(:,i);
        
        % Inequality
        % -100 * y_i <= B_i <= 100 * y_i
        A = [-eye(P_1) (-100 * eye(P_1)); eye(P_1) (-100 * eye(P_1))];
        b = zeros(2 * P_1, 1);
        
        % Equality
        % Sum of y_i = K
        Aeq = [zeros(1, P_1) ones(1, P_1)];
        beq = K;
        
        % Boundary
        % b_i is unbounded, 0 <= yi <= 1
        lb = [(-ones(1, P_1) * 100) zeros(1, P_1)];
        ub = [(ones(1, P_1) * 100) ones(1, P_1)];
        
        %% Setup the Gurobi model
        clear model;
        
        % Define the c vector in the objective
        model.obj = [-2 * r_i.' * X zeros(1, P_1)];
        % Define the Q matrix in the objective 
        model.Q = sparse(Q);
        
        % Define the constraints
        model.A = [sparse(A); sparse(Aeq)];
        model.rhs = [b; beq];
        model.sense = [repmat('<', 2 * P_1, 1); '='];
        model.lb = lb;
        model.ub = ub;

        % Define the variable types: (p+1 b_i, p+1 y_i)
        % 'C' defines a continuous variable, 'B' defines a binary variable
        varTypes = [repmat('C', P_1, 1); repmat('B', P_1, 1)];
        model.vtype = varTypes;
        
        % Set some Gurobi parameters 
        clear params;
        params.TimeLimit = 100; % limit the runtime
        params.OutputFlag = 0; % avoid printing the output to the console
        
        %% Interprete and store the Gurobi model result
        results = gurobi(model,params);% optimization result
        B_i = results.x(1:9);
        expectedReturns_i = X * B_i;
        
        % expected returns
        expectedReturns(:, i) = expectedReturns_i;
        % optimized betas
        betas(:, i) = B_i;
        % residual variances
        D(i) = norm(expectedReturns_i - r_i)^2 / (T - K);
    end
    
    % matrix of factor loadings for all assets (size: P*N)
    V = betas(2:9, :);
    % factor covariance matrix (size: P*P)
    F = cov(factRet);
    % diagonal matrix of residual variances (size: N*N)
	D = diag(D);
    % asset covariance matrix (size: N*N)
    Q = V.' * F * V + D;
    % vector of asset expected returns (size: N*1)
    mu = mean(expectedReturns)';

    % R_squre_adj
    SS_tot = sum((returns - mean(returns)).^2);
    SS_res = sum((returns - expectedReturns).^2);
    R_2 = 1 - SS_res./SS_tot;
    Adjusted_R2 = 1 - (1 - R_2) * (T - 1) / (T - K);
    %----------------------------------------------------------------------
    
end
