function x = Project2_Function(periodReturns, periodFactRet, x0)

    % Use this function to implement your algorithmic asset management
    % strategy. You can modify this function, but you must keep the inputs
    % and outputs consistent.
    %
    % INPUTS: periodReturns, periodFactRet, x0 (current portfolio weights)
    % OUTPUTS: x (optimal portfolio)
    %
    % An example of an MVO implementation with OLS regression is given
    % below. Please be sure to include comments in your code.
    %
    % *************** WRITE YOUR CODE HERE ***************
    %----------------------------------------------------------------------

    % Example: subset the data to consistently use the most recent 5 years
    % for parameter estimation
    returns = periodReturns(end-59:end,:);
    factRet = periodFactRet(end-59:end,:);

    % parameters
    lambda = 0.03;
    K = 5;

    % regime detection
    periodQ = cov(periodReturns(end-5:end,:));
    avg_vol = trace(periodQ) / size(periodReturns, 2);
    lin_comb_coeff = max(0, 1 - avg_vol/0.02);

    % Use an OLS regression and PCA to estimate mu and Q
    [mu_OLS, Q_OLS, B_OLS, errors_OLS, D_OLS, V_OLS, F_OLS] = OLS(returns, factRet, lambda, K);
    [mu_PCA, Q_PCA, B_PCA, errors_PCA, D_PCA, V_PCA, F_PCA] = PCA(returns);
    
    % Use RP to optimize our portfolio
    x_OLS = RP(mu_OLS, Q_OLS);
    x_PCA = RP(mu_PCA, Q_PCA);
    
    x = lin_comb_coeff*x_PCA + (1-lin_comb_coeff)*x_OLS;
    %----------------------------------------------------------------------
end
