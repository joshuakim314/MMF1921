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
    K = 4;

    % Example: Use an OLS regression to estimate mu and Q
    % [mu, Q] = OLS(returns, factRet, lambda, K);
    [mu, Q, D, V, F, S, K] = PCA(returns);
    
    % Example: Use MVO to optimize our portfolio
    x = MVO(mu, Q);
    % NoSims = 10000;
    % Confidence level for CVaR
    % alpha = 0.95;
    % targetRet_percentile = 1.0;
    % targetRet = targetRet_percentile * mean(mu);
    % scenarios = MCHM(mu, D, V, F, S, K, NoSims);
    % x = CVaR(scenarios, mu, targetRet, alpha);

    %----------------------------------------------------------------------
end
