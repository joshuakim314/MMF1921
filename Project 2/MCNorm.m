function  scenarios = MCNorm(mu, D, V, F, NoSims)
    
    % This function performs Monte Carlo simulations under the assumption
    % of normality. Use the function 'mvnrnd()' to generate both the
    % simulated factor returns and the asset residual terms.
    %
    % This function has already been coded for you. Note that this function
    % does not use the inputs 'S' or 'K'.
    
    % Find the total number of assets
    n = size(mu, 1); 
    
    % Generate the factor scenarios under the assumption of normality
    factorScnrs = mvnrnd(zeros(size(F,1),1), F, NoSims)';
    
    % Generate the idiosyncratic noise terms for all assets 
    idiosyncScnrs = mvnrnd(zeros(n,1), D, NoSims)';
    
    % Prepare the asset scenarios (dim. of scenario matrix: n x NoSims)
    scenarios = mu + V' * factorScnrs + idiosyncScnrs; 
    %----------------------------------------------------------------------
    
end