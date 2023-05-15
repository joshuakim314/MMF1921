function  scenarios = MCHM(mu, D, V, F, S, K, NoSims)
    
    % Use this function to produce your Monte Carlo simulations without
    % assuming factor normality. Use the function 'pearsrnd()' to generate 
    % the simulated factor returns with higher moments and use the function
    % 'mvnrnd()' to generate the asset residual terms.
    %
    % Be sure to comment on your code to (briefly) explain your procedure.
    
    % Find the total number of assets
    n = size(mu, 1); 
    
    % *************** WRITE YOUR CODE HERE ***************
    %----------------------------------------------------------------------
    
    % Generate the factor scenarios using higher moments of distributions
    factorScnrs = zeros(size(F,1),NoSims);
    for i = 1:size(F,1)
        factorScnrs(i,:) = pearsrnd(0, F(i,i), S(i,1), K(i,1), [1,NoSims]);
    end
    
    % Generate the idiosyncratic noise terms for all assets 
    idiosyncScnrs = mvnrnd(zeros(n,1), D, NoSims)';
    
    % Generate the asset scenarios (dim. of scenario matrix: n x NoSims)
    scenarios = mu + V' * factorScnrs + idiosyncScnrs; 
    
    %----------------------------------------------------------------------
    
end