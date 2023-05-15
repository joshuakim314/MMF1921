function  x = RP(mu, Q)
   
    % Use this function to construct your risk parity portfolio.
    %
    % You can use fmincon to solve this problem. Just be sure to comment on
    % your code to (briefly) explain your steps.
 
    % Find the total number of assets
    n = size(mu, 1);
   
    % *************** WRITE YOUR CODE HERE ***************
    %----------------------------------------------------------------------
    
    % A convex RP model is implemented here.
    
    y0 = ones(n, 1);  % dim = [N, 1] - initial value vector of 1's to be feeded into the minimization problem
    c = 1;  % dim = [1, 1] - an arbitrary positive scalar value
    lb = zeros(n,1);  % dim = [N, 1] - lower bound on y's
   
    fun = @(y) (0.5 * y.' * Q * y) - c*sum(log(y));  % the objective function to be minimized
   
    y = fmincon(fun, y0, [], [], [], [], lb, []);  % solve the minimization problem given the inputs

    x = y/sum(y);  % dim = [N, 1] - optimal asset weight vector is calculated by normalizing y's by its sum
    
    %----------------------------------------------------------------------
   
end