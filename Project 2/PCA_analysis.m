function [var_explained, cumul_var_explained, H_sw, H_ad, S_diff, K_diff] = PCA_analysis(returns, p, t)

    % Find the total number of time periods and assets, respectively.
    [T,n] = size(returns); 
    
    X = returns.';  % dim = [N, T] - transpose the input to match the dimensions throughout the calculation
    mu = mean(X,2);  % dim = [N, 1] - vector of asset expected returns
    Q = cov(X.');  % dim = [N, N] - matrix of asset covariances
    [Gamma, Lambda] = eig(Q);  % dim = [N, N], [N, N] - matrix of eigenvectors as columns, matrix of corresponding eigenvalues as diagonal elements
    
    % flip Gamma and Lambda such that they are in decreasing order of eigenvalues
    Gamma = flip(Gamma,2);
    Lambda = flip(flip(Lambda,1),2);
    
    pcs = Gamma.' * (X - mu);  % dim = [N, T] - matrix of principal components (factors) in increasing order of eigenvalues
    
    % set flag as 1 (default value) for skewness and kurtosis
    S = skewness(pcs,1,2);  % dim = [N, 1] - vector of factor skewness
    K = kurtosis(pcs,1,2);  % dim = [N, 1] - vector of factor kurtosis
    
    
    % Variance of PCA factors
    eig_vals = diag(Lambda);
    var_explained = zeros(1,n);
    cumul_var_explained = zeros(1,n);
    for i = 1:n
        var_explained(1,i) = eig_vals(i,1)/sum(eig_vals);
        cumul_var_explained(1,i) = sum(var_explained(1,1:i));
    end
    
    fprintf('Cumulative variance explained by %d PCA factors at time %d: %f\n', p, t, cumul_var_explained(1,p));
    
    domain = linspace(1,n,n);
    fig = figure();
    plot(domain, var_explained, domain, cumul_var_explained);
    tags = {'Variance Explained', 'Cumulative Variance Explained'};
    legend(tags, 'Location', 'Best');
    title_name = ['PCA Scree Plot, Time Period = '  num2str(t)];
    title(title_name, 'FontSize', 14);
    ylabel('Variance','FontSize',12);
    xlabel('PCA Factor','FontSize',12);
    
    % If you want to save the figure as .pdf for use in LaTeX
    % print(fig,'Sharpe Ratio Portfolio Weights','-dpdf','-r0');

    % If you want to save the figure as .png for use in MS Word
    print(fig,title_name,'-dpng','-r0');
    hold on;
    
    
    % Normality Testing
    % Histogram of first PCA factor to illustrate normality
    fig_hist = figure();
    histogram(pcs(1,:), 10);  % nbins = 10
    title_name = ['First Principal Factor Distribution, Time Period = '  num2str(t)];
    title(title_name, 'FontSize', 14);
    xlabel('Factors','interpreter','latex','FontSize',14);
    ylabel('Frequency','interpreter','latex','FontSize',14);
    
    % If you want to save the figure as .pdf for use in LaTeX
    % print(fig,'Sharpe Ratio Portfolio Weights','-dpdf','-r0');

    % If you want to save the figure as .png for use in MS Word
    print(fig_hist,title_name,'-dpng','-r0');
    hold on;
    
    % QQ plot of first PCA factor to illustrate normality
    fig_qq = figure();
    qqplot(pcs(1,:));
    title_name = ['First Principal Factor QQ Plot, Time Period = '  num2str(t)];
    title(title_name, 'FontSize', 14);
    
    % If you want to save the figure as .pdf for use in LaTeX
    % print(fig,'Sharpe Ratio Portfolio Weights','-dpdf','-r0');

    % If you want to save the figure as .png for use in MS Word
    print(fig_qq,title_name,'-dpng','-r0');
    hold on;
    
    % Shapiro-Wilk test for normality
    alpha = 0.05;
    H_sw = zeros(1,n);
    for i = 1:n
        [H, pValue, W] = swtest(pcs(i,:).', alpha);
        H_sw(1,i) = H;
    end
    
    % Anderson-Darling test for normality
    H_ad = zeros(1,n);
    for i = 1:n
        [h,p] = adtest(pcs(i,:).');
        H_ad(1,i) = h;
    end
    
    
    % Higher Moments Analysis
    S_normal = zeros(n,1);
    K_normal = 3 * ones(n,1);
    S_diff = S - S_normal;
    K_diff = K - K_normal;
    
end