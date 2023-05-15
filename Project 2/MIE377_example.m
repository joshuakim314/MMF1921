%% MMF1921 (Summer 2023) - Project 2
%
% Student Name: Joshua Ha Rim Kim
% Student ID: 1004391339

clc
clear all
format short

% Program Start
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. Read input files
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load the stock monthly prices
adjClose = readtable('/Users/joshuakim/Desktop/WINTER2020/MIE377/Projects/Project 2 (Final Assessment)/MIE377 - Final Assessment Files/MIE377_AssetPrices.csv');
adjClose.Properties.RowNames = cellstr(datetime(adjClose.Date));
adjClose.Properties.RowNames = cellstr(datetime(adjClose.Properties.RowNames));
adjClose.Date = [];

% Load the monthly risk-free rate
riskFree = readtable('/Users/joshuakim/Desktop/WINTER2020/MIE377/Projects/Project 2 (Final Assessment)/MIE377 - Final Assessment Files/MIE377_RiskFree.csv');
riskFree.Properties.RowNames = cellstr(datetime(riskFree.Date));
riskFree.Properties.RowNames = cellstr(datetime(riskFree.Properties.RowNames));
riskFree.Date = [];

% Identify the tickers and the dates 
tickers = adjClose.Properties.VariableNames';
dates   = datetime(riskFree.Properties.RowNames);

% Calculate the stocks' monthly EXCESS returns
prices  = table2array(adjClose);
returns = ( prices(2:end,:) - prices(1:end-1,:) ) ./ prices(1:end-1,:);
returns = returns - ( diag( table2array(riskFree) ) * ones( size(returns) ) );
returns = array2table(returns);
returns.Properties.VariableNames = tickers;
returns.Properties.RowNames = cellstr(datetime(riskFree.Properties.RowNames));

% Align the price table to the asset and factor returns tables by
% discarding the first observation.
adjClose = adjClose(2:end,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2. Define your initial parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initial budget to invest ($100,000)
initialVal = 100000;

% Start of in-sample calibration period 
calStart = datetime('2008-01-01');
calEnd   = calStart + calyears(4) - days(1);

% Start of out-of-sample test period 
testStart = datetime('2012-01-01');
testEnd   = testStart + calyears(1) - days(1);

% Number of Monte Carlo simulations
NoSims = 2500;

% Number of investment periods (each investment period is 1 year long)
NoPeriods = 5;

% Number of Assets
NoAssets = length(tickers);

% Confidence level for CVaR
alpha = 0.95;

% Number of models: i) CVaR Opt with normality, ii) CVaR Opt with higher
% moments, iii) risk parity, iv) Sharpe ratio maximization
MCList   = {'MCNorm' 'MCHM'};
MCList   = cellfun(@str2func, MCList, 'UniformOutput', false);
optList  = {'CVaR' 'CVaR' 'RP' 'MaxSR'};
optList  = cellfun(@str2func, optList, 'UniformOutput', false);
NoModels = length(optList);

% Tags for the portfolios under the different simulation methods
tags = {'CVaR w/ normality' 'CVaR w/ HM' 'Risk parity' 'Max SR'};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3. Construct and rebalance your portfolios
%
% Here you will estimate your input parameters, perform your Monte Carlo 
% simulations, and optimize your portfolio. You will have to re-estimate 
% your parameters and re-optimize your portfolios at the end 
% re-estimate your parameters at the start of each rebalance period, and 
% then re-optimize and rebalance your portfolios accordingly. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initiate counter for the number of observations per investment period
toDay = 0;

% Preallocate the space for the per period value of the portfolios 
currentVal = zeros(NoPeriods, NoModels);

% Preallocate the space for variance and normality testing per period value of PCA factors
var_explained = zeros(NoPeriods, NoAssets);
cumul_var_explained = zeros(NoPeriods, NoAssets);
H_sw = zeros(NoPeriods, NoAssets);
H_ad = zeros(NoPeriods, NoAssets);
S_diff = zeros(NoPeriods, NoAssets);
K_diff = zeros(NoPeriods, NoAssets);

% Preallocate the space for ex-ante sharpe ratio per period of the portfolios
ex_ante_SR = zeros(NoPeriods, NoModels);

% Preallocate the space for VaR and CVaR of confidence=alpha per period of the portfolios
VaR_estimated = zeros(NoPeriods, NoModels);
CVaR_estimated = zeros(NoPeriods, NoModels);

for t = 1 : NoPeriods
  
    % Subset the returns and factor returns corresponding to the current
    % calibration period.
    periodReturns = table2array( returns( calStart <= dates & dates <= calEnd, :) );
    currentPrices = table2array( adjClose( ( calEnd - days(7) ) <= dates ... 
                                                    & dates <= calEnd, :) )';
    
    % Subset the prices corresponding to the current out-of-sample test 
    % period.
    periodPrices = table2array( adjClose( testStart <= dates & dates <= testEnd,:) );
    
    % Set the initial value of the portfolio or update the portfolio value
    if t == 1
        
        currentVal(t,:) = initialVal;
        
    else
        for i = 1 : NoModels
            
            currentVal(t,i) = currentPrices' * NoShares{i};
            
        end
    end
    
    % Update counter for the number of observations per investment period
    fromDay = toDay + 1;
    toDay   = toDay + size(periodPrices,1);
    
    % Use the PCA factor model to calculate following items
    % i)   Asset expected returns 'mu' 
    % ii)  Asset covariance matrix 'Q' 
    % iii) Diagonal matrix of idiosyncratic risk 'D'
    % iv)  Asset factor loadings 'V'
    % v)   Factor covariance matrix 'F'
    % vi)  Factor skewness 'S'
    % vii) Factor kurtosis 'K'
    % Note: You need to write the code for the PCA factor model.     
    [mu, Q, D, V, F, S, K] = PCA(periodReturns);
    
    % See the explained variance and normality testing by PCA factors.
    [var_explained(t,:), cumul_var_explained(t,:), H_sw(t,:), H_ad(t,:), S_diff(t,:), K_diff(t,:)] = analyze_PCA(periodReturns, 3, t);
            
    % Define the target return as the mean of the asset returns for the 
    % current calibration period
    targetRet_percentile = 1.0;
    targetRet = targetRet_percentile * mean(mu);
    
    % Optimize your portfolios to get the weights 'x'
    for i = 1 : NoModels
        if i <= 2
            
            % Note: You need to write the code for the MC simulations with  
            % higher moments (the function for normally-distributed factors
            % 'MCNorm' is already coded for you). The output of the MC 
            % functions should be the asset scenarios (not the factor
            % scenarios)
            scenarios = MCList{i}(mu, D, V, F, S, K, NoSims); 
        
            % Note: You need to write the code for the optimization models
            x{i}(:,t) = optList{i}(scenarios, mu, targetRet, alpha); 
        else
            
            % Note: You need to write the code for the optimization models
            x{i}(:,t) = optList{i}(mu, Q); 
        end
        
        % Calculate the risk contribution of the model
        [VaR_estimated(t,i), CVaR_estimated(t,i)] = analyze_CVaR(x{i}(:,t), periodReturns, alpha, 0);
        risk_contribution{i}(:,t) = analyze_RP(x{i}(:,t), Q);
        ex_ante_SR(t,i) = analyze_SR(x{i}(:,t), mu, Q);
    end
    
    % Calculate the optimal number of shares of each stock you should hold
    for i = 1 : NoModels
        
        % Number of shares your portfolio holds per stock
        NoShares{i} = x{i}(:,t) .* currentVal(t,i) ./ currentPrices;
        
        % Monthly portfolio value during the out-of-sample window
        portfValue(fromDay:toDay,i) = periodPrices * NoShares{i};
        
    end

    % Update your calibration and out-of-sample test periods
    calStart = calStart + calyears(1);
    calEnd   = calStart + calyears(4) - days(1);
    
    testStart = testStart + calyears(1);
    testEnd   = testStart + calyears(1) - days(1);

end


% SPY portfolio evolution over the investment horizon
spy_portfValue = zeros(1 + NoPeriods*12,1);
spy_portfValue(1,1) = initialVal;
spy_periodReturns = table2array(spy_returns);
for t = 2:(NoPeriods*12)+1
    spy_portfValue(t,1) = spy_portfValue(t-1,1) * (1+spy_periodReturns(t-1));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4. Results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%--------------------------------------------------------------------------
% 4.1 Evaluate any in-sample measures of risk or return, such as the
% ex-ante Sharpe ratio or the portfolio VaR and/or CVaR. 
%--------------------------------------------------------------------------

% PCA Factor Normality Testing
H_ad_summed = sum(H_ad, 2);
H_sw_summed = sum(H_sw, 2);


% Skewness Analysis Plots
fig = figure();
for t = 1:5
    scatter(t*ones(1,NoAssets), S_diff(t,:));
    hold on;
end
yline(0.8, '--', 'Acceptable Upper Bound = 0.8', 'LineWidth', 1, 'Color', 'r');
yline(0, '--', 'Skewness of Normal Distribution = 0', 'LineWidth', 2, 'Color', 'r');
yline(-0.8, '--', 'Acceptable Lower Bound = -0.8', 'LineWidth', 1, 'Color', 'r');
title('Factor Skewness', 'FontSize', 14);
ylabel('Skewness','interpreter','latex','FontSize',12);
xlabel('Rebalance period','interpreter','latex','FontSize',12);

% If you want to save the figure as .pdf for use in LaTeX
% print(fig,'CVaR Normality Portfolio Risk Contributions','-dpdf','-r0');

% If you want to save the figure as .png for use in MS Word
print(fig,'Factor Skewness','-dpng','-r0');

% Kurtosis Analysis Plots
fig = figure();
for t = 1:5
    scatter(t*ones(1,NoAssets), (K_diff(t,:)+3));
    hold on;
end
yline(6, '--', 'Acceptable Upper Bound = 6', 'LineWidth', 1, 'Color', 'r');
yline(3, '--', 'Kurtosis of Normal Distribution = 3', 'LineWidth', 2, 'Color', 'r');
yline(0, '--', 'Acceptable Lower Bound = 0', 'LineWidth', 1, 'Color', 'r');
title('Factor Kurtosis', 'FontSize', 14);
ylabel('Kurtosis','interpreter','latex','FontSize',12);
xlabel('Rebalance period','interpreter','latex','FontSize',12);

% If you want to save the figure as .pdf for use in LaTeX
% print(fig,'CVaR Normality Portfolio Risk Contributions','-dpdf','-r0');

% If you want to save the figure as .png for use in MS Word
print(fig,'Factor Kurtosis','-dpng','-r0');


% Risk contribution area plots
% CVaR w/ normality risk contribution area plot
fig = figure();
area(risk_contribution{1}')
legend(tickers, 'Location', 'eastoutside','FontSize',12);
title('CVaR Normality Portfolio Risk Contributions', 'FontSize', 14)
ylabel('Contribution Weigths','interpreter','latex','FontSize',12);
xlabel('Rebalance period','interpreter','latex','FontSize',12);

% Define the plot size in inches
set(fig,'Units','Inches', 'Position', [0 0 8, 5]);
pos1 = get(fig,'Position');
set(fig,'PaperPositionMode','Auto','PaperUnits','Inches',...
    'PaperSize',[pos1(3), pos1(4)]);

% If you want to save the figure as .pdf for use in LaTeX
% print(fig,'CVaR Normality Portfolio Risk Contributions','-dpdf','-r0');

% If you want to save the figure as .png for use in MS Word
print(fig,'CVaR Normality Portfolio Risk Contributions','-dpng','-r0');


% CVaR w/ HM risk contribution area plot
fig = figure();
area(risk_contribution{2}')
legend(tickers, 'Location', 'eastoutside','FontSize',12);
title('CVaR HM Portfolio Risk Contributions', 'FontSize', 14)
ylabel('Contribution Weigths','interpreter','latex','FontSize',12);
xlabel('Rebalance period','interpreter','latex','FontSize',12);

% Define the plot size in inches
set(fig,'Units','Inches', 'Position', [0 0 8, 5]);
pos1 = get(fig,'Position');
set(fig,'PaperPositionMode','Auto','PaperUnits','Inches',...
    'PaperSize',[pos1(3), pos1(4)]);

% If you want to save the figure as .pdf for use in LaTeX
% print(fig,'CVaR HM Portfolio Risk Contributions','-dpdf','-r0');

% If you want to save the figure as .png for use in MS Word
print(fig,'CVaR HM Portfolio Risk Contributions','-dpng','-r0');


% Risk Parity risk contribution area plot
fig = figure();
area(risk_contribution{3}')
legend(tickers, 'Location', 'eastoutside','FontSize',12);
title('Risk Parity Portfolio Risk Contributions', 'FontSize', 14)
ylabel('Contribution Weigths','interpreter','latex','FontSize',12);
xlabel('Rebalance period','interpreter','latex','FontSize',12);

% Define the plot size in inches
set(fig,'Units','Inches', 'Position', [0 0 8, 5]);
pos1 = get(fig,'Position');
set(fig,'PaperPositionMode','Auto','PaperUnits','Inches',...
    'PaperSize',[pos1(3), pos1(4)]);

% If you want to save the figure as .pdf for use in LaTeX
% print(fig,'Risk Parity Portfolio Risk Contributions','-dpdf','-r0');

% If you want to save the figure as .png for use in MS Word
print(fig,'Risk Parity Portfolio Risk Contributions','-dpng','-r0');


% Sharpe Ratio risk contribution area plot
fig = figure();
area(risk_contribution{4}')
legend(tickers, 'Location', 'eastoutside','FontSize',12);
title('Sharpe Ratio Portfolio Risk Contributions', 'FontSize', 14)
ylabel('Contribution Weigths','interpreter','latex','FontSize',12);
xlabel('Rebalance period','interpreter','latex','FontSize',12);

% Define the plot size in inches
set(fig,'Units','Inches', 'Position', [0 0 8, 5]);
pos1 = get(fig,'Position');
set(fig,'PaperPositionMode','Auto','PaperUnits','Inches',...
    'PaperSize',[pos1(3), pos1(4)]);

% If you want to save the figure as .pdf for use in LaTeX
% print(fig,'Sharpe Ratio Portfolio Risk Contributions','-dpdf','-r0');

% If you want to save the figure as .png for use in MS Word
print(fig,'Sharpe Ratio Portfolio Risk Contributions','-dpng','-r0');


% Sharpe Ratio plot
fig = figure();

for i = 1 : NoModels
    
    plot( ex_ante_SR(:,i) )
    hold on
    
end

legend(tags, 'Location', 'eastoutside','FontSize',12);
title('Portfolio Sharpe Ratio', 'FontSize', 14)
ylabel('Sharpe Ratio','interpreter','latex','FontSize',12);
xlabel('Rebalance period','interpreter','latex','FontSize',12);

% Define the plot size in inches
set(fig,'Units','Inches', 'Position', [0 0 8, 5]);
pos1 = get(fig,'Position');
set(fig,'PaperPositionMode','Auto','PaperUnits','Inches',...
    'PaperSize',[pos1(3), pos1(4)]);

% If you want to save the figure as .pdf for use in LaTeX
% print(fig,'Sharpe Ratio Plot','-dpdf','-r0');

% If you want to save the figure as .png for use in MS Word
print(fig,'Sharpe Ratio Plot','-dpng','-r0');


% VaR/CVaR Analysis
% Start of in-sample calibration period 
calStart = datetime('2008-01-01');
calEnd   = calStart + calyears(4) - days(1);
for t = 1:NoPeriods
    for i = 1:NoModels
        fig = figure();
        periodReturns = table2array( returns( calStart <= dates & dates <= calEnd, :) );
        losses = -periodReturns * x{i}(:,t);
        histogram(losses, 15);
        hold on;
        line([VaR_estimated(t,i), VaR_estimated(t,i)], ylim, 'LineWidth', 1, 'Color', 'r', 'LineStyle','--');
        hold on;
        line([CVaR_estimated(t,i), CVaR_estimated(t,i)], ylim, 'LineWidth', 1, 'Color', 'r', 'LineStyle','--');
        text(VaR_estimated(t,i),3.5,[' \leftarrow VaR = '  num2str(VaR_estimated(t,i),3)  '%'], 'Color','red');
        text(CVaR_estimated(t,i),2.5,[' \leftarrow CVaR = '  num2str(CVaR_estimated(t,i),3)  '%'], 'Color','red');
        xlabel('Portfolio losses ($\%$)','interpreter','latex','FontSize',14);
        ylabel('Frequency','interpreter','latex','FontSize',14);
        model_name = '';
        if i == 1
            model_name = 'CVaR Normality';
        elseif i == 2
            model_name = 'CVaR HM';
        elseif i == 3
            model_name = 'Risk Parity';
        elseif i == 4
            model_name = 'Sharpe Ratio';
        end
        title_name = [model_name  ' Portfolio VaR & CVaR: Period = '  num2str(t)];
        title(title_name, 'FontSize', 14);

        set(fig,'Units','Inches', 'Position', [0 0 10, 4]);
        pos2 = get(fig,'Position');
        set(fig,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos2(3), pos2(4)])
        
        % If you want to save the figure as .pdf for use in LaTeX
        % print(fig,title_name,'-dpdf','-r0');

        % If you want to save the figure as .png for use in MS Word
        print(fig,title_name,'-dpng','-r0');
    end
    % Update your calibration periods
    calStart = calStart + calyears(1);
    calEnd   = calStart + calyears(4) - days(1);
end

%--------------------------------------------------------------------------
% 4.2 Calculate the portfolio average return, variance (or standard 
% deviation), Sharpe ratio, and any other performance and/or risk metric 
% you wish to include in your report.
%--------------------------------------------------------------------------

N = size(portfValue,1);  % number of months in investment horizon (60 months in total = 5 years * 12 months)
portf_returns = (portfValue(2:end,:) ./ portfValue(1:end-1,:)) - 1;  % monthly returns of the portfolio
portf_mu = geomean(portf_returns + 1) - 1;  % monthly average return of the portfolio
portf_Q  = cov(portf_returns);  % covariance matrix of the portfolio
portf_std = (diag(portf_Q).^0.5).';  % monthly portfolio standard deviation
ex_post_SR = portf_mu./portf_std;  % ex-post Sharpe ratio of the portfolio


% Observed VaR/CVaR under the assumption that asset returns are normally distributed
z_score = 0.0;  % one-sided tail z-score of a normal distribution
if alpha == 0.90
    z_score = 1.282;
elseif alpha == 0.95
    z_score = 1.645;
elseif alpha == 0.99
    z_score = 2.326;
end
H_sw_portf = zeros(1, NoModels);
H_ad_portf = zeros(1, NoModels);
VaR_observed_normality = -(portf_mu - z_score*portf_std);
CVaR_observed_normality = zeros(1, NoModels);
for i = 1:NoModels
    cum_prob = integral(@(y) y.*normpdf(y, portf_mu(i), portf_std(i)), -Inf, -VaR_observed_normality(i));
    CVaR_observed_normality(1,i) = -cum_prob / (1-alpha);
     
    % QQ Plots to see if portfolio returns are normal
    fig_qq = figure();
    qqplot(portf_returns(:,i));
    
    model_name = '';
    if i == 1
        model_name = 'CVaR Normality';
    elseif i == 2
        model_name = 'CVaR HM';
    elseif i == 3
        model_name = 'Risk Parity';
    elseif i == 4
        model_name = 'Sharpe Ratio';
    end
    
    title_name = [model_name  ' Portfolio Returns QQ Plot'];
    title(title_name, 'FontSize', 14);
    hold on;
     
    % If you want to save the figure as .pdf for use in LaTeX
    % print(fig,title_name,'-dpdf','-r0');

    % If you want to save the figure as .png for use in MS Word
    print(fig_qq,title_name,'-dpng','-r0');
     
    % Portfolio Shapiro-Wilk Test
    [H, pValue, W] = swtest(portf_returns(:,i), alpha);
    H_sw_portf(1,i) = H;
    % Portfolio Anderson-Darling Test
    [h,p] = adtest(portf_returns(:,i));
    H_ad_portf(1,i) = h;
    
    fig = figure();
    domain_normality = [-0.12:0.0001:0.12];
    range_normality = normpdf(domain_normality, -portf_mu(i), portf_std(i));
    losses = -portf_returns(:,i);
    histogram(losses, 15);
    hold on;
    plot(domain_normality, range_normality);
    hold on;
    
    line([VaR_observed_normality(1,i), VaR_observed_normality(1,i)], ylim, 'LineWidth', 1, 'Color', 'r', 'LineStyle','--');
    hold on;
    line([CVaR_observed_normality(1,i), CVaR_observed_normality(1,i)], ylim, 'LineWidth', 1, 'Color', 'r', 'LineStyle','--');
    text(VaR_observed_normality(1,i),6.5,[' \leftarrow VaR (PDF) = '  num2str(VaR_observed_normality(1,i),3)  '%'], 'Color','red');
    text(CVaR_observed_normality(1,i),5.5,[' \leftarrow CVaR (PDF) = '  num2str(CVaR_observed_normality(1,i),3)  '%'], 'Color','red');
    
    line([VaR_estimated(t,i), VaR_estimated(t,i)], ylim, 'LineWidth', 1, 'Color', 'k', 'LineStyle','--');
    hold on;
    line([CVaR_estimated(t,i), CVaR_estimated(t,i)], ylim, 'LineWidth', 1, 'Color', 'k', 'LineStyle','--');
    hold on;
    text(VaR_estimated(t,i),3.5,[' \leftarrow VaR (MC) = '  num2str(VaR_estimated(t,i),3)  '%'], 'Color','black');
    text(CVaR_estimated(t,i),2.5,[' \leftarrow CVaR (MC) = '  num2str(CVaR_estimated(t,i),3)  '%'], 'Color','black');
    
    xlabel('Portfolio losses ($\%$)','interpreter','latex','FontSize',14);
    ylabel('Frequency','interpreter','latex','FontSize',14);
    
    title_name = [model_name  ' Portfolio VaR & CVaR under Normality'];
    title(title_name, 'FontSize', 14);
    
    % If you want to save the figure as .pdf for use in LaTeX
    % print(fig,title_name,'-dpdf','-r0');

    % If you want to save the figure as .png for use in MS Word
    print(fig,title_name,'-dpng','-r0');
    
end

% Preallocate the space for observed VaR and CVaR of confidence=alpha of the portfolios over the entire investment horizon
VaR_observed = zeros(1, NoModels);
CVaR_observed = zeros(1, NoModels);

for i = 1:NoModels
    [VaR_observed(1,i), CVaR_observed(1,i)] = analyze_CVaR([], portf_returns(:,i), alpha, 1);
end

for i = 1:NoModels
    fig = figure();
    losses = -portf_returns(:,i);
    histogram(losses, 15);
    hold on;
    line([VaR_observed(1,i), VaR_observed(1,i)], ylim, 'LineWidth', 1, 'Color', 'r', 'LineStyle','--');
    hold on;
    line([CVaR_observed(1,i), CVaR_observed(1,i)], ylim, 'LineWidth', 1, 'Color', 'r', 'LineStyle','--');
    text(VaR_observed(1,i),3.5,[' \leftarrow VaR = '  num2str(VaR_observed(1,i),3)  '%'], 'Color','red');
    text(CVaR_observed(1,i),2.5,[' \leftarrow CVaR = '  num2str(CVaR_observed(1,i),3)  '%'], 'Color','red');
    xlabel('Portfolio losses ($\%$)','interpreter','latex','FontSize',14);
    ylabel('Frequency','interpreter','latex','FontSize',14);
    model_name = '';
    if i == 1
        model_name = 'CVaR Normality';
    elseif i == 2
        model_name = 'CVaR HM';
    elseif i == 3
        model_name = 'Risk Parity';
    elseif i == 4
        model_name = 'Sharpe Ratio';
    end
    title_name = [model_name  ' Portfolio Observed VaR & CVaR'];
    title(title_name, 'FontSize', 14);

    set(fig,'Units','Inches', 'Position', [0 0 10, 4]);
    pos2 = get(fig,'Position');
    set(fig,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos2(3), pos2(4)])
        
    % If you want to save the figure as .pdf for use in LaTeX
    % print(fig,title_name,'-dpdf','-r0');

    % If you want to save the figure as .png for use in MS Word
    print(fig,title_name,'-dpng','-r0');
end

%--------------------------------------------------------------------------
% 4.3 Plot the portfolio values 
%--------------------------------------------------------------------------

plotDates = dates(dates >= datetime('2012-01-01') );

fig1 = figure();

for i = 1 : NoModels
    
    plot( plotDates, portfValue(:,i) );
    hold on;
    
end

plot( plotDates, spy_portfValue(2:end,1) );
hold on;

legend([tags  'SPY'], 'Location', 'eastoutside','FontSize',12);
datetick('x','dd-mmm-yyyy','keepticks','keeplimits');
set(gca,'XTickLabelRotation',30);
title('Portfolio value', 'FontSize', 14)
ylabel('Value','interpreter','latex','FontSize',12);

% Define the plot size in inches
set(fig1,'Units','Inches', 'Position', [0 0 8, 5]);
pos1 = get(fig1,'Position');
set(fig1,'PaperPositionMode','Auto','PaperUnits','Inches',...
    'PaperSize',[pos1(3), pos1(4)]);

% If you want to save the figure as .pdf for use in LaTeX
% print(fig1,'Portfolio Values','-dpdf','-r0');

% If you want to save the figure as .png for use in MS Word
print(fig1,'Portfolio Values','-dpng','-r0');

%--------------------------------------------------------------------------
% 4.4 Plot the portfolio weights period-over-period
%--------------------------------------------------------------------------

% CVaR w/ normality portfolio Plot
fig2 = figure();
area(x{1}')
legend(tickers, 'Location', 'eastoutside','FontSize',12);
title('CVaR Normality Portfolio Weights', 'FontSize', 14)
ylabel('Weights','interpreter','latex','FontSize',12);
xlabel('Rebalance period','interpreter','latex','FontSize',12);

% Define the plot size in inches
set(fig2,'Units','Inches', 'Position', [0 0 8, 5]);
pos1 = get(fig2,'Position');
set(fig2,'PaperPositionMode','Auto','PaperUnits','Inches',...
    'PaperSize',[pos1(3), pos1(4)]);

% If you want to save the figure as .pdf for use in LaTeX
% print(fig2,'CVaR Normality Portfolio Weights','-dpdf','-r0');

% If you want to save the figure as .png for use in MS Word
print(fig2,'CVaR Normality Portfolio Weights','-dpng','-r0');


% CVaR w/ HM portfolio Plot
fig3 = figure();
area(x{2}')
legend(tickers, 'Location', 'eastoutside','FontSize',12);
title('CVaR HM Portfolio Weights', 'FontSize', 14)
ylabel('Weights','interpreter','latex','FontSize',12);
xlabel('Rebalance period','interpreter','latex','FontSize',12);

% Define the plot size in inches
set(fig3,'Units','Inches', 'Position', [0 0 8, 5]);
pos1 = get(fig3,'Position');
set(fig3,'PaperPositionMode','Auto','PaperUnits','Inches',...
    'PaperSize',[pos1(3), pos1(4)]);

% If you want to save the figure as .pdf for use in LaTeX
% print(fig3,'CVaR HM Portfolio Weights','-dpdf','-r0');

% If you want to save the figure as .png for use in MS Word
print(fig3,'CVaR HM Portfolio Weights','-dpng','-r0');


% Risk Parity portfolio Plot
fig4 = figure();
area(x{3}')
legend(tickers, 'Location', 'eastoutside','FontSize',12);
title('Risk Parity Portfolio Weights', 'FontSize', 14)
ylabel('Weights','interpreter','latex','FontSize',12);
xlabel('Rebalance period','interpreter','latex','FontSize',12);

% Define the plot size in inches
set(fig4,'Units','Inches', 'Position', [0 0 8, 5]);
pos1 = get(fig4,'Position');
set(fig4,'PaperPositionMode','Auto','PaperUnits','Inches',...
    'PaperSize',[pos1(3), pos1(4)]);

% If you want to save the figure as .pdf for use in LaTeX
% print(fig4,'Risk Parity Portfolio Weights','-dpdf','-r0');

% If you want to save the figure as .png for use in MS Word
print(fig4,'Risk Parity Portfolio Weights','-dpng','-r0');


% Sharpe Ratio portfolio Plot
fig5 = figure();
area(x{4}')
legend(tickers, 'Location', 'eastoutside','FontSize',12);
title('Sharpe Ratio Portfolio Weights', 'FontSize', 14)
ylabel('Weights','interpreter','latex','FontSize',12);
xlabel('Rebalance period','interpreter','latex','FontSize',12);

% Define the plot size in inches
set(fig5,'Units','Inches', 'Position', [0 0 8, 5]);
pos1 = get(fig5,'Position');
set(fig5,'PaperPositionMode','Auto','PaperUnits','Inches',...
    'PaperSize',[pos1(3), pos1(4)]);

% If you want to save the figure as .pdf for use in LaTeX
% print(fig5,'Sharpe Ratio Portfolio Weights','-dpdf','-r0');

% If you want to save the figure as .png for use in MS Word
print(fig5,'Sharpe Ratio Portfolio Weights','-dpng','-r0');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Program End