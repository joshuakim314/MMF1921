%% MMF1921 (Summer 2023) - Project 1
% 
% The purpose of this program is to implement the following factor models
% a) Multi-factor OLS regression
% b) Fama-French 3-factor model
% c) LASSO
% d) Best Subset Selection
% 
% and to use these factor models to estimate the asset expected returns and 
% covariance matrix. These parameters will then be used to test the 
% out-of-sample performance using MVO to construct optimal portfolios.
% 
% Use can use this template to write your program.
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

% Load the stock weekly prices
adjClose = readtable('MMF1921_AssetPrices.csv');
adjClose.Properties.RowNames = cellstr(datetime(adjClose.Date));
adjClose.Properties.RowNames = cellstr(datetime(adjClose.Properties.RowNames));
adjClose.Date = [];

% Load the factors weekly returns
factorRet = readtable('MMF1921_FactorReturns.csv');
factorRet.Properties.RowNames = cellstr(datetime(factorRet.Date));
factorRet.Properties.RowNames = cellstr(datetime(factorRet.Properties.RowNames));
factorRet.Date = [];

riskFree = factorRet(:,9);
factorRet = factorRet(:,1:8);

% Identify the tickers and the dates 
tickers = adjClose.Properties.VariableNames';
dates   = datetime(factorRet.Properties.RowNames);

% Calculate the stocks' weekly EXCESS returns
prices  = table2array(adjClose);
returns = ( prices(2:end,:) - prices(1:end-1,:) ) ./ prices(1:end-1,:);
returns = returns - ( diag( table2array(riskFree) ) * ones( size(returns) ) );
returns = array2table(returns);
returns.Properties.VariableNames = tickers;
returns.Properties.RowNames = cellstr(datetime(factorRet.Properties.RowNames));

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

% Number of investment periods (each investment period is 1 year long)
NoPeriods = 5;

% Factor models
% Note: You must populate the functios OLS.m, FF.m, LASSO.m and BSS.m with your
% own code.
FMList = {'OLS' 'FF' 'LASSO' 'BSS'};
FMList = cellfun(@str2func, FMList, 'UniformOutput', false);
NoModels = length(FMList);

% Tags for the portfolios under the different factor models
tags = {'OLS portfolio' 'FF portfolio' 'LASSO portfolio' 'BSS portfolio'};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3. Construct and rebalance your portfolios
%
% Here you will estimate your input parameters (exp. returns and cov. 
% matrix etc) from the Fama-French factor models. You will have to 
% re-estimate your parameters at the start of each rebalance period, and 
% then re-optimize and rebalance your portfolios accordingly. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initiate counter for the number of observations per investment period
toDay = 0;

% Preallocate the space for the per period value of the portfolios 
currentVal = zeros(NoPeriods, NoModels);

%--------------------------------------------------------------------------
% Set the value of lambda and K for the LASSO and BSS models, respectively
%--------------------------------------------------------------------------
lambda = 0.029;
K      = 9;

for t = 1 : NoPeriods
  
    % Subset the returns and factor returns corresponding to the current
    % calibration period.
    periodReturns = table2array( returns( calStart <= dates & dates <= calEnd, :) );
    periodFactRet = table2array( factorRet( calStart <= dates & dates <= calEnd, :) );
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
    
    % Calculate 'mu' and 'Q' using the 4 factor models.
    % Note: You need to write the code for the 4 factor model functions. 
    for i = 1 : NoModels
        
        [mu{i}, Q{i}, B{i}, errors{i}] = FMList{i}(periodReturns, periodFactRet, lambda, K);
        
    end

    % Calculate adjusted coefficient of determination for 4 factor models.
    NoAssets = size(periodReturns,2);
    coeff_det = zeros(NoAssets, NoModels);
    adj_coeff_det = zeros(NoAssets, NoModels);
    for i = 1 : NoModels
        B_non_zero_LASSO = sum(abs(B{3}) > 1e-10);
        for j = 1 : NoAssets
            N = size(periodReturns,1);
            p = [size(periodFactRet,2), 3, B_non_zero_LASSO(j), K];
            coeff_det(j,i) = 1 - sum(errors{i}(:,j).^2, 'all')/sum((periodReturns(:,j)-mean(periodReturns(:,j))).^2, 'all');
            adj_coeff_det(j,i) = 1 - ((N-1)/(N-p(i)-1))* (1 - coeff_det(j,i));
        end
    end
    coeff_dets{t} = coeff_det;
    adj_coeff_dets{t} = adj_coeff_det;
            
    % Optimize your portfolios to get the weights 'x'
    % Note: You need to write the code for MVO with no short sales
    for i = 1 : NoModels
        
        % Define the target return as the geometric mean of the market 
        % factor for the current calibration period
        targetRet = geomean(periodFactRet(:,1) + 1) - 1;
        
        x{i}(:,t) = MVO(mu{i}, Q{i}, targetRet); 
            
    end
    
    % Calculate the optimal number of shares of each stock you should hold
    for i = 1 : NoModels
        
        % Number of shares your portfolio holds per stock
        NoShares{i} = x{i}(:,t) .* currentVal(t,i) ./ currentPrices;
        
        % Weekly portfolio value during the out-of-sample window
        portfValue(fromDay:toDay,i) = periodPrices * NoShares{i};
        
        %------------------------------------------------------------------
        % Calculate your transaction costs for the current rebalance
        % period. The first period does not have any cost since you are
        % constructing the portfolios for the first time. 
        
        if t ~= 1
            
            tFee = 0.05;
            tCost(t-1,i) = tFee * sum(abs(NoShares{i} - NoSharesOld{i}) .* currentPrices);
            
        end
        
        NoSharesOld{i} = NoShares{i};
        %------------------------------------------------------------------
    end

    % Standard deviation of the period returns for each asset
    periodReturns_cov = cov(periodReturns);
    periodReturns_std{t} = diag(periodReturns_cov).^2;

    % Update your calibration and out-of-sample test periods
    calStart = calStart + calyears(1);
    calEnd   = calStart + calyears(4) - days(1);
    
    testStart = testStart + calyears(1);
    testEnd   = testStart + calyears(1) - days(1);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4. Results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%--------------------------------------------------------------------------
% 4.1 Evaluate any measures of fit of the regression models to assess their
% in-sample quality. You may want to modify Section 3 of this program to
% calculate the quality of fit each time the models are recalibrated.
%--------------------------------------------------------------------------

% Adjusted coefficients of determination of each asset per period are
% calculated in Section 3.

% Average Adjusted Coefficient of Determination
for i = 1 : NoPeriods
    fprintf('Adjusted Coefficient of Determination for Period %i\n', i);
    % disp(adj_coeff_dets{i});
    adj_coeff_dets_avg{i} = mean(adj_coeff_dets{i});
    disp(adj_coeff_dets_avg{i});
end

% Weighted Average Adjusted Coefficient of Determination
adj_coeff_dets_weighted = zeros(NoModels, NoPeriods);
for i = 1 : NoModels
    for j = 1 : NoPeriods
        for k = 1 : NoAssets
            adj_coeff_dets_weighted(i,j) = adj_coeff_dets_weighted(i,j) + x{i}(k,j)*adj_coeff_dets{j}(k,i);
        end
    end
end

%--------------------------------------------------------------------------
% 4.2 Calculate the portfolio average return, variance (or standard 
% deviation), and any other performance and/or risk metric you wish to 
% include in your report.
%--------------------------------------------------------------------------
N = size(portfValue,1);
portf_returns = (portfValue(2:end,:) ./ portfValue(1:end-1,:));
portf_mu = geomean(portf_returns + 1) - 1;
portf_Q  = cov(portf_returns);
portf_std = ((N*(diag(portf_Q))).^0.5).';  % monthly portfolio standard deviation
sharpe_ratio = (portf_mu.^N - 1)./portf_std;

portfValue_period = [initialVal*ones(1,NoModels); portfValue(N/NoPeriods:N/NoPeriods:end,:)];
portf_returns_period = (portfValue_period(2:end,:) ./ portfValue_period(1:end-1,:)) - 1;

%--------------------------------------------------------------------------
% 4.3 Plot the portfolio wealth evolution 
% 
% Note: The code below plots all portfolios onto a single plot. However,
% you may want to split this into multiple plots for clarity, or to
% compare a subset of the portfolios. 
%--------------------------------------------------------------------------
plotDates = dates(dates >= datetime('2012-01-01') );

fig1 = figure(1);

for i = 1 : NoModels
    
    plot( plotDates, portfValue(:,i) )
    hold on
    
end

legend(tags, 'Location', 'eastoutside','FontSize',12);
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
% print(fig1,'fileName','-dpdf','-r0');

% If you want to save the figure as .png for use in MS Word
print(fig1,'Wealth Evolution','-dpng','-r0');

%--------------------------------------------------------------------------
% 4.4 Plot the portfolio weights period-over-period
%--------------------------------------------------------------------------

% OLS portfolio Plot
fig2 = figure(2);
area(x{1}')
legend(tickers, 'Location', 'eastoutside','FontSize',12);
title('OLS portfolio weights', 'FontSize', 14)
ylabel('Weights','interpreter','latex','FontSize',12);
xlabel('Rebalance period','interpreter','latex','FontSize',12);

% Define the plot size in inches
set(fig2,'Units','Inches', 'Position', [0 0 8, 5]);
pos1 = get(fig2,'Position');
set(fig2,'PaperPositionMode','Auto','PaperUnits','Inches',...
    'PaperSize',[pos1(3), pos1(4)]);

% If you want to save the figure as .pdf for use in LaTeX
% print(fig2,'fileName2','-dpdf','-r0');

% If you want to save the figure as .png for use in MS Word
print(fig2,'OLS Portfolio Weights','-dpng','-r0');

% FF portfolio Plot
fig3 = figure(3);
area(x{2}')
legend(tickers, 'Location', 'eastoutside','FontSize',12);
title('FF portfolio weights', 'FontSize', 14)
ylabel('Weights','interpreter','latex','FontSize',12);
xlabel('Rebalance period','interpreter','latex','FontSize',12);

% Define the plot size in inches
set(fig3,'Units','Inches', 'Position', [0 0 8, 5]);
pos1 = get(fig3,'Position');
set(fig2,'PaperPositionMode','Auto','PaperUnits','Inches',...
    'PaperSize',[pos1(3), pos1(4)]);

% If you want to save the figure as .pdf for use in LaTeX
% print(fig2,'fileName3','-dpdf','-r0');

% If you want to save the figure as .png for use in MS Word
print(fig3,'FF Portfolio Weights','-dpng','-r0');

% LASSO portfolio Plot
fig4 = figure(4);
area(x{3}')
legend(tickers, 'Location', 'eastoutside','FontSize',12);
title('LASSO portfolio weights', 'FontSize', 14)
ylabel('Weights','interpreter','latex','FontSize',12);
xlabel('Rebalance period','interpreter','latex','FontSize',12);

% Define the plot size in inches
set(fig4,'Units','Inches', 'Position', [0 0 8, 5]);
pos1 = get(fig4,'Position');
set(fig4,'PaperPositionMode','Auto','PaperUnits','Inches',...
    'PaperSize',[pos1(3), pos1(4)]);

% If you want to save the figure as .pdf for use in LaTeX
% print(fig2,'fileName4','-dpdf','-r0');

% If you want to save the figure as .png for use in MS Word
print(fig4,'LASSO Portfolio Weights','-dpng','-r0');

% BSS portfolio Plot
fig5 = figure(5);
area(x{4}')
legend(tickers, 'Location', 'eastoutside','FontSize',12);
title('BSS portfolio weights', 'FontSize', 14)
ylabel('Weights','interpreter','latex','FontSize',12);
xlabel('Rebalance period','interpreter','latex','FontSize',12);

% Define the plot size in inches
set(fig5,'Units','Inches', 'Position', [0 0 8, 5]);
pos1 = get(fig5,'Position');
set(fig5,'PaperPositionMode','Auto','PaperUnits','Inches',...
    'PaperSize',[pos1(3), pos1(4)]);

% If you want to save the figure as .pdf for use in LaTeX
% print(fig2,'fileName5','-dpdf','-r0');

% If you want to save the figure as .png for use in MS Word
print(fig5,'BSS Portfolio Weights','-dpng','-r0');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Program End