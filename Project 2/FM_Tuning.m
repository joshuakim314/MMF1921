%% MMF1921 (2023) - Project 2
% 
% The purpose of this program is to provide a template with which to
% develop Project 2. The project requires you to test different models 
% (and/or different model combinations) to create an asset management
% algorithm. 

% This template will be used by the instructor and TA to assess your  
% trading algorithm using different datasets.

% PLEASE DO NOT MODIFY THIS TEMPLATE

clc
clear all
format short

% Program Start
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. Read input files 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Input file names
assetData  = 'MMF1921_AssetPrices_1.csv';
factorData = 'MMF1921_FactorReturns_1.csv';

% Initial budget to invest ($100,000)
initialVal = 100000;

% Length of investment period (in months)
investPeriod = 6;

% Load the stock weekly prices
adjClose = readtable(assetData);
adjClose.Properties.RowNames = cellstr(datetime(adjClose.Date));
adjClose.Properties.RowNames = cellstr(datetime(adjClose.Properties.RowNames));
adjClose.Date = [];

% Load the factors weekly returns
factorRet = readtable(factorData);
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

% Start of out-of-sample test period 
testStart = datetime(returns.Properties.RowNames{1}) + calyears(5);

% End of the first investment period
testEnd = testStart + calmonths(investPeriod) - days(1);

% End of calibration period (note that the start date is the first
% observation in the dataset)
calEnd = testStart - days(1);

% Total number of investment periods
NoPeriods = ceil( days(datetime(returns.Properties.RowNames{end}) - testStart) / (30.44*investPeriod) );

% Number of assets      
n = size(adjClose,2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2. Run your program
% 
% This section will run your Project1_Function in a loop. The data will be
% loaded progressively as a growing window of historical observations.
% Rebalancing will take place after every loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nn = 21;
lambdas = linspace(0.03,0.07,nn);
% nn = 10;
% lambdas = linspace(0.01,0.1,nn);
lambda_i = 0;

% nn = 4;
% Ks = [2, 3, 4, 5];
% K_i = 0;
% portfValues_K = zeros(4, NoPeriods*6);

r_sqs = zeros(nn, NoPeriods);
adj_r_sqs = zeros(nn, NoPeriods);
Bs_non_zero = zeros(nn, NoPeriods);

for lambda = lambdas
    lambda_i = lambda_i + 1;

% for K = Ks
%     K_i = K_i + 1;

    % Start of out-of-sample test period 
    testStart = datetime(returns.Properties.RowNames{1}) + calyears(5);
    
    % End of the first investment period
    testEnd = testStart + calmonths(investPeriod) - days(1);
    
    % End of calibration period (note that the start date is the first
    % observation in the dataset)
    calEnd = testStart - days(1);
    
    % Total number of investment periods
    NoPeriods = ceil( days(datetime(returns.Properties.RowNames{end}) - testStart) / (30.44*investPeriod) );
    
    % Number of assets      
    n = size(adjClose,2);
        
    % parameters
    % lambda = 0.03;
    K = 5;
    NoSims = 10000;
    alpha = 0.95;
    
    % Factor models: OLS, LASSO, BSS, PCA
    FMList = {'LASSO'};  % without gurobi
    FMList = cellfun(@str2func, FMList, 'UniformOutput', false);
    NoFactorModels = length(FMList);
    
    % Tags for the portfolios under the different factor models
    FMtags = {'LASSO'};
    
    % Number of models: i) MVO, ii) CVaR Opt with normality
    %                   iii) risk parity, iv) Sharpe ratio maximization
    % MCList   = {'MCNorm' 'MCHM'};
    % MCList   = cellfun(@str2func, MCList, 'UniformOutput', false);
    optList  = {'MVO'};
    optList  = cellfun(@str2func, optList, 'UniformOutput', false);
    NoModels = length(optList);
    
    % Tags for the portfolios under the different formulations
    tags = {'MVO'};
    
    % Tags for pairs of factor models and optimization models
    for i = 1 : NoFactorModels
        for j = 1 : NoModels
            Pairtags{(i-1)*NoModels + j} = append(FMtags{i},' + ',tags{j});
        end
    end

    % Preallocate space for the portfolio per period value and turnover
    currentVal = zeros(NoPeriods, NoFactorModels*NoModels);
    turnover   = zeros(NoPeriods, NoFactorModels*NoModels);
    
    % Initiate counter for the number of observations per investment period
    toDay = 0;
    
    % Meaure runtime: start the clock
    tic
    
    for t = 1 : NoPeriods
      
        % Subset the returns and factor returns corresponding to the current
        % calibration period.
        periodReturns = table2array( returns( dates <= calEnd, :) );
        periodFactRet = table2array( factorRet( dates <= calEnd, :) );
        currentPrices = table2array( adjClose( ( calEnd - calmonths(1) - days(5) ) <= dates & dates <= calEnd, :) )';
        
        % Subset the prices corresponding to the current out-of-sample test 
        % period.
        periodPrices = table2array( adjClose( testStart <= dates & dates <= testEnd,:) );
        
        % Set the initial value of the portfolio or update the portfolio value
        if t == 1
            currentVal(t,:) = initialVal;
        else
            for i = 1 : NoFactorModels
                for j = 1 : NoModels
                    ij = (i-1)*NoModels + j;
                    currentVal(t,ij) = currentPrices' * NoShares{ij};
    
                    % Store the current asset weights (before optimization takes place)
                    x0{ij}(:,t) = (currentPrices .* NoShares{ij}) ./ currentVal(t,ij);
                end
            end
        end
        
        %----------------------------------------------------------------------
        % Portfolio optimization
        % You must write code your own algorithmic trading function 
        %----------------------------------------------------------------------
        for i = 1 : NoFactorModels
            [mu, Q, B, errors, D, V, F] = FMList{i}(periodReturns(end-59:end,:), periodFactRet(end-59:end,:), lambda, K);

            % Calculate adjusted coefficient of determination for factor model
            r_sq = zeros(n, 1);
            adj_r_sq = zeros(n, 1);
            B_non_zero = sum(abs(B) > 1e-10);
            Bs_non_zero(lambda_i,t) = sum((B_non_zero >= 2) & (B_non_zero <= 5));
            for j = 1 : n
                N = size(periodReturns,1);
                p = B_non_zero(j);
                r_sq(j,1) = 1 - sum(errors(:,j).^2, 'all')/sum((periodReturns(:,j)-mean(periodReturns(:,j))).^2, 'all');
                adj_r_sq(j,1) = 1 - ((N-1)/(N-p(1)-1))* (1 - r_sq(j,1));
            end
            r_sqs(lambda_i, t) = mean(r_sq);
            adj_r_sqs(lambda_i, t) = mean(adj_r_sq);
            % r_sqs(K_i, t) = mean(r_sq);
            % adj_r_sqs(K_i, t) = mean(adj_r_sq);

            for j = 1 : NoModels
                ij = (i-1)*NoModels + j;
                if j == 2
                    targetRet_percentile = 1.0;
                    targetRet = targetRet_percentile * mean(mu);
                    scenarios = MCNorm(mu, D, V, F, NoSims);
                    x{ij}(:, t) = optList{j}(scenarios, mu, targetRet, alpha);
                else
                    x{ij}(:, t) = optList{j}(mu, Q);
                end
            end
        end
    
        % x(:,t) = Project2_Function(periodReturns, periodFactRet, x0(:,t));
    
        % Calculate the turnover rate 
        if t > 1
            for i = 1 : NoFactorModels
                for j = 1 : NoModels
                    ij = (i-1)*NoModels + j;
                    turnover(t,ij) = sum( abs( x{ij}(:,t) - x0{ij}(:,t) ) );
                end
            end
            % turnover(t) = sum( abs( x(:,t) - x0(:,t) ) );
        end
    
        % Update counter for the number of observations per investment period
        fromDay = toDay + 1;
        toDay   = toDay + size(periodPrices,1);
    
        for i = 1 : NoFactorModels
            for j = 1 : NoModels
                ij = (i-1)*NoModels + j;
                NoShares{ij} = x{ij}(:, t) .* currentVal(t, ij) ./ currentPrices;
                portfValue(fromDay:toDay,ij) = periodPrices * NoShares{ij};
            end
        end
        
        % Number of shares your portfolio holds per stock
        % NoShares = x(:,t) .* currentVal(t) ./ currentPrices;
    
        % Weekly portfolio value during the out-of-sample window
        % portfValue(fromDay:toDay) = periodPrices * NoShares;
    
        % Update your calibration and out-of-sample test periods
        testStart = testStart + calmonths(investPeriod);
        testEnd   = testStart + calmonths(investPeriod) - days(1);
        calEnd    = testStart - days(1);
    
    end
    
    % Transpose the portfValue into a column vector
    % portfValue = portfValue';
    % portfValues_K(K_i,:) = portfValue;
    
    % Measure runtime: stop the clock
    toc

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3. Results - LASSO
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Calculate the dates of the out-of-sample period
plotDates = dates(dates >= datetime(returns.Properties.RowNames{1}) + calyears(5) );

fig1 = figure(1);
% plot(plotDates, portfValue)

for lambda_i = 1 : nn
    plot(plotDates(6:6:end), adj_r_sqs(lambda_i,:))
    hold on
end

legend("$\lambda =$ " + string(lambdas), 'interpreter', 'latex', 'Location', 'eastoutside','FontSize',12);
datetick('x','dd-mmm-yyyy','keepticks','keeplimits');
set(gca,'XTickLabelRotation',30);
h = title('Average $R^2_{adj}$ for Varying $\lambda$', 'interpreter', 'latex', 'FontSize', 14);
ylabel('$R^2_{adj}$','interpreter','latex','FontSize',12);

% Define the plot size in inches
set(fig1,'Units','Inches', 'Position', [0 0 8, 5]);
pos1 = get(fig1,'Position');
set(fig1,'PaperPositionMode','Auto','PaperUnits','Inches',...
    'PaperSize',[pos1(3), pos1(4)]);

% If you want to save the figure as .pdf for use in LaTeX
% print(fig1,'fileName','-dpdf','-r0');

% If you want to save the figure as .png for use in MS Word
print(fig1,'figures/adj_r_sqs_lambda','-dpng','-r0');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4. Results - BSS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%--------------------------------------------------------------------------
% 4.1 Calculate the portfolio average return, standard deviation, Sharpe
% ratio and average turnover.
%--------------------------------------------------------------------------

% Calculate the observed portfolio returns
portfRets = portfValues_K(:, 2:end) ./ portfValues_K(:, 1:end-1) - 1;

% Calculate the portfolio excess returns
portfExRets = portfRets' - table2array(riskFree(dates >= datetime(returns.Properties.RowNames{1}) + calyears(5) + calmonths(1),: ));

% Calculate the portfolio Sharpe ratio 
SR = (geomean(portfExRets + 1) - 1) ./ std(portfExRets);

% Calculate the average turnover rate
avgTurnover = mean(turnover(2:end, :));

% Print Sharpe ratio and Avg. turnover to the console
disp(['Sharpe ratio : [' num2str(SR(:).') ']']) ;
disp(['Avg. turnover: [' num2str(avgTurnover(:).') ']']) ;
% disp(['Sharpe ratio: ', num2str(SR)]);
% disp(['Avg. turnover: ', num2str(avgTurnover)]);

%--------------------------------------------------------------------------
% 4.2 Portfolio wealth evolution plot
%--------------------------------------------------------------------------

% Calculate the dates of the out-of-sample period
plotDates = dates(dates >= datetime(returns.Properties.RowNames{1}) + calyears(5) );

fig1 = figure(1);
% plot(plotDates, portfValue)

for K = Ks
    plot(plotDates, portfValues_K(K-1,:))
    hold on
end

legend("K = " + string(Ks), 'Location', 'eastoutside','FontSize',12);
datetick('x','dd-mmm-yyyy','keepticks','keeplimits');
set(gca,'XTickLabelRotation',30);
h = title('Portfolio wealth evolution for Varying $K$', 'interpreter', 'latex', 'FontSize', 14);
ylabel('Total wealth','interpreter','latex','FontSize',12);

% Define the plot size in inches
set(fig1,'Units','Inches', 'Position', [0 0 8, 5]);
pos1 = get(fig1,'Position');
set(fig1,'PaperPositionMode','Auto','PaperUnits','Inches',...
    'PaperSize',[pos1(3), pos1(4)]);

% If you want to save the figure as .pdf for use in LaTeX
% print(fig1,'fileName','-dpdf','-r0');

% If you want to save the figure as .png for use in MS Word
print(fig1,'figures/wealth_evolution_K','-dpng','-r0');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Program End