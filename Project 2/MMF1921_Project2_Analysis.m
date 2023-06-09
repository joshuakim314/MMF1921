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
assetData  = 'MMF1921_AssetPrices_3.csv';
factorData = 'MMF1921_FactorReturns_3.csv';

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

% parameters
lambda = 0.03;
K = 5;
NoSims = 10000;
% alpha = 0.95;

% Factor models: OLS, LASSO, BSS, PCA
FMList = {'OLS' 'LASSO' 'BSS' 'PCA'};
% FMList = {'OLS' 'LASSO' 'PCA'};  % without gurobi
FMList = cellfun(@str2func, FMList, 'UniformOutput', false);
NoFactorModels = length(FMList);

% Tags for the portfolios under the different factor models
FMtags = {'OLS' 'LASSO' 'BSS' 'PCA'};

% Number of models: i) MVO, ii) CVaR Opt with normality
%                   iii) risk parity, iv) Sharpe ratio maximization
% MCList   = {'MCNorm' 'MCHM'};
% MCList   = cellfun(@str2func, MCList, 'UniformOutput', false);
optList  = {'MVO' 'CVaR' 'RP' 'MaxSR'};
optList  = cellfun(@str2func, optList, 'UniformOutput', false);
NoModels = length(optList);

% Tags for the portfolios under the different formulations
tags = {'MVO' 'CVaR' 'RP' 'MaxSR'};

% Tags for pairs of factor models and optimization models
for i = 1 : NoFactorModels
    for j = 1 : NoModels
        Pairtags{(i-1)*NoModels + j} = append(FMtags{i},' + ',tags{j});
    end
end

% for regime detection
avg_vols = zeros(NoPeriods, 1);

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
    
    % regime detection
    periodQ = cov(periodReturns(end-5:end,:));
    avg_vol = trace(periodQ) / n;
    avg_vols(t, 1) = avg_vol;

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
        % if i == 4
        %     [var_explained(t,:), cumul_var_explained(t,:), H_sw(t,:), H_ad(t,:), S_diff(t,:), K_diff(t,:)] = PCA_analysis(periodReturns(end-59:end,:), 3, t); 
        % end
        if i == 1
            alpha = 0.9;
            targetRet_percentile = 0.9;
        elseif i == 2
            alpha = 0.9;
            targetRet_percentile = 0.7;
        elseif i == 3
            alpha = 0.95;
            targetRet_percentile = 0.7;
        elseif i == 4
            alpha = 0.95;
            targetRet_percentile = 0.7;
        end
        for j = 1 : NoModels
            ij = (i-1)*NoModels + j;
            if j == 2
                % targetRet_percentile = 1.0;
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
portfValue = portfValue';

% Measure runtime: stop the clock
toc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3. Results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%--------------------------------------------------------------------------
% 3.1 Calculate the portfolio average return, standard deviation, Sharpe
% ratio and average turnover.
%--------------------------------------------------------------------------

% Calculate the observed portfolio returns
portfRets = portfValue(:, 2:end) ./ portfValue(:, 1:end-1) - 1;

% Calculate the portfolio excess returns
portfExRets = portfRets' - table2array(riskFree(dates >= datetime(returns.Properties.RowNames{1}) + calyears(5) + calmonths(1),: ));

% Calculate the portfolio Sharpe ratio 
SR = (geomean(portfExRets + 1) - 1) ./ std(portfExRets);
% SR = (geomean(portfExRets(42:60,:) + 1) - 1) ./ std(portfExRets(42:60,:));
% SR = (geomean(cat(1, portfExRets(1:41,:), portfExRets(61:end,:)) + 1) - 1) ./ std(cat(1, portfExRets(1:41,:), portfExRets(61:end,:)));

% Calculate the average turnover rate
avgTurnover = mean(turnover(2:end, :));
% avgTurnover = mean(turnover(7:10, :));
% avgTurnover = mean(cat(1, turnover(1:6,:), turnover(11:end,:)));

% Print Sharpe ratio and Avg. turnover to the console
disp(['Sharpe ratio : [' num2str(SR(:).') ']']) ;
disp(['Avg. turnover: [' num2str(avgTurnover(:).') ']']) ;
display(0.8*SR - 0.2*avgTurnover)
% disp(['Sharpe ratio: ', num2str(SR)]);
% disp(['Avg. turnover: ', num2str(avgTurnover)]);

%--------------------------------------------------------------------------
% 3.2 Portfolio wealth evolution plot
%--------------------------------------------------------------------------
%%
% Calculate the dates of the out-of-sample period
plotDates = dates(dates >= datetime(returns.Properties.RowNames{1}) + calyears(5) );

fig1 = figure(1);
% plot(plotDates, portfValue)

for i = 1 : NoFactorModels
    for j = 1 : NoModels
        ij = (i-1)*NoModels + j;
        plot(plotDates, portfValue(ij,:))
        hold on
    end
end

legend(Pairtags, 'Location', 'eastoutside','FontSize',12);
% legend({'CI = 70%' 'CI = 90%' 'CI = 95%' 'CI = 99%'}, 'Location', 'eastoutside','FontSize',12);
datetick('x','dd-mmm-yyyy','keepticks','keeplimits');
set(gca,'XTickLabelRotation',30);
title('Portfolio wealth evolution', 'FontSize', 14)
ylabel('Total wealth','interpreter','latex','FontSize',12);

% Define the plot size in inches
set(fig1,'Units','Inches', 'Position', [0 0 8, 5]);
pos1 = get(fig1,'Position');
set(fig1,'PaperPositionMode','Auto','PaperUnits','Inches',...
    'PaperSize',[pos1(3), pos1(4)]);

% If you want to save the figure as .pdf for use in LaTeX
% print(fig1,'fileName','-dpdf','-r0');

% If you want to save the figure as .png for use in MS Word
print(fig1,'figures/wealth_evolution','-dpng','-r0');

%--------------------------------------------------------------------------
% 3.3 Portfolio weights plot
%--------------------------------------------------------------------------
for i = 1 : NoFactorModels
    for j = 1 : NoModels
        ij = (i-1)*NoModels + j;
        % Portfolio weights
        fig2 = figure(2);
        area(x{ij}')
        legend(tickers, 'Location', 'eastoutside','FontSize',12);
        title(append(Pairtags{ij},' Portfolio weights'), 'FontSize', 14)
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
        print(fig2,append('figures/',Pairtags{ij},' Composition'),'-dpng','-r0');
    end
end

% % Portfolio weights
% fig2 = figure(2);
% area(x')
% legend(tickers, 'Location', 'eastoutside','FontSize',12);
% title('Portfolio weights', 'FontSize', 14)
% ylabel('Weights','interpreter','latex','FontSize',12);
% xlabel('Rebalance period','interpreter','latex','FontSize',12);
% 
% % Define the plot size in inches
% set(fig2,'Units','Inches', 'Position', [0 0 8, 5]);
% pos1 = get(fig2,'Position');
% set(fig2,'PaperPositionMode','Auto','PaperUnits','Inches',...
%     'PaperSize',[pos1(3), pos1(4)]);
% 
% % If you want to save the figure as .pdf for use in LaTeX
% % print(fig2,'fileName2','-dpdf','-r0');
% 
% % If you want to save the figure as .png for use in MS Word
% print(fig2,'fileName2','-dpng','-r0');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Program End