clc
clear all
format short

nn = 31;
Bs_non_zero_LASSO = zeros(nn, 21);
i = 0;

for lambda = linspace(0.02,0.035,nn)
    
    i = i + 1;

    % Load the stock weekly prices
    adjClose = readtable('MMF1921_AssetPrices_1.csv');
    adjClose.Properties.RowNames = cellstr(datetime(adjClose.Date));
    adjClose.Properties.RowNames = cellstr(datetime(adjClose.Properties.RowNames));
    adjClose.Date = [];

    % Load the factors weekly returns
    factorRet = readtable('MMF1921_FactorReturns_1.csv');
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

    FMList = {'LASSO'};
    FMList = cellfun(@str2func, FMList, 'UniformOutput', false);

    % Initiate counter for the number of observations per investment period
    toDay = 0;

    % Preallocate the space for the per period value of the portfolios 
    currentVal = zeros(NoPeriods, 1);

    K = 4;

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

            currentVal(t,1) = initialVal;

        else
            currentVal(t,1) = currentPrices' * NoShares{1};
        end

        % Update counter for the number of observations per investment period
        fromDay = toDay + 1;
        toDay   = toDay + size(periodPrices,1);

        % Calculate 'mu' and 'Q' using the 4 factor models.
        % Note: You need to write the code for the 4 factor model functions. 
        [mu{1}, Q{1}, B{1}, errors{1}] = FMList{1}(periodReturns, periodFactRet, lambda, K);

        % Calculate adjusted coefficient of determination for 4 factor models.
        NoAssets = size(periodReturns,2);
        coeff_det = zeros(NoAssets, 1);
        adj_coeff_det = zeros(NoAssets, 1);
        B_non_zero_LASSO = sum(abs(B{1}) > 1e-10);
        Bs_non_zero_LASSO(i,1:20) = B_non_zero_LASSO;
        Bs_non_zero_LASSO(i,21) = lambda;
        for j = 1 : NoAssets
            N = size(periodReturns,1);
            p = B_non_zero_LASSO(j);
            coeff_det(j,1) = 1 - sum(errors{1}(:,j).^2, 'all')/sum((periodReturns(:,j)-mean(periodReturns(:,j))).^2, 'all');
            adj_coeff_det(j,1) = 1 - ((N-1)/(N-p(1)-1))* (1 - coeff_det(j,1));
        end
        coeff_dets{t} = coeff_det;
        adj_coeff_dets{t} = adj_coeff_det;

        % Optimize your portfolios to get the weights 'x'
        % Note: You need to write the code for MVO with no short sales
        % Define the target return as the geometric mean of the market 
        % factor for the current calibration period
        targetRet = geomean(periodFactRet(:,1) + 1) - 1;

        x{1}(:,t) = MVO(mu{1}, Q{1}, targetRet); 

        % Calculate the optimal number of shares of each stock you should hold
        % Number of shares your portfolio holds per stock
        NoShares{1} = x{1}(:,t) .* currentVal(t,1) ./ currentPrices;

        % Weekly portfolio value during the out-of-sample window
        portfValue(fromDay:toDay,1) = periodPrices * NoShares{1};

        %------------------------------------------------------------------
        % Calculate your transaction costs for the current rebalance
        % period. The first period does not have any cost since you are
        % constructing the portfolios for the first time. 

        if t ~= 1

            tFee = 0.05;
            tCost(t-1,1) = tFee * sum(abs(NoShares{1} - NoSharesOld{1}) .* currentPrices);

        end

        NoSharesOld{1} = NoShares{1};
        %------------------------------------------------------------------

        % Update your calibration and out-of-sample test periods
        calStart = calStart + calyears(1);
        calEnd   = calStart + calyears(4) - days(1);

        testStart = testStart + calyears(1);
        testEnd   = testStart + calyears(1) - days(1);

    end
    
end
cardinality = (Bs_non_zero_LASSO >= 2) & (Bs_non_zero_LASSO <= 5);
disp(sum(cardinality,2));
