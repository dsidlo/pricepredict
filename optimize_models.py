from pricepredict import PricePredict
import concurrent.futures as cf
import json

def tst_prediction_analysis(in_ticker):
    # Create an instance of the price prediction object
    pp = PricePredict(model_dir='./models/',
                      chart_dir='./charts/',
                      preds_dir='./predictions/')

    ticker = in_ticker
    test_ticker = "Test-" + ticker
    # Data download dates
    start_date = "2015-01-01"
    end_date = "2023-12-31"
    pred_start_date = "2024-01-01"
    pred_end_date = "2024-10-25"

    pp.cache_training_data(ticker, start_date, end_date, PricePredict.PeriodDaily)
    pp.cache_prediction_data(ticker, pred_start_date, pred_end_date, PricePredict.PeriodDaily)
    pp.cached_train_predict_report(save_plot=False, show_plot=True)

    return pp


def tst_bayes_opt(in_ticker, pp_obj=None, opt_csv=None,
                  only_fetch_opt_data=False, do_optimize=False,
                  cache_train=False, cache_predict=False, train_and_predict=False):
    if pp_obj is None:
        # Create an instance of the price prediction object
        pp = PricePredict(model_dir='./models/',
                          chart_dir='./charts/',
                          preds_dir='./predictions/')
    else:
        pp = pp_obj

    # Load data from Yahoo Finance
    ticker = in_ticker
    # Training Data
    start_date = "2015-01-01"
    end_date = "2023-12-31"
    # Prediction Data
    pred_start_date = "2024-01-01"
    pred_end_date = "2024-10-25"

    if only_fetch_opt_data:
        data, pp.features = pp.fetch_data_yahoo(ticker, start_date, end_date)

        # Augment the data with additional indicators/features
        if data is None:
            print(f"'Close' column not found in {ticker}'s data. Skipping...")
            return None

        return pp

    if do_optimize:
        aug_data, features, targets, dates_data = pp.augment_data(pp.orig_data, 0)

        # Scale the data so the model can use it more effectively
        scaled_data, scaler = pp.scale_data(aug_data)

        # Prepare the scaled data for model inputs
        X, y = pp.prep_model_inputs(scaled_data, pp.features)

        # Use a small batch size and epochs to test the model training
        pp.epochs = 3
        pp.batch_size = 1

        # Train the model
        model, y_pred, mse = pp.train_model(X, y)

        # Perform Bayesian optimization
        pp.bayesian_optimization(X, y, opt_csv=opt_csv)

    if cache_train:
        pp.cache_training_data(ticker, start_date, end_date, PricePredict.PeriodDaily)
    if cache_predict:
        pp.cache_prediction_data(ticker, pred_start_date, pred_end_date, PricePredict.PeriodDaily)
    if train_and_predict:
        pp.cached_train_predict_report(save_plot=False, show_plot=True)

    return pp


def main():
    tickers = ['AAPL', 'ABT', 'ACN', 'ADBE', 'ADM', 'ADP', 'AIG', 'ALKS', 'ALL', 'AMGN', 'AMX', 'AMZN', 'ANTM.JK', 'AON',
               'APD', 'APTV', 'AVGO', 'AXON', 'AXP', 'BA', 'BAC', 'BAX', 'BKNG', 'BLK', 'BRK-B', 'CAT', 'CB', 'CCI', 'CI',
               'CL', 'CLDX', 'CLS', 'CLSK', 'CMCSA', 'CNC', 'CRM', 'CRSP', 'CSCO', 'CSX', 'CVX', 'DHR', 'DIS', 'DUK', 'ECL',
               'EGIO', 'EL', 'EMR', 'EOG', 'ERIC', 'EURUSD=X', 'EXAS', 'EXEL', 'FOLD', 'FXI', 'FXP', 'GBPUSD=X', 'GC=F',
               'GD', 'GE', 'GILD', 'GLPG', 'GOOG', 'GOOGL', 'HAE', 'HD', 'HON', 'IART', 'IBM', 'IMAX', 'INTC', 'INTU',
               'IRTC', 'ISRG', 'ITW', 'JAZZ', 'JD', 'JNJ', 'JPM', 'JPYUSD=X', 'KO', 'LIN', 'LIVN', 'LLY', 'LMT', 'LOW',
               'LRCX', 'LSCC', 'MA', 'MASI', 'MCD', 'MDLZ', 'MDT', 'META', 'MMM', 'MO', 'MRK', 'MRVL', 'MSFT', 'MTCH',
               'NEE', 'NFLX', 'NKE', 'NNDM', 'NOW', 'NVCR', 'NVDA', 'ORCL', 'PACB', 'PEP', 'PFE', 'PG', 'PLD', 'PNC', 'PSA',
               'PTCT', 'PYPL', 'QCOM', 'RTX', 'SBUX', 'SCHW', 'SEDG', 'SO', 'SPGI', 'SYK', 'SYY', 'T', 'TMO', 'TRV', 'TSLA',
               'TXN', 'UCTT', 'UNH', 'UPS', 'USB', 'V', 'VRTX', 'VZ', 'WM', 'WMT', 'WOLF', 'XAB=F', 'XAE=F', 'XAF=F',
               'XAI=F', 'XAK=F', 'XAU=F', 'XNCR', 'XOM', 'ZM', '^DJI', '^GSPC', '^IXIC', '^N225', '^XAX']

    already_optimized = {}
    with open(f"./optimize_models.json", 'r') as f:
        for line in f:
            opt_params = json.loads(line)
            sym = next(iter(opt_params['Ticker']))
            already_optimized[sym] = opt_params

    ticker_pp = {}
    futures = []
    with cf.ThreadPoolExecutor(8) as ex:
        for ticker in tickers:
            if ticker in already_optimized:
                continue
            # Sync: Pull in Training and Prediction Data for each Ticker
            print(f"Pulling Optimization data for {ticker}...")
            pp = tst_bayes_opt(ticker, only_fetch_opt_data=True)
            ticker_pp[ticker] = pp
        for ticker in tickers:
            if ticker in already_optimized:
                continue
            # Async: Optimize the Model's Hyperparameters for each Ticker
            print(f"Optimizing model for {ticker}...")
            pp = ticker_pp[ticker]
            kawrgs={'pp_obj': pp, 'do_optimize': True}
            future = ex.submit(tst_bayes_opt, ticker, **kawrgs)
            futures.append(future)
        print("Waiting for tasks to complete...")
        with open(f"./ticker_bopts.json", 'w') as f:
            for future in cf.as_completed(futures):
                try:
                    pp = future.result()
                except Exception as e:
                    print(f"Optimization for {ticker} generated an exception: {e}")
                else:
                    # Write out the optimized hyperparameters to a JSON file
                    f.write(f"{{ {pp.ticker}: {pp.opt_hypers} }}")
                    print(f"Completed Hyperparameters Optimization: {pp.ticker}")

main()
