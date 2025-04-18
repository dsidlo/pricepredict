"""
File: pricepredict_strat_01.py

In this strategy, we will use the PricePredict class to predict the price of a stock.

We will run prediction for the last 3 months of data:
  - test_start_date = 3 months ago test_end_date = today

This will require training the model on the prior 3 years:
  - train_start_date = pre_start_date - 3 years - 3 months
  - train_end_date = pre_start_date - 3 months

Trading Strategy:
  - Buy if the acuatl price opens above the predicted price
  - Sell if the actual price opens below the predicted price
  - Buy if the open price is near the predicted Low price
    - Need to determine what constitutes "near"
  - Sell if the open price is near the predicted High price
    - Need to determine what constitutes "near"

Exit Strategy:
  - Sell if the actual price on a long trade hits the predicted high price
  - Buy if the actual price on a short trade hits the predicted low price

Notes:
  - I was concerned with the adjusted predictions potentially slightly shifting the original prediction values
    once the actual price was determined. But, it does not look like that happens.

"""
from pricepredict import PricePredict
from datetime import datetime, timedelta
import time


def main():
    test_days = 90
    pred_days = 7
    train_days = 3 * 365

    # Calculate the start and end dates for training and testing periods
    train_end_date = datetime.now() - timedelta(days=test_days)
    train_start_date = train_end_date - timedelta(days=train_days)
    pred_end_date = datetime.now()
    pred_start_date = train_end_date - timedelta(days=pred_days) - timedelta(days=7)
    test_end_date = datetime.now()
    test_start_date = test_end_date - timedelta(days=test_days) - timedelta(days=7)
    
    # Convert dates to strings
    train_end_date_str = train_end_date.strftime("%Y-%m-%d")
    train_start_date_str = train_start_date.strftime("%Y-%m-%d")
    pred_end_date_str = pred_end_date.strftime("%Y-%m-%d")
    pred_start_date_str = pred_start_date.strftime("%Y-%m-%d")
    test_end_date_str = test_end_date.strftime("%Y-%m-%d")
    test_start_date_str = test_start_date.strftime("%Y-%m-%d")

    # Train the model
    ticker = 'AAPL'
    pp = PricePredict(ticker, period=PricePredict.PeriodDaily)
    pp.cache_training_data(ticker, train_start_date_str, train_end_date_str, pp.period)
    pp.cache_prediction_data(ticker, pred_start_date_str, pred_end_date_str, pp.period)
    pp.cached_train_predict_report()

    plot_predictions(pp, pred_start_date, pred_end_date)

    # Get the list of testings dates
    pp.fetch_data_yahoo(ticker=ticker, date_start=test_start_date_str, date_end=test_end_date_str)
    dates_list = list(pp.date_data[:len(pp.adj_pred_close)])

    cnt = 0
    for date in dates_list:
        end_date = date
        start_date = end_date - timedelta(days=300)
        pp.fetch_and_predict(None, start_date, end_date)
        plot_predictions(pp, start_date, end_date)
        cnt += 1
        if (cnt % 10) == 0:
            time.sleep(2)


def plot_predictions(pp, start_date, end_date):
    # View the test prediction results
    title = f"test_train_model: {pp.ticker} -- Period {pp.period} {end_date}"
    pred_close = pp.adj_pred_close
    pred_high = pp.adj_pred_high
    pred_low = pp.adj_pred_low
    close = list(pp.orig_data.iloc[15:len(pred_close)+13, 4])
    pp.plot_pred_results(close, None, None,
                         pred_close, pred_high, pred_low, title=title)

main()
