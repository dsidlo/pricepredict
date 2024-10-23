"""
File: pricepredict_rt.py

This script generates a near real-time price prediction for a given symbol.
This script generates a chart every hour.
It tracks each prediction to a CSV file.
"""

from pricepredict import PricePredict
from datetime import datetime, timedelta
import time
import sys

fn main():

    print("sys.path: ",sys.path)
    try:
        symbol = 'FOLD'
        pp = PricePredict(symbol, period=PricePredict.Period5min)
        # start_date = today - 14 days
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        # end_date = today
        end_date = datetime.now().strftime("%Y-%m-%d")
        pp.cache_training_data(symbol, start_date, end_date, pp.period)
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        pp.cache_prediction_data(symbol, start_date, end_date, pp.period)
        pp.force_training = True
        pp.cached_train_predict_report()

        while True:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"Prediction for {symbol} at {current_time}: {pp}")
            pp.predict_price(None)
            # Get the latest date_time from pp.date_data
            latest_date_time = pp.date_data[len(pp.date_data)-1]
            # Format the date_time to be used in the chart file name
            ldt = latest_date_time.strftime("%Y-%m-%d_%H-%M")
            pp.chart_dir = f"./rt_charts/"
            pp.gen_prediction_chart(show_plot=True, last_candles=150)
            time.sleep(1)  # Sleep for 15 minutes, then predict again
    except RuntimeError as e:
        print(f"Error: {e}")



