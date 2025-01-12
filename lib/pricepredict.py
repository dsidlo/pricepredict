"""
Class: PricPredict

When instantiated, this class will till stock date from Yahoo Finance, augment the data with technical indicators,
and train an LSTM model to predict the price of a stock.

See tests for examples of how to use this class.

"""
import asyncio
import datetime
import time
import ipywidgets as widgets
import keras
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import os
import sys
import pandas as pd
import pandas_ta as ta
import pydot
import re
import seaborn
import shap
import sklearn as skl
import sys
import tensorflow as tf
import yfinance as yf
import yfinance_cache as yfc
import logging
import statsmodels.api as sm
import json
import jsonify
import pydantic
import lzma
import dill

from dataclasses import dataclass
from decimal import Decimal
from io import StringIO
from ipywidgets import Output
from keras.callbacks import History
from keras import optimizers
from keras.callbacks import CSVLogger
from keras.layers import Dense
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.models import Model
from keras.models import Sequential
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from IPython.display import Image
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from pydantic import validate_arguments
from typing import Any, Dict, List, Optional, Union
from groq import Groq
from bayes_opt import BayesianOptimization
from silence_tensorflow import silence_tensorflow
from statsmodels.tsa.stattools import coint

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

"""
Notes: Regarding training...
Training the model should requires at least 2 years of prior data for a daily period. 
For other periodicies, that's from 600 to 800 prior periods.

Notes: Regarding prediction...
To perform a prediction, the model requires enough data to fill the back_candles period,
and enough data to fill have data for any added technical indicators.
"""


class DataCache():
    """
    DataCache Simple Properties only Class
    Moved away from pydantic to eliminate issues with pickling.
    """

    def __init__(self):
        self.symbol: str = ''
        self.dataStart: str = ''
        self.dataEnd: str = ''
        self.period: str = ''
        self.data: str = ''
        self.feature_cnt: int = None
        self.data_scaled: list[Decimal] = []
        self.target_cnt: int = None
        self.dates_data: str = ''
        self.split_pcnt: Decimal = None
        self.X: list[Decimal] = []
        self.y: list[Decimal] = []


# ================================
# This is the PricePredict class.
# ================================
class PricePredict():
    """
    PricePredict Class

    Concurrent Processing:

    - Pulling data from Yahoo Finance must be done in a synchronous manner,
      as we can only have one connection to Yahoo Finance at a time.
      But, pulling the required data is pretty fast.
      So we place the data pull in a separate method, and call it first
      with blocking, and Cache the data.

    - We can run the training and prediction in parallel, as they can be
      called on after the data has been cached into the object.
      We can do all of the parallel process in a single method, and
      have multiple object instances running in parallel doing training
      and prediction, charting and reporting on their respective cached data.
    """

    # Constants
    PeriodWeekly = 'W'
    PeriodDaily = 'D'
    Period1hour = '1h'
    Period5min = '5m'
    Period1min = '1m'
    PeriodValues = [PeriodWeekly, PeriodDaily,
                    Period1hour, Period5min, Period1min]

    # Periods between 5min and 1hour don't have enough data for training.
    PeriodMultiplier = {Period1hour: 7,
                        Period5min: 84,
                        Period1min: 420}

    def __init__(self,
                 ticker='',  # The ticker for the data
                 model_dir='./models/',  # The directory where the model is saved
                 chart_dir='./charts/',  # The directory where the charts are saved
                 preds_dir='./predictions/',  # The directory where the predictions are saved
                 period=PeriodDaily,  # The period for the data (D, W)
                 back_candles=15,  # The number of candles to look back for each period.
                 split_pcnt=0.8,  # The value for splitting data into training and testing.
                 batch_size=30,  # The batch size for training the model.
                 epochs=50,  # The number of epochs for training the model.
                 lstm_units=256,  # The current models units
                 lstm_dropout=0.2,  # The current models dropout
                 adam_learning_rate=0.035,  # The current models learning rate
                 shuffle=True,  # Shuffle the data for training the model.
                 val_split=0.1,  # The validation split used during training.
                 keras_verbosity=0,
                 verbose=True,  # Print debug information
                 logger=None,  # The logger for this object
                 logger_file_path=None,  # The path to the log file
                 log_level=None,  # The logging level
                 force_training=False,  # Force training the model
                 keras_log='PricePredict_keras.log',  # The keras log file
                 yf_sleep=61,  # The sleep time for yfinance requests
                 ):
        """
        Initialize the PricePredict class.
        This class is used to predict stock prices using a LSTM model.
        It has methods to load data, augment data, scale data, train the model,
        and predict the price the next period given a series of OHLC, Adj Close and Volume.
        There are methods to adjust the prediction based on prior actual prices
        to make the prediction highly accurate.
        And there are methods to output a chart of the prediction.
        :param model_dir:
        :param back_candles:
        :param split_pcnt:
        :param batch_size:
        :param epochs:
        :param shuffle:
        :param validation_split:
        :param verbose:
        :return PricePredict:   # An instance of the PricePredict class
        """
        self.model_dir = model_dir  # The directory where the model is saved
        self.model_path = ''  # The path to the current loaded model
        self.preds_dir = preds_dir  # The directory where the predictions are saved
        self.preds_path = ''  # The path to the current predictions
        self.chart_dir = chart_dir  # The directory where the charts are saved
        self.chart_path = ''  # The path to the current chart
        self.seasonal_chart_path = ''  # The path to the seasonal decomposition chart
        self.period = period  # The period for the data (D, W)
        self.model = None  # The current loaded model
        self.bayes_best_loss = None  # The best loss from the bayesian optimization
        self.bayes_best_model = None  # The best bayesian optimized model, temp holder.
        self.bayes_opt_hypers = {}  # The optimized hyperparameters
        self.lstm_units = lstm_units  # The current models units
        self.lstm_dropout = lstm_dropout  # The current models dropout
        self.adam_learning_rate = adam_learning_rate  # The current models learning rate
        self.scaler = None  # The current models scaler
        self.Verbose = verbose  # Print debug information
        self.ticker = ticker  # The ticker for the data
        self.ticker_data = None  # The data for the ticker (Long Name, last price, etc.)
        self.date_start = ''  # The start date for the data
        self.date_end = ''  # The end date for the data
        self.orig_data = None  # Save the originally downloaded data for later use.
        self.orig_downloaded_data = None  # Save the originally downloaded data for later use.
        self.missing_rows_analysis = None  # Save the missing rows analysis for later review.
        self.unagg_data = None  # Save the unaggregated data for later use.
        self.date_data = None  # Save the date data for later use.
        self.aug_data = None  # Save the augmented data for later use.
        self.features = None  # The number of features in the data
        self.targets = None  # The number of targets (predictions) in the data
        self.data_scaled = None  # Save the scaled data for later use.
        self.force_training = force_training  # Force training the model
        self.X = None  # 3D array of training data.
        self.y = None  # Target values (Adj Close)
        self.X_train = None  # Training data
        self.X_test = None  # Test data
        self.y_test = None  # Test data
        self.y_test_closes = None  # Test data closes
        self.y_train = None  # Training data
        self.y_pred = None  # The prediction
        self.y_pred_rescaled = None  # The rescaled prediction
        self.mean_squared_error = None  # The mean squared error for the model
        self.target_close = None  # The target close
        self.target_high = None  # The target high
        self.target_low = None  # The target low
        self.pred = None  # The predictions (4 columns)
        self.pred_rescaled = None  # The  predictions rescaled (4 columns)
        self.pred_class = None  # The prediction class
        self.pred_close = None  # The adjusted prediction close
        self.pred_high = None  # The adjusted prediction close
        self.pred_low = None  # The adjusted prediction close
        self.adj_pred = None  # The adjusted predictions (3 columns)
        self.adj_pred_class = None  # The adjusted prediction class
        self.adj_pred_close = None  # The adjusted prediction close
        self.adj_pred_high = None  # The adjusted prediction high
        self.adj_pred_low = None  # The adjusted prediction low
        self.dateStart_train = None  # The start date for the training period
        self.analysis = None  # The analysis of the model
        self.analysis_path = None  # The path to the analysis file
        self.dateEnd_train = None  # The end date for the training period
        self.dateStart_pred = None  # The start date for the prediction period
        self.dateEnd_pred = None  # The end date for the prediction period
        self.back_candles = back_candles  # The number of candles to look back for each period.
        self.split_pcnt = split_pcnt  # The value for splitting data into training and testing.
        self.split_limit = None  # The split limit for training and testing data.
        self.batch_size = batch_size  # The batch size for training the model.
        self.epochs = epochs  # The number of epochs for training the model.
        self.shuffle = shuffle  # Shuffle the data for training the model.
        self.val_split = val_split  # The validation split used during training.
        self.seasonal_dec = None  # The seasonal decomposition
        self.keras_log = keras_log  # The keras log file
        self.keras_verbosity = keras_verbosity  # The keras verbosity level
        self.cached_train_data = DataCache()  # Cached training data
        self.cached_pred_data = DataCache()  # Cached prediction data
        self.last_fetch_start_date = None  # The last fetch start date
        self.last_fetch_end_date = None  # The last fetch end date
        # Analitics...
        self.last_analysis = None  # The last analysis
        self.preds_path = None  # The path to the predictions file
        self.pred_last_delta = None  # The last delta in the prediction
        self.pred_rank = None  # The rank of the prediction
        self.season_last_delta = None  # The last delta in the seasonal decomposition
        self.season_rank = None  # The rank of the seasonal decomposition
        self.season_corr = None  # The corr of the seasonal decomposition
        self.pred_strength = None  # The strength of the prediction
        self.top10coint = None  # The top 10 cointegrations dict {'<Sym>': <coint_measure>}
        self.trading_pairs = 0  # The number of trading pairs
        self.top10corr = None  # The top 10 correlations dict {'<Sym>': <Corr%>}
        self.top10xcorr = None  # The top 10 cross correlations dict {'<Sym>': <xCorr%>}
        self.sentiment_json = {}  # The sentiment as json
        self.sentiment_text = ''  # The sentiment as text
        self.yf_sleep = yf_sleep  # The sleep time for yfinance requests
        self.yf_cached = False  # Using the yfinance_cache
        self.yf = None  # The yfinance/yfinance_cache object used
        self.spread_analysis = {}  # Spread analysis against other tickers
        self.logger = None  # The logger for this object


        # Create a logger for this object.
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        silence_tensorflow()

        # # Set the objects logging to stdout by default for testing.
        # # The logger and the logging-level can be overridden by the calling program
        # # once the object is instantiated.
        # if self.logger.hasHandlers():
        #     self.logger.handlers.clear()
        # if tf.get_logger().hasHandlers():
        #     tf.get_logger().handlers.clear()

        # # Turn off this objects logging to stdout
        # self.logger.propagate = False
        # # Turn off tensorflow logging to stdout
        # tf.get_logger().propagate = False

        # # Set the logger to stdout by default.
        # if logger_file_path is None:
        #     self.logger.addHandler(logging.StreamHandler(stream=sys.stdout))
        # else:
        #     self.logger.addHandler(logging.FileHandler(filename=logger_file_path))
        #     tf.get_logger().addHandler(logging.FileHandler(filename=logger_file_path))

        # Set the logging level.
        if log_level is None:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(log_level)

        # Verify that we can see the chart directory.
        if not os.path.exists(self.chart_dir):
            raise RuntimeError(f"*** Exception: Chart directory [{self.chart_dir}] does not exist at [{os.getcwd()}]")
        # Verify that we can see the model directory.
        if not os.path.exists(self.model_dir):
            raise RuntimeError(f"*** Exception: Model directory [{self.model_dir}] does not exist at [{os.getcwd()}]")
        # Verify that we can see the predictions directory.
        if not os.path.exists(self.preds_dir):
            raise RuntimeError(
                f"*** Exception: Predictions directory [{self.preds_dir}] does not exist at [{os.getcwd()}]")

        # Set the yfinance mode...
        if self.yf is None:
            if self.yf_cached:
                self.yf = yfc
            else:
                self.yf = yf

        self.logger.debug(f"=== PricePredict: Initialized.")

    def chk_yahoo_ticker(self, ticker):
        """
        Verify that the ticker is valid against Yahoo Finance.
        If the yahoo ticker is valid...
          - Save the ticker to this object (self.ticker).
          - Save the ticker data to this object (self.ticker_data).
          - The ticker data has the long name, last price,
            and other key information about the ticker.
          - return the ticker data.
        :param ticker:
        :return:
        """
        if self.orig_data is not None and self.ticker_data is not None:
            # Assume that the ticker is valid if we already have data.
            return self.ticker_data

        if 'Test-' in ticker:
            # Remove the 'Test-' from the symbol.
            # Special case for unit testing.
            chk_ticker = re.sub(r'^Test-', '', ticker)
        else:
            chk_ticker = ticker
        # Remove any quotes from the ticker string.
        chk_ticker = re.sub(r'[\"\']', '', chk_ticker)
        # Make the ticker uppercase.
        chk_ticker = chk_ticker.upper()
        ticker_data = None
        self.ticker = None
        self.ticker_data = None
        retry_needed = False
        retry_success = False
        retry_cnt = 0
        sleep_n = self.yf_sleep

        xyf = self.yf
        # Check if ticker begins with '^'
        if chk_ticker[0] == '^' and self.yf_cached:
            # Don't use yfinance_cache for indexes.
            xyf = yf

        # Check yahoo if symbol is valid...
        while True:
            try:
                # period='5d' can fail for some tickers such as SRCL
                try:
                    ticker_data = xyf.Ticker(chk_ticker).info
                except Exception as e:
                    if 'json' in str(e):
                        self.logger.warn(f"Warn: 1 - Ticker().info JSON processing for Ticker [{chk_ticker}].")
                    else:
                        self.logger.warn(f"Warn: 2 - Ticker().info failed for Ticker [{chk_ticker}].\n{e}")
                    retry_needed = True
                    if retry_cnt < 2:
                        self.logger.warn(f"Too Many yfinance Requests. Sleeping for {sleep_n} second.")
                        self.logger.warn(f"WARN: Retrying after sleep({sleep_n})...")
                        time.sleep(sleep_n)
                        retry_cnt += 1
                        continue
                    self.logger.warn(f"Warn: 3 - Ticker().info failed for Ticker [{chk_ticker}].")
                    break

                if ticker_data is None or len(ticker_data) <= 1:
                    ticker_data = None
                    self.logger.error(f"Error: Ticker [{chk_ticker}] - ticker_data has no data. Seems to be Invalid.")
                    raise RuntimeError(f"Error: Ticker [{chk_ticker}] - ticker_data has no data. Seems to be Invalid.")

                ticker_ = xyf.Ticker(chk_ticker).history(period='1mo', interval='1d')
                if ticker_ is None or len(ticker_) == 0:
                    self.logger.error(f"Error: Ticker [{chk_ticker}] - ticker_ has no data. Seems to be Invalid.")
                    raise RuntimeError(f"Error: Ticker [{chk_ticker}] - ticker_ has no data. Seems to be Invalid.")

                self.ticker = ticker
                self.ticker_data = ticker_data
                if retry_needed:
                    retry_success = True
                break
            except Exception as e:
                self.logger.error(f"Error: in Ticker().history(): {ticker}\n{e}")
                if 'has no data.' in str(e):
                    # When API requests limit is reached, we have to sleep for at least 60 seconds.
                    # The "Too Many Requests" exception gets hidden behind a JSON error, so we assume that
                    # if we get the "has no data" error, we hit the API limit, and must sleep.
                    # For testing on an invalid ticker, we make yf_sleep = 3 for faster testing.
                    retry_needed = True
                    if retry_cnt < 2:
                        self.logger.warn(f"Too Many yfinance Requests. Sleeping for {sleep_n} second.")
                        self.logger.warn(f"WARN: Retrying after sleep({sleep_n})...")
                        time.sleep(sleep_n)
                        retry_cnt += 1
                        continue
                    else:
                        break
                else:
                    self.logger.error(f"Ticker().history() or Ticker().info failed for Ticker [{chk_ticker}]. {e}")
                    ticker_data = None
                    break

        if retry_needed and retry_success is True:
            self.logger.info(f"Ticker [{chk_ticker}] pulled successfully on retry.")

        return ticker_data

    def fetch_data_yahoo(self, ticker, date_start, date_end, period=None, force_fetch=False):
        """
        Load data from Yahoo Finance.
        Caveats: Yahoo download data may not be the same as the original data from the exchange.

        Note: self.date_start and self.date_end are set based on parameters of the last
              call to this function. They are not set by the data in the self.orig_data.

        :param ticker:
        :param date_start:
        :param date_end:
        - Optional
        :param period:      # The period for the data (D, W)
        :return:
            data,           # An array of data Open, High, Low, Close, Adj Close, Volume
            feature_cnt:    # The number of features in the data
        """

        xyf = self.yf
        # Check if ticker begins with '^'
        if ticker[0] == '^' and self.yf_cached:
            # Don't use yfinance_cache for indexes.
            xyf = yf

        if period is None:
            period = self.period
        else:
            if period not in PricePredict.PeriodValues:
                self.logger.error(
                    f"*** Exception: [{ticker}]: period[{period}] must be \"{'"| "'.join(PricePredict.PeriodValues)}\"")
                raise ValueError(f"period[{period}]: man only be \"{'", "'.join(PricePredict.PeriodValues)}\"")
            self.period = period

        if self.Verbose:
            self.logger.info(f"Loading Data for: {ticker}  from: {date_start} to {date_end}, Period: {self.period}")

        # if self.orig_data is not None:
        # Check the start and end dates to see if the requested data is in self.orig_data.
        # if the requested data is in self.orig_data, then return the data from self.orig_data.
        if self.orig_data is not None and force_fetch is False:

            if date_start >= self.date_start and date_end <= self.date_end:
                if self.period == self.PeriodDaily:
                    # Get the data from the self.orig_data using the date range.
                    data = self.orig_data.loc[date_start:date_end]
                    return data, data.shape[1]
                else:
                    # Get the data from the self.orig_data using the date range.
                    # The date aggregation returns data indexed by the Sunday date.
                    # Calculate the Sunday for the date_start.
                    date_obj = datetime.strptime(date_start, '%Y-%m-%d')
                    monday = date_obj - timedelta(days=date_obj.weekday())
                    w_date_start = monday.strftime('%Y-%m-%d')
                    # Calculate the Sunday date_end.
                    date_obj = datetime.strptime(date_end, '%Y-%m-%d')
                    monday = date_obj + timedelta(days=6) - timedelta(days=date_obj.weekday())
                    w_date_end= monday.strftime('%Y-%m-%d')

                    data = self.orig_data.loc[w_date_start:w_date_end]
                    return data, data.shape[1]

        # Remove "Test-" from the start of the ticker (used for model testing)
        f_ticker = re.sub(r'^Test-', '', ticker)
        retry_cnt = 0
        retry_needed = False
        retry_success = False
        data = []
        while True:
            try:
                if self.period in [PricePredict.PeriodWeekly, PricePredict.PeriodDaily]:
                    data = xyf.download(tickers=f_ticker, start=date_start, end=date_end)
                else:
                    data = xyf.download(tickers=f_ticker, start=date_start, end=date_end, interval=self.period)
                break
            except Exception as e:
                if 'Expecting value' in str(e):
                    retry_needed = True
                    if retry_cnt < 2:
                        self.logger.warn(f"Too Many yfinance Requests. Sleeping for {sleep_n} second.")
                        self.logger.warn(f"WARN: Retrying after sleep({sleep_n})...")
                        time.sleep(sleep_n)
                        retry_cnt += 1
                        continue
                    else:
                        break
                else:
                    self.logger.error(f" 1: Ticker().history() or Ticker().info failed for Ticker [{f_ticker}].\n {e}")
                    ticker_data = None
                    break

        if len(data) == 0:
            self.logger.error(f"Error: No data for {ticker} from {date_start} to {date_end}")
            return None, None

        # If the column is a tuple, then we only want the first part of the tuple.
        if len(data) > 0:
            cols = data.columns
            if type(cols[0]) == tuple:
                cols = [col[0] for col in cols]
                data.columns = cols

        if self.yf_cached and 'Adj Close' not in data.columns:
            data['Adj Close'] = data['Close']
            data = data[['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']]

        if data.index.name is None and isinstance(data.index, pd.DatetimeIndex):
            # This happens when using yfinance_cache.
            # Give the unnamed DateTimeIndex the name 'Date'.
            data.index.name = 'Date'

        if 'Date' != data.index.name:
            if 'Datetime' == data.index.name:
                # Rename the data's index.name to 'Date'
                data.index.name = 'Date'
            else:
                self.logger.error(
                    f"Error: No Date or Datetime in data.index.name for {ticker} from {date_start} to {date_end}")
                return None, None

        best_start_date = data.index[0].strftime('%Y-%m-%d')
        best_end_date = data.index[-1].strftime('%Y-%m-%d')
        # Create a data frome that contains all business days.
        if self.period == self.PeriodWeekly:
            all_days = pd.date_range(start=best_start_date, end=best_end_date, freq='W')
        else:
            all_days = pd.date_range(start=best_start_date, end=best_end_date, freq='B')
        # Add rows into self_data that are not in ppo_data, such that fields other than 'Date' are NaN
        data_nan_filled = data.reindex(all_days)
        # Interpolate the missing data in the dataframe, given the existing data.
        interpolated_data = data_nan_filled.interpolate(method='time')
        # Missing rows analysis on self_data
        # Step 1: Identify missing rows (where all data columns are NaN)
        missing_rows_mask = data_nan_filled.isnull().all(axis=1)
        # Step 2: Count the number of missing rows
        missing_rows = missing_rows_mask.sum()
        # Group by month to get the distribution
        missing_distribution = missing_rows_mask.groupby(pd.Grouper(freq='ME')).sum()

        # Aggregate the data to a weekly period, if nd
        unagg_data = None
        orig_data = data.copy(deep=True)
        if self.period == self.PeriodWeekly:
            wkl_data = self.aggregate_data(interpolated_data, period=self.PeriodWeekly)
            # Make new copies of the data so that these separate dataframes don't share the same memory.
            unagg_data = interpolated_data.copy(deep=True)
            orig_data = wkl_data.copy(deep=True)
            data = wkl_data.copy(deep=True)

        if self.Verbose:
            self.logger.info(f"data.len(): {len(orig_data)}  data.shape: {orig_data.shape}")

        self.ticker = ticker
        self.date_start = date_start
        self.date_end = date_end
        self.missing_rows_analysis = {'missing_rows': missing_rows, 'missing_distribution': missing_distribution}
        self.orig_downloaded_data = orig_data
        self.orig_data = interpolated_data
        self.unagg_data = unagg_data

        return data, data.shape[1]

    def aggregate_data(self, data, period):
        """
        Aggregate the data to a weekly period.
        :param data:
        :return:
        """
        data = data.resample(period).agg(
            {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Adj Close': 'last', 'Volume': 'sum'})
        return data

    def augment_data(self, data, feature_cnt):
        """
        Add technical indicators to the data.
        Add the target and target class to the data, needed for training and testing.
        :param feature_cnt:
        :return:
            data,           # An array of data Open, High, Low, Close, Adj Close, Volume, + indicators
            feature_cnt,    # Number of features in the data
            target_cnt,     # Number of targets in the data
            data_date       # An array of dates (Extracted from the data)
        """

        # logger.info("= Before Adding Indicators ========================================================")
        # logger.info(data.tail(10))

        # Make a copy of the data to that we don't modify orig_data
        data = data.copy(deep=True)

        data['RSI'] = ta.rsi(data.Close, length=3);
        feature_cnt += 1
        # data['EMAF']=ta.ema(data.Close, length=3); feature_cnt += 1
        # data['EMAM']=ta.ema(data.Close, length=6); feature_cnt += 1
        data['EMAS'] = ta.ema(data.Close, length=9)
        feature_cnt += 1
        data['DPO3'] = ta.dpo(data.Close, length=3, lookahead=False, centered=False)
        feature_cnt += 1
        data['DPO6'] = ta.dpo(data.Close, length=6, lookahead=False, centered=False)
        feature_cnt += 1
        data['DPO9'] = ta.dpo(data.Close, length=9, lookahead=False, centered=False)
        feature_cnt += 1

        # logger.info("= After Adding DPO2 ========================================================")
        # logger.info(data.tail(10))
        #
        # On Balance Volume
        if data['Volume'].iloc[-1] > 0:
            data = data.join(ta.aobv(data.Close, data.Volume, fast=True, min_lookback=3, max_lookback=9))
            feature_cnt += 7  # ta.aobv adds 7 columns

        # logger.info("= After Adding APBV ========================================================")
        # logger.info(data.tail(10))

        # Target is the difference between the adjusted close and the open price.
        data['Target'] = data['Adj Close'] - data.Open
        feature_cnt += 1
        data['TargetH'] = data.High - data.Open
        feature_cnt += 1
        data['TargetL'] = data.Low - data.Open
        feature_cnt += 1
        # Shift the target up by one day.Target is the difference between the adjusted close and the open price.
        # That is, the target is the difference between the adjusted close and the open price.
        # Our model will predict the target close for the next day. So we shift the target up by one day.
        data['Target'] = data['Target'].shift(-1);
        data['TargetH'] = data['TargetH'].shift(-1)
        data['TargetL'] = data['TargetL'].shift(-1)

        # 1 if the price goes up, 0 if the price goes down.
        # Not a feature: Needed to test prediction accuracy.
        data['TargetClass'] = [1 if data['Target'].iloc[i] > 0 else 0 for i in range(len(data))]
        target_cnt = 1

        # The TargetNextClose is the adjusted close price for the next day.
        # This is the value we want to predict.
        # Not a feature: Needed to test prediction accuracy.
        data['TargetNextClose'] = data['Adj Close'].shift(-1)
        target_cnt += 1
        # TargetNextHigh and TargetNextLow are the high and low prices for the next day.
        data['TargetNextHigh'] = data['High'].shift(-1)
        target_cnt += 1
        # TargetNextLow are the low prices for the next day.
        data['TargetNextLow'] = data['Low'].shift(-1)
        target_cnt += 1

        # Before scaling the data, we need to use the last good value for rows that have NaN values.
        data.ffill(inplace=True)

        # Reset the index of the dataframe.
        data.reset_index(inplace=True)

        if 'Date' not in data.columns and data['index'].dtype == '<M8[ns]':
            # Rename the 'index' column to 'Date'
            data.rename(columns={'index': 'Date'}, inplace=True)
        if 'Date' not in data.columns:
            raise ValueError("Error: 'Date' column not found in data.")

        # Save the date data for later use.
        if 'Date' in data.columns:
            dates_data = data['Date'].copy()
        elif 'Date' in data.index.names:
            dates_data = data.index.copy()

        data.drop(['Date'], axis=1, inplace=True)
        feature_cnt -= 1
        data.drop(['Close'], axis=1, inplace=True)
        feature_cnt -= 1
        # data.drop(['Volume'], axis=1, inplace=True); feature_cnt -= 1

        # Add one more row to the data file, this will be our next day's prediction.
        data = pd.concat([data, data[-1:]])
        # And, reindex the dataframe.
        data.reset_index(inplace=True)

        self.aug_data = data.copy(deep=True)
        self.features = feature_cnt
        self.targets = target_cnt
        self.date_data = dates_data.copy(deep=True)

        return data, feature_cnt, target_cnt, dates_data

    def scale_data(self, data_set_in):
        """
        Scale and clean (no NaNs) the data using MinMaxScaler.
        :param data_set_in:           # An array of data Open, High, Low, Close, Adj Close, Volume, + indicators
        :return data_scaled, sc:   # An array of scaled data, and the scaler used to scale the data
        """
        sc = MinMaxScaler(feature_range=(0, 1))
        data_set_scaled = sc.fit_transform(np.array(data_set_in))

        # Cleanup NaNs from the scaled data.
        nan_indices = np.argwhere(np.isnan(data_set_scaled))
        zer_indices = np.argwhere(data_set_scaled == 0)

        for i in range(len(nan_indices)):
            j = nan_indices[i][1] - 1
            if j < 0:
                j = nan_indices[i][1] + 1
            data_set_scaled[nan_indices[i][0], nan_indices[i][1]] = data_set_scaled[nan_indices[i][0], j]
        for i in range(len(zer_indices)):
            j = zer_indices[i][1] - 1
            if j < 0:
                j = zer_indices[i][1] + 1
            data_set_scaled[zer_indices[i][0], zer_indices[i][1]] = data_set_scaled[zer_indices[i][0], j]

        nan_indices = np.argwhere(np.isnan(data_set_scaled))
        zer_indices = np.argwhere(data_set_scaled == 0)

        self.data_scaled = data_set_scaled
        self.scaler = sc

        return data_set_scaled, sc

    def restore_scale_pred(self, pred_data):
        """
        Restore the scale of the prediction.
        :param pred_data:               # The prediction Nx1 array
        :return pred_restored_scale:    # And Nx1 array of the restored scale data
        """

        # - We take our predictions and use them to replace the last column of our scaled data, for the same period.
        # - We then inverse transform the data to get the original scale.
        # The prediction data needs to be reshaped to an N x Targets array
        pred_data = np.reshape(pred_data, (len(pred_data), self.targets))
        # Get the last N rows [number of predictions to re-scale] of the scaled data.
        data_set_scaled = self.data_scaled[-len(pred_data):]
        # Replace the Target Columns of the scaled data with the prediction data.
        data_set_scaled[-len(pred_data):, -self.targets:] = pred_data

        # Inverse transform the data (this restores the original scale)
        pred_restored_scale = self.scaler.inverse_transform(data_set_scaled)

        self.pred = pred_restored_scale
        self.pred_rescaled = pred_restored_scale
        self.pred_class = pred_restored_scale[:, -4]
        self.pred_close = pred_restored_scale[:, -3]
        self.pred_high = pred_restored_scale[:, -2]
        self.pred_low = pred_restored_scale[:, -1]

        # Pull the Target Columns (Predictions) from the restored data.
        pred_restored_scale = pred_restored_scale[:, -self.targets:]

        self.pred_rescaled = pred_restored_scale

        return pred_restored_scale

    def load_model(self, *args, **kwargs):
        """
        Emulates an overloaded class method.
        - Loads a model given a moden path.
        - Loads a model given a ticker, start date, and end date.
        :param args:     # model_path
        :param kwargs:   # model_path, ticker, dateStart, dateEnd
        :return model:   # Returns the loaded keras model
        """
        model = None
        if len(args) == 1 and args[0] != '':
            return self._fetch_model_1(args[0])
        if len(kwargs) and 'model_path' in kwargs.keys():
            return self._fetch_model_1(kwargs['model_path'])

        if len(args) > 1 and args[0] != '':
            ticker = args[0]
            dateStart = args[1]
            dateEnd = args[2]
            model_dir = args[3]
            return self._fetch_model_2(ticker=ticker, dateStart=dateStart, dateEnd=dateEnd, modelDir=model_dir)
        if len(kwargs) > 0 and kwargs['ticker'] != '':
            return self._fetch_model_2(ticker=kwargs['ticker'], dateStart=kwargs['dateStart'],
                                       dateEnd=kwargs['dateEnd'], modelDir=kwargs['modelDir'])
        self.logger.error("Error: Invalid parameters.", sys.stderr)
        return None

    def _fetch_model_1(self, model_path=''):
        """
        Load a model given a model path.
        :param model_path:
        :return model:
        """
        model_path = model_path
        # Verify the model file exists.
        if not os.path.exists(model_path):
            self.logger.error(f"Error: Model file [{model_path}] does not exist.")
            raise ValueError(f"Error: Model file [{model_path}] does not exist.")
        if model_path is None or model_path == '':
            self.logger.error(f"Error: Model path var is empty [{model_path}].")
            raise ValueError(f"Error: Model path var is empty [{model_path}].")
        model = keras.models.load_model(model_path)
        if model is None:
            self.logger.error("Error: Could not load model.")
            raise ValueError("Error: Could not load model.")
        self.logger.debug(f"Model Loaded: {model_path}")

        if '_' + self.PeriodWeekly + '_' in model_path.lower():
            self.period = self.PeriodWeekly
        else:
            self.period = self.PeriodDaily

        if self.Verbose:
            self.logger.info(f"Model Loaded: {model_path}")
        self.model = model
        return model

    def _fetch_model_2(self, ticker='', dateStart='', dateEnd='', modelDir=None):
        """
        Load a model given a ticker, start date, and end date.
        :param ticker:
        :param dateStart:
        :param dateEnd:
        :return model:
        """
        # Build the filepath to the model
        if modelDir is None:
            modelDir = self.model_dir

        # Verify that the modelDir exists.
        if not os.path.exists(modelDir):
            self.logger.error(f"Error: Model directory [{modelDir}] does not exist.")
            raise ValueError(f"Error: Model directory [{modelDir}] does not exist.")

        model_file = ticker + f"_{self.period}_" + self.date_start + "_" + self.date_end + ".keras"
        # Python does not like the = sign in filenames.
        model_file = model_file.replace('=', '~')
        model_path = modelDir + model_file
        # Load the model
        model = keras.models.load_model(model_path)
        if model is None:
            self.logger.error(f"Error: load_model returned None: {model_path}")
            raise ValueError(f"Error: load_model returned None: {model_path}")
        self.logger.debug(f"Model Loaded: {model_path}")
        if self.Verbose:
            self.logger.info(f"Model Loaded: {model_path}")

        self.ticker = ticker
        self.model_dir = modelDir
        self.model = model
        self.model_path = model_path
        return model

    def check_for_recent_model(self, ticker, dateStart, dateEnd, period, within_days=3):
        """
        Check if a model exists for the given parameters.
        We look for a model file that matches our ticker, and is
        within 3 days of the starting date and ending date.
        :param ticker:
        :param dateStart:
        :param dateEnd:
        :param period:
        :return bool:     # Return: True if a model exists, False if not.
        """
        model_path = ''
        model_path = ticker + f"_{period}_" + dateStart + "_" + dateEnd + ".keras"
        # Python does not like the = sign in filenames.
        model_path = model_path.replace('=', '~')
        model_path = self.model_dir + model_path
        # Find all .keras model files in the model directory that start with the ticker.
        try:
            model_files = [f for f in os.listdir(self.model_dir) if ticker in f and f.endswith('.keras')]
        except Exception as e:
            self.logger.error(f"Exception: Getting model files: {e}")
            # No model files found.
            return False
        # Loop through the model files and see if a model is within 3 days of the starting date.
        # and within 3 days of the ending date.
        # - Convert dateStart and dateEnd to datetime objects.
        _dateStart = datetime.strptime(dateStart, '%Y-%m-%d')
        _dateEnd = datetime.strptime(dateEnd, '%Y-%m-%d')
        for model_file in model_files:
            # Verify that the model file is in the correct format.
            if len(re.split('[_|.]', model_file)) != 5:
                # Skip over files that don't match the format.
                continue
            # Split the model file name into its parts.
            sym, period, strt_dt, end_dt, suffix = re.split('[_|.]', model_file)
            # Check if the model file is within 3 days of the starting date.
            # Convert the dates to datetime objects.
            # Cheeck that strt_dt has a validate date format, using a regex.
            if len(strt_dt) != 10 or not re.match(r'\d{4}-\d{2}-\d{2}', strt_dt):
                self.logger.error(f"Error: Invalid date format [{strt_dt}] in model file: {model_file}")
                raise ValueError(f"Error: Invalid date format [{strt_dt}] in model file: {model_file}")
            if len(end_dt) != 10 or not re.match(r'\d{4}-\d{2}-\d{2}', end_dt):
                self.logger.error(f"Error: Invalid date format [{end_dt}] in model file: {model_file}")
                raise ValueError(f"Error: Invalid date format [{end_dt}] in model file: {model_file}")
            strt_dt = datetime.strptime(strt_dt, '%Y-%m-%d')
            end_dt = datetime.strptime(end_dt, '%Y-%m-%d')
            # Get the timedelta between the model file start date and the input start date.
            delta_start = abs((_dateStart - strt_dt).days)
            delta_end = abs((_dateEnd - end_dt).days)
            # Check if the model file is within (within_days) days of the input start date and end date.
            if delta_start <= within_days and delta_end <= within_days:
                # We found a model that matches, so save_plot it off and return True.
                self.model_path = self.model_dir + model_file
                self.period = period
                return True
        # No matching model found.
        return False

    def cache_training_data(self, symbol, dateStart_, dateEnd_, period):
        """
        Pull and Cache the training data.
        :param data:
        :param symbol:
        :param dateStart_:
        :param dateEnd_:
        :param period:
        :return:
        """
        if period is None:
            self.period = period
        self.logger.info(
            f"Cache Training Data:[{symbol}]  period:[{period}]  dateStart:[{dateStart_}]  dateEnd:[{dateEnd_}]")
        # =======================
        # Cache the training data
        # =======================
        # Verify that the Symbol is valid.
        if self.chk_yahoo_ticker(symbol) is None:
            self.logger.error(f"Error: Invalid Symbol: {symbol}")
            raise ValueError(f"Error: Invalid Symbol: {symbol}")

        # First, Check if the input of symbol, dateStart, dateEnd, and period
        # match an existing model. That is within 3 days of the model's date range.
        # If so, load the model and return it.
        # if not, we pull data and load up the training cache.

        # Clear out this object's ML model.
        self.model = None

        have_model = self.check_for_recent_model(symbol, dateStart_, dateEnd_, period)
        if have_model and not self.force_training:
            # Load the model
            self.model = self.load_model(model_path=(self.model_path))
            self.logger.info(f"Model Loaded: {self.model_path}. Training cache will not be loaded.")

        # Allways pull data for seasonality decomposition.
        # Load training data and prepare the data
        X, y = self._fetch_n_prep(symbol, dateStart_, dateEnd_,
                                  period=period)

        # Store the date data as a strings so that pydantic can serialize it.
        # It does not do a proper job if the date is a datetime object.
        str_datesData = []
        if self.period in [PricePredict.PeriodWeekly, PricePredict.PeriodDaily]:
            for item in self.date_data:
                str_datesData.append(item.strftime('%Y-%m-%d'))
        else:
            for item in self.date_data:
                str_datesData.append(item.strftime('%Y-%m-%d %H:%M:%S'))
        str_datesData = pd.Series(str_datesData)

        # Cache the data
        training_cache = DataCache()
        training_cache.symbol = symbol if symbol is not None else ''
        training_cache.dateStart = dateStart_ if dateStart_ is not None else ''
        training_cache.dateEnd = dateEnd_ if dateEnd_ is not None else ''
        training_cache.period = period if period is not None else ''
        tc_orig_data = self.orig_data.copy(deep=True)
        tc_orig_data.reset_index(inplace=True)
        tc_orig_data_str = tc_orig_data.to_json()
        training_cache.data = tc_orig_data_str if tc_orig_data_str is not None else '{}'
        training_cache.feature_cnt = self.features
        training_cache.data_scaled = list(self.data_scaled)
        training_cache.target_cnt = self.targets
        training_cache.dates_data = str_datesData.to_json()
        training_cache.X = list(self.X)
        training_cache.y = list(self.y)

        # Save the cached data into this object...
        self.cached_train_data = training_cache

        # Do a seasonality decomposition which possible requires pulling
        # data from Yahoo Finance.
        self.seasonality()

    def cache_prediction_data(self, symbol, dateStart_, dateEnd_, period):
        """
        Pull and Cache some Prediction data.
        :param data:
        :param symbol:
        :param dateStart_:
        :param dateEnd_:
        :param period:
        :return:
        """
        # =======================
        # Cache the training data
        # =======================
        # Verify that the Symbol is valid.
        if self.chk_yahoo_ticker(symbol) is None:
            self.logger.error(f"Error: Invalid Symbol: {symbol}")
            raise ValueError(f"Error: Invalid Symbol: {symbol}")
        # Load the data

        # Load training data and prepare the data
        X, y = self._fetch_n_prep(symbol, dateStart_, dateEnd_,
                                  period=period, split_pcnt=1)

        # Store the date data as a strings so that pydantic can serialize it.
        # It does not do a proper job if the date is a datetime object.
        str_datesData = []
        if self.period in [PricePredict.PeriodWeekly, PricePredict.PeriodDaily]:
            for item in self.date_data:
                # Daily and Weekly data does not have hours and minutes.
                str_datesData.append(item.strftime('%Y-%m-%d'))
        else:
            for item in self.date_data:
                # Hours and minutes data includes date and time.
                str_datesData.append(item.strftime('%Y-%m-%d %H:%M:%S'))
        str_datesData = pd.Series(str_datesData)

        # Cache the data
        prediction_cache = DataCache()
        prediction_cache.symbol = symbol if symbol is not None else ''
        prediction_cache.dateStart = dateStart_ if dateStart_ is not None else ''
        prediction_cache.dateEnd = dateEnd_ if dateEnd_ is not None else ''
        prediction_cache.period = period if period is not None else ''
        tc_orig_data = self.orig_data.copy(deep=True)
        tc_orig_data.reset_index(inplace=True)
        tc_orig_data_str = tc_orig_data.to_json()
        prediction_cache.data = tc_orig_data_str if tc_orig_data_str is not None else '{}'
        prediction_cache.feature_cnt = self.features
        prediction_cache.data_scaled = list(self.data_scaled)
        prediction_cache.target_cnt = self.targets
        prediction_cache.dates_data = str_datesData.to_json()
        prediction_cache.X = list(X)
        prediction_cache.y = list(y)

        # Save the cached data into this object...
        self.cached_pred_data = prediction_cache

    def cached_train_predict_report(self, force_training=None, no_report=False, save_plot=True, show_plot=False):
        """
        Train the model, make a prediction, and output a report.
        This method uses the cached training data and the cached prediction data,
        to train the model and make a prediction. Separating the training and prediction
        process allows for training and prediction to run concurrently, while pulling
        the training data and prediction data and caching it can be done with blocking calls.
        :return boolean:  # Returns True if the training and prediction were successful.
        """

        if force_training is None and self.force_training is not None:
            force_training = self.force_training
        else:
            force_training = False

        self.logger.debug(f"=== Started: Training and Predicting for [{self.ticker}] using cached data...")
        if self.model is None or force_training is True:
            tc = self.cached_train_data
            if tc is None:
                self.logger.error(
                    f"Error: No training data cached for {self.ticker}. Cached training data was expected.")
                raise ValueError(
                    f"Error: No training data cached for {self.ticker}. Cached training data was expected.")
            self.ticker = tc.symbol
            self.dateStart_train = tc.dateStart
            self.dateEnd_train = tc.dateEnd
            self.period = tc.period
            self.split_pcnt = tc.split_pcnt
            self.orig_data = pd.read_json(StringIO(tc.data))
            self.features = tc.feature_cnt
            self.data_scaled = np.array(tc.data_scaled)
            self.targets = tc.target_cnt
            str_datesdata = pd.Series(json.loads(tc.dates_data))
            if self.period in [PricePredict.PeriodWeekly, PricePredict.PeriodDaily]:
                # Daily and Weekly data does not have hours and minutes.
                self.date_data = pd.to_datetime(str_datesdata, format='%Y-%m-%d')
            else:
                # Hours and minutes data includes date and time.
                self.date_data = pd.to_datetime(str_datesdata, format='%Y-%m-%d %H:%M:%S')
            self.X = np.array(tc.X)
            self.y = np.array(tc.y)
            self.logger.info(f"Training [{self.ticker}] using cached data...")
            # Train a new model using the cached training data.
            model, y_pred, mse = self.train_model(self.X, self.y)
            if model is None:
                self.logger.error("Error: Could not train the model.")
                raise ValueError("Error: Could not train the model.")
            else:
                # Save the model
                self.model = model
                self.logger.info(f"Using existing model for [{self.ticker}], file-path {self.model_path}...")
                self.save_model(ticker=self.ticker,
                                date_start=self.dateStart_train,
                                date_end=self.dateEnd_train)

        # At this point, we have loaded a model.
        self.cached_predict_report(no_report=no_report, save_plot=save_plot, show_plot=show_plot)

    def cached_predict_report(self, no_report=False, save_plot=True, show_plot=False):
        # Load the cached prediction
        pc = self.cached_pred_data
        if pc is None:
            self.logger.error(
                f"Exception Error: No prediction data cached for {self.ticker}. Cached prediction data was expected.")
            return
        try:
            self.ticker = pc.symbol
            self.dateStart_pred = pc.dateStart
            self.dateEnd_pred = pc.dateEnd
            self.orig_data = pd.read_json(StringIO(pc.data))
            self.features = pc.feature_cnt
            self.data_scaled = np.array(pc.data_scaled)
            self.targets = pc.target_cnt
            str_datesdata = pd.Series(json.loads(pc.dates_data))
            if self.period in [PricePredict.PeriodWeekly, PricePredict.PeriodDaily]:
                # Daily and Weekly data does not have hours and minutes.
                self.date_data = pd.to_datetime(str_datesdata, format='%Y-%m-%d')
            else:
                # Hours and minutes data includes date and time.
                self.date_data = pd.to_datetime(str_datesdata, format='%Y-%m-%d %H:%M:%S')
            self.X = np.array(pc.X)
            self.y = np.array(pc.y)
        except Exception as e:
            self.logger.error(f"Exception Error: Could not load prediction data: {e}")
            return

        # Make Predictions on all the data
        self.split_pcnt = 1.0
        # Perform the prediction
        # This call will also save_plot the prediction data to this object.
        self.predict_price(self.X)
        # Perform data alignment on the prediction data.
        # Doing so makes use the the prediction deltas rather than the actual values.
        self.adjust_prediction()

        if no_report is False:
            """
            - Produce a prediction chart.
            - Save the prediction data to a file or database.
            - Save to weekly or daily data to a file or database.
            - Save up/down corr data to a file or database.
            - Perform Seasonality Decomposition.
            - Save the Seasonality Decomposition to a file or database.
            """
            self.logger.info(f"Performing price prediction for [{self.ticker}] using cached data...")
            try:
                self.gen_prediction_chart(last_candles=75, save_plot=save_plot, show_plot=show_plot)
            except Exception as e:
                self.logger.error(f"Exception Error: Could not generate prediction chart: {e}")

            self.save_prediction_data()

        # Save current datetime of the last analysis.
        self.last_analysis = datetime.now()
        self.logger.debug(f"=== Completed: Training and Predicting for [{self.ticker}] using cached data...Done.")

    def prep_model_inputs(self, data_set_scaled, feature_cnt, backcandles=15):
        """
        Prepare the model inputs.
        barckcandles is the number of candles to look back for each period.
        Returns...
        X is a 3D array of the feature data.
        y is a 2D array of the target data.
        :param data_set_scaled:
        :param feature_cnt:
        :param backcandles:
        :return X, y:
        """

        X = []
        # logger.info(data_set_scaled[0].size)
        # data_set_scaled=data_set.values
        # logger.info(data_set_scaled.shape[0])

        # Create a 3D array of the data. X[features][periods][candles]
        # Where candles is the number of candles that rolls by 1 period for each period.
        for j in range(feature_cnt):  # data_set_scaled[0].size):# last 2 columns are target not X
            X.append([])
            for i in range(backcandles, data_set_scaled.shape[0]):  # backcandles+2
                X[j].append(data_set_scaled[i - backcandles:i, j])

        # logger.info("X.shape:", np.array(X).shape)
        X = np.array(X)
        # Move axis from 0 to position 2
        X = np.moveaxis(X, [0], [2])
        # logger.info("X.shape:", X.shape)

        # The last 4 columns are the Targets.
        # The ML model will learn to predict the 4 Target columns.
        X, yi = np.array(X), np.array(data_set_scaled[backcandles:, -self.targets:])
        y = np.reshape(yi, (len(yi), self.targets))

        self.logger.info(f"data_set_scaled.shape: {data_set_scaled.shape}  X.shape: {X.shape}  y.shape: {y.shape}")
        # logger.info(X)
        # logger.info("=========================================================")
        # logger.info(y.shape)
        # logger.info(y)

        self.X = X
        self.y = y

        return X, y

    def fetch_opt_hyperparameters(self, ticker, hparams_file=None):
        """
        Fetch the training parameters for the given ticker from the ./gui_data/ticker_bopts.json file.
        :param ticker:
        :return:
        """
        # Get the training parameters for the ticker.
        # If the ticker is not in the training parameters, then use the default parameters.

        if hparams_file is None:
            hyperparams_files = [f"{self.model_dir}~ticker_bopts.json"]
            for hf in hyperparams_files:
                if os.path.exists(hf):
                    hparams_file = hf
                    break
        else:
            if os.path.exists(hparams_file):
                self.logger.error(f"Error: File [{hparams_file}] does not exist.")
                raise ValueError(f"Error: File [{hparams_file}] does not exist.")

        with open(hparams_file, 'r') as f:
            for line in f:
                opt_params = json.loads(line)
                sym = opt_params['symbol']
                hyper_params = opt_params['hparams']
                hyper_params_s = json.dumps(hyper_params)
                if sym == ticker:
                    self.bayes_opt_hypers = hyper_params
                    self.lstm_units = int(hyper_params['params']['lstm_units'])
                    self.lstm_dropout = hyper_params['params']['lstm_dropout']
                    self.adam_learning_rate = hyper_params['params']['adam_learning_rate']
                    self.ticker = ticker

    def train_model(self, X, y, split_pcnt=0.8, backcandels=None):
        """
        Train the model.
        Given the training data, split it into training and testing data.
        Train the model and return the model and the prediction.
        Adjust the prediction are not made here.
        :param X:
        :param y:
        :param split_pcnt:
        :param backcandels:
        :return:
        """

        self.logger.info(f"=== Training Model [{self.ticker}] [{self.period}]...")
        # Handle the optional parameters
        if split_pcnt is None:
            split_pcnt = self.split_pcnt
        splitlimit = int(len(X) * split_pcnt)

        self.split_limit = splitlimit
        if backcandels is None:
            backcandels = self.back_candles

        data_set = self.data_scaled
        feature_cnt = self.features

        # Fetch optimized hyperparameters, if available.
        self.fetch_opt_hyperparameters(self.ticker)

        # Split the scaled data into training and testing
        # logger.info("lenX:",len(X), "splitLimit:",splitlimit)
        X_train, X_test = X[:splitlimit], X[splitlimit:]  # Training data, Test Data
        y_train, y_test = y[:splitlimit], y[splitlimit:]  # Training data, Test Data

        # Get the model parameters from this object
        batch_size = self.batch_size
        epochs = self.epochs
        shuffle = self.shuffle
        validation_split = self.val_split

        lstm_units = 200
        lstm_dropout = 0.2
        adam_learning_rate = 0.035
        if self.lstm_units is not None:
            lstm_units = self.lstm_units
        if self.lstm_dropout is not None:
            lstm_dropout = self.lstm_dropout
        if self.adam_learning_rate is not None:
            adam_learning_rate = self.adam_learning_rate

        # Create the LSTM model
        if self.model is None:
            self.logger.debug("Creating a new model...")
            lstm_input = Input(shape=(backcandels, feature_cnt), name='lstm_input')
            inputs = LSTM(lstm_units, name='first_layer', dropout=lstm_dropout)(lstm_input)
            inputs = Dense(self.targets, name='dense_layer')(inputs)
            output = Activation('linear', name='output')(inputs)
            model = Model(inputs=lstm_input, outputs=output)
            adam = optimizers.Adam(learning_rate=adam_learning_rate)
            model.compile(optimizer=adam, loss='mse')
        else:
            self.logger.debug("Using existing self.model...")
            model = self.model

        # Define the CSV logger
        csv_logger = CSVLogger('PricePred_keras_training_log.csv')

        # Train the model
        model.fit(x=X_train, y=y_train,
                  batch_size=batch_size, epochs=epochs,
                  shuffle=shuffle, validation_split=validation_split,
                  callbacks=[csv_logger],
                  verbose=self.keras_verbosity)

        if len(X_test) > 0:
            y_pred = model.predict(X_test, verbose=self.keras_verbosity)
            fy_pred = np.array(pd.DataFrame(y_pred).replace({np.nan: 0}))
            fy_test = np.array(pd.DataFrame(y_test).replace({np.nan: 0}))
            mse = mean_squared_error(fy_test, fy_pred)
            if self.Verbose:
                self.logger.info(f"Mean Squared Error: {mse}")

            # Restore the scale of the prediction
            pred_rescaled = self.restore_scale_pred(y_pred.reshape(-1, self.targets))
        else:
            self.logger.warn("*** Won't predict. No test data.")
            y_pred = []
            mse = 0
            pred_rescaled = []

        # Save the model and the test and prediction
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.target_close = np.array(self.orig_data['Adj Close'].iloc[backcandels + splitlimit - 1:])
        self.target_high = np.array(self.orig_data['High'].iloc[backcandels + splitlimit - 1:])
        self.target_low = np.array(self.orig_data['Low'].iloc[backcandels + splitlimit - 1:])
        self.y_pred = y_pred
        self.mean_squared_error = mse
        self.pred = y_pred
        self.pred_rescaled = pred_rescaled
        self.dateStart_train = pd.to_datetime(self.date_data.iloc[0]).strftime("%Y-%m-%d")
        self.dateEnd_train = pd.to_datetime(self.date_data.iloc[-1]).strftime("%Y-%m-%d")
        self.dateStart_pred = pd.to_datetime(self.date_data.iloc[splitlimit + 1]).strftime("%Y-%m-%d")
        self.dateEnd_pred = pd.to_datetime(self.date_data.iloc[-1]).strftime("%Y-%m-%d")
        self.model = model

        self.logger.info(f"=== Model Training Completed [{self.ticker}] [{self.period}]...")

        return model, y_pred, mse

    def bayes_train_model(self, X, y, split_pcnt=0.8, backcandels=None,
                          lstm_units=256, lstm_dropout=0.2, adam_learning_rate=0.001):
        """
        Train the model.
        Given the training data, split it into training and testing data.
        Train the model and return the model and the prediction.
        Adjust the prediction are not made here.
        :param X:
        :param y:
        :param split_pcnt:
        :param backcandels:
        :return:
        """

        self.logger.info(f"=== Training Model [{self.ticker}] [{self.period}]...")
        # Handle the optional parameters
        if split_pcnt is None:
            split_pcnt = self.split_pcnt
        splitlimit = int(len(X) * split_pcnt)

        self.split_limit = splitlimit
        if backcandels is None:
            backcandels = self.back_candles

        data_set = self.data_scaled
        feature_cnt = self.features

        # Get the model parameters from this object
        batch_size = self.batch_size
        epochs = self.epochs
        shuffle = self.shuffle
        validation_split = self.val_split

        self.logger.debug("Creating a new model...")
        lstm_input = Input(shape=(backcandels, feature_cnt), name='lstm_input')
        inputs = LSTM(int(lstm_units), name='first_layer', dropout=lstm_dropout)(lstm_input)
        inputs = Dense(self.targets, name='dense_layer')(inputs)
        output = Activation('linear', name='output')(inputs)
        model = Model(inputs=lstm_input, outputs=output)
        adam = optimizers.Adam(learning_rate=adam_learning_rate)
        model.compile(optimizer=adam, loss='mse')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=splitlimit, random_state=42)
        # Callback: Define the CSV logger
        csv_logger = CSVLogger('PricePred_keras_training_log.csv')
        # Callback: Early Stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001)

        # Train the model
        model.fit(x=X_train, y=y_train,
                  batch_size=batch_size, epochs=epochs,
                  shuffle=shuffle, validation_split=validation_split,
                  validation_data=(X_test, y_test),
                  callbacks=[early_stopping, csv_logger],
                  verbose=self.keras_verbosity)

        loss = model.evaluate(X_test, y_test, verbose=self.keras_verbosity)
        # We are looking for a loss value that is as close to zero as possible.
        if self.bayes_best_loss is None or abs(loss) < abs(self.bayes_best_loss):
            self.bayes_best_loss = loss
            # Hold on to the best Hyperparameters
            self.lstm_units = lstm_units
            self.lstm_dropout = lstm_dropout
            self.adam_learning_rate = adam_learning_rate
            # Hold onto the best model found thus far.
            self.bayes_best_model = model

        return loss

    def bayesian_optimization(self, X, y,
                              # Hyperparameter Ranges
                              pb_lstm_units=(32, 256),
                              pb_lstm_dropout=(0.1, 0.5),
                              pb_adam_learning_rate=(0.001, 0.1),
                              opt_csv=None):
        """
        Perform Bayesian Optimization on the model.
        """

        # ======================================================
        # === This is the function that we want to optimize ===
        def optimize_model(lstm_units=None, lstm_dropout=None, adam_learning_rate=None):
            loss = self.bayes_train_model(X, y,
                                          lstm_units=lstm_units,
                                          lstm_dropout=lstm_dropout,
                                          adam_learning_rate=adam_learning_rate)
            # Loss is flipped because we want to maximize the loss.
            return -loss

        # === End of the function to optimize ===
        # ======================================================

        optimizer = BayesianOptimization(f=optimize_model,
                                         pbounds={'lstm_units': pb_lstm_units,
                                                  'lstm_dropout': pb_lstm_dropout,
                                                  'adam_learning_rate': pb_adam_learning_rate},
                                         random_state=42)
        optimizer.maximize(init_points=10, n_iter=20)

        # Save the best parameters
        self.bayes_opt_hypers = optimizer.max
        self.lstm_dropout = optimizer.max['params']['lstm_dropout']
        self.lstm_units = optimizer.max['params']['lstm_units']
        self.adam_learning_rate = optimizer.max['params']['adam_learning_rate']

        self.logger.info(f"Ticker: {self.ticker} Bayesian Optimization: {optimizer.max}")
        print(f"Bayesian Optimization: {{ Ticker: {self.ticker}: {optimizer.max} }}")

        # Make this PP objects models the best model found by the optimizer.
        self.model = self.bayes_best_model
        # Clear the saved model to reduce this object's pickle size
        self.bayes_best_model = None
        # Save out eh best model
        self.save_model(model=self.model, ticker=self.ticker)

        if opt_csv is not None:
            # Append the results to a CSV file.
            with open(opt_csv, 'a') as f:
                f.write(f"{{ Ticker: {self.ticker}: {optimizer.max} }}\n")

        return optimizer

    def save_model(self, model=None, model_dir: str = None,
                   ticker: str = None, date_start: str = None, date_end: str = None):
        """
        Save the model.
        And hold on to the model and data for later use.
        :param model:
        :param model_dir:
        :param ticker:
        :param date_start:
        :param date_end:
        :return:
        """
        if model is None:
            model = self.model
        if ticker is None:
            ticker = self.ticker
        if date_start is None:
            date_start = self.date_start
        if date_end is None:
            date_end = self.date_end

        # date_start = data.index[0].strftime("%Y-%m-%d")
        # date_end = data.index[-1].strftime("%Y-%m-%d")

        if model_dir is None:
            model_dir = self.model_dir

        if ticker == '':
            self.logger.error("Error: ticker is empty.")
            raise ValueError("Error: ticker is empty.")

        model_path = self.model_dir + ticker + f"_{self.period}_" + date_start + "_" + date_end + ".keras"
        # Python does not like the = sign in filenames.
        model_path = model_path.replace('=', '~')

        i = 0
        while i <= 3:
            i += 1
            try:
                # Save the model...
                model.save(model_path)
            except Exception as e:
                if i < 3:
                    self.logger.warn(f"Warning: Failed to Save model [{i}] [{model_path}]\n{e}, will retry...")
                    time.sleep(2)
                    continue
                else:
                    self.logger.error(f"Error: Saving model [{model_path}]\n{e}")
                    raise ValueError(f"Error: Saving model [{model_path}]\n{e}")

        self.ticker = ticker
        self.model_path = model_path
        self.last_analysis = datetime.now()

        return model, model_path

    def predict_price(self, X_data):
        """
        Predict the next price.
        If  X_data is None, then we will fetch the require data from Yahoo Finance and pre-process it.
        """
        self.logger.info(f"=== Predicting Price for [{self.ticker}] [{self.period}]...")

        if X_data is None:
            data, features = self.fetch_data_yahoo(self.ticker, self.dateStart_pred, self.dateEnd_pred)
            # Augment the data with additional indicators/features
            aug_data, features, targets, dates_data = self.augment_data(data, features)
            # Scale the augmented data
            scaled_data, scaler = self.scale_data(aug_data)
            # Prepare the scaled data for model inputs
            X_data, y = self.prep_model_inputs(scaled_data, features)
            # Make Predictions on all the data
            self.split_pcnt = 1.0

        try:
            # Perform the prediction
            y_pred = self.model.predict(X_data, verbose=self.keras_verbosity)
        except Exception as e:
            self.logger.error(f"Error: Predicting Price: {e}")
            return None
        else:
            # if self.split_limit is None:
            #     self.split_limit = int(len(X_data) * self.split_pcnt)
            self.split_limit = int(len(X_data) * self.split_pcnt)

            # Rescaled the predicted values to dollars...
            data_set_scaled_y = self.data_scaled[-(self.back_candles + self.split_limit):, :].copy()
            # Replace the last columns 4 in data_set_scaled_y with the predicted column values...
            min_len = min(len(y_pred), len(data_set_scaled_y))
            data_set_scaled_y[-min_len:, -self.targets:] = y_pred[-min_len:]

            y_pred_rs = self.scaler.inverse_transform(data_set_scaled_y)
            self.pred = y_pred
            self.pred_rescaled = y_pred_rs
            self.pred_class = y_pred_rs[:, -4]
            self.pred_close = y_pred_rs[:, -3]
            self.pred_high = y_pred_rs[:, -2]
            self.pred_low = y_pred_rs[:, -1]

            self.logger.debug(f"=== Price Prediction Completed [{self.ticker}] [{self.period}]...")

        return y_pred

    def adjust_prediction(self):
        """
        The adjusted prediction leverages the deltas between the predicted values
        and pins the delta to the prior actual close, high, and low values, rather
        than pining the prediction to the prior predicted value.
        This results in predictions that do not wander from the actual price action.

        :return y_p_adj,     # The adjusted prediction
                y_p_delta:   # The deltas between the actual price and the prediction
        """
        # Gather the predicted data for the test period.
        pred_class = self.pred_class
        pred_close = self.pred_close
        pred_high = self.pred_high
        pred_low = self.pred_low

        # Gather the target data for the test period.
        target_close = np.array(self.orig_data['Adj Close'].iloc[-len(pred_close):])
        target_high = np.array(self.orig_data['High'].iloc[-len(pred_high):])
        target_low = np.array(self.orig_data['Low'].iloc[-len(pred_low):])

        # Predictions are in y_red_rs
        # Generate deltas between current prediction and prior prediction...
        # -- Adjust Predicted Close
        pred_delta_c = [pred_close[i - 1] - pred_close[i] for i in range(1, len(pred_close))]
        min_len = min(len(target_close), len(pred_close))
        target_close = target_close[-min_len:]
        pred_adj_close = [target_close[i] + pred_delta_c[i] for i in range(0, len(pred_delta_c))]

        # -- Adjust Predicted High
        pred_delta_h = [pred_high[i - 1] - pred_high[i] for i in range(1, len(pred_high))]
        min_len = min(len(target_high), len(pred_high))
        target_high = target_high[-min_len:]
        pred_adj_high = [target_high[i] + abs(pred_delta_h[i]) for i in range(0, len(pred_delta_h))]

        #    -- Adjusted Close Prediction should not be higher than Adjusted High Prediction
        #       TODO: Adjust the both high and close to be median between the two.
        pred_adj_high = [pred_adj_close[i] if pred_adj_close[i] > pred_adj_high[i] else pred_adj_high[i] for i in
                         range(0, len(pred_delta_c))]

        # -- Adjust Predicted low
        pred_delta_l = [pred_low[i - 1] - pred_low[i] for i in range(1, len(pred_low))]
        min_len = min(len(target_low), len(pred_low))
        target_low = target_low[-min_len:]
        pred_adj_low = [target_low[i] - abs(pred_delta_l[i]) for i in range(0, len(pred_delta_l))]

        #    -- Adjusted Close Prediction should not be lower than Adjusted Low Prediction
        #       TODO: Adjust the both Low and close to be median between the two.
        pred_adj_low = [pred_adj_close[i] if pred_adj_close[i] < pred_adj_low[i] else
                        pred_adj_low[i] for i in range(0, len(pred_delta_l))]

        adj_pred = np.array([pred_class[-len(pred_adj_close):], pred_adj_close, pred_adj_high, pred_adj_low])
        adj_pred = np.moveaxis(adj_pred, [0], [1])

        # Calculate the strength of the prediction vs. other predictions.
        abs_deltas = np.abs(pred_delta_c)
        # Determine the rank of the last prediction from 1 to 10.
        ranking = np.digitize(abs_deltas, np.histogram(abs_deltas, bins=10)[1])
        # rank of the last value
        pred_rank = ranking[-2]  # Index to 2nd to last value, as the last value is a placeholder.

        self.pred_last_delta = pred_delta_c[-2]  # Index to 2nd to last value, as the last value is a placeholder.
        pred_sign = np.sign(pred_delta_c[-2])  # Index to 2nd to last value, as the last value is a placeholder.
        # Invert the rank so that longs are positive and shorts are negative
        self.pred_rank = (pred_sign * pred_rank) * -1

        self.target_close = target_close
        self.target_high = target_high
        self.target_low = target_low

        self.adj_pred = adj_pred  # The adjusted predictions
        self.adj_pred_class = pred_class  # Does not get adjusted
        self.adj_pred_close = pred_adj_close  # Adjusted close
        self.adj_pred_high = pred_adj_high  # Adjusted high
        self.adj_pred_low = pred_adj_low  # Adjusted low

        return pred_adj_close, pred_adj_high, pred_adj_low

    def gen_prediction_chart(self, last_candles=50,
                             file_path=None,
                             save_plot=False, show_plot=True):

        if file_path is None or file_path == '':
            last_date = self.date_data.iloc[-1]
            file_path = self.chart_dir + self.ticker + f"_{self.period}_{last_date}.png"
            self.chart_path = file_path
        if file_path is None or file_path == '':
            self.logger.error("Error: file_path is empty.")
            raise ValueError("Error: file_path is empty.")

        if self.date_data is None or len(self.date_data) == 0:
            self.logger.error("Error: self.date_data is empty. Prior data load is required.")
            raise ValueError("Error: self.date_data is empty. Prior data load is required.")
        if self.adj_pred is None or len(self.adj_pred) == 0:
            self.logger.error("Error: self.adj_pred is empty. Prior prediction is required.")
            raise ValueError("Error: self.adj_pred is empty. Prior prediction is required.")

        split_limit = int(len(self.date_data) * self.split_pcnt)
        self.split_limit = split_limit

        # Createe a dataframe of the data for plt_test_usd, with a datetime index...
        df_plt_test_usd = pd.DataFrame()

        split_start = self.back_candles + split_limit + 4
        if split_start >= len(self.date_data):
            split_start = 0

        plt_date = self.date_data.iloc[split_start:]

        # Set the date column for the plot...
        df_plt_test_usd.insert(0, 'Date', plt_date)

        # Setup the OHLCV data for the plot...
        if self.orig_data.shape[1] == 7:
            plt_ohlcv = self.orig_data.iloc[split_start:, [0, 1, 2, 3, 4, 5, 6]].copy()
        else:
            plt_ohlcv = self.orig_data.iloc[split_start:, [0, 1, 2, 3, 4, 5]].copy()
        if len(plt_ohlcv) == 0:
            # Handle the side effect of dealing with data from the prediction cache...
            if self.orig_data.shape[1] == 7:
                plt_ohlcv = self.orig_data.iloc[:, [0, 1, 2, 3, 4, 5, 6]].copy()
            else:
                plt_ohlcv = self.orig_data.iloc[:, [0, 1, 2, 3, 4, 5]].copy()

        ohlcv = plt_ohlcv.copy()
        if 'index' in ohlcv.columns:
            ohlcv.drop(columns='index', inplace=True)
            plt_ohlcv.drop(columns='index', inplace=True)
        ohlcv.reset_index()

        ticker = self.ticker

        if 'Date' not in ohlcv.columns and 'Date' == ohlcv.index.name:
            # df_plt_test_usd = pd.concat([df_plt_test_usd, ohlcv.set_axis(df_plt_test_usd.index)], axis=1)
            # df_plt_test_usd = pd.concat([df_plt_test_usd, ohlcv[-len(df_plt_test_usd):]], axis=1)
            df_plt_test_usd = ohlcv
            df_plt_test_usd.reset_index(inplace=True)
        else:
            df_plt_test_usd = ohlcv.copy()

        # Append a row do df_plt_test_usd for the prediction period
        # where open, high, low, and close are all equal to the last close price.
        if 'Close' in df_plt_test_usd.columns:
            last_close = df_plt_test_usd['Close'].iloc[-1]
            last_adj_close = df_plt_test_usd['Adj Close'].iloc[-1]
            if 'Date' not in df_plt_test_usd.columns and  df_plt_test_usd.index.dtype == '<M8[ns]':
                last_date = df_plt_test_usd.index[-1]
            else:
                if 'Date' not in df_plt_test_usd.columns:
                    raise ValueError("Error: 'Date' column is missing from the data in 'df_plt_test_usd'.")
                else:
                    last_date = df_plt_test_usd['Date'].iloc[-1]
            next_date = pd.to_datetime(last_date) + self.next_pd_DateOffset()
            new_row = {"Date": next_date,
                       "Open": last_close, "High": last_close, "Low": last_close, "Close": last_close,
                       "Adj Close": last_adj_close, "Volume": 0}
        else:
            last_close = df_plt_test_usd['Adj Close'].iloc[-1]
            last_adj_close = df_plt_test_usd['Adj Close'].iloc[-1]
            if 'Date' not in df_plt_test_usd.columns:
                df_dates = self.date_data.iloc[-len(df_plt_test_usd):]
                df_plt_test_usd.reset_index(drop=True, inplace=True)
                # Force the first column of df_dates to be 'Date'...
                df_dates = pd.DataFrame(df_dates, columns=['Date'])
                df_plt_test_usd = pd.concat([df_dates, df_plt_test_usd], axis=1)
            last_date = df_plt_test_usd['Date'].iloc[-1]
            next_date = pd.to_datetime(last_date) + self.next_pd_DateOffset()
            new_row = {"Date": next_date,
                       "Open": last_close, "High": last_close, "Low": last_close,
                       "Adj Close": last_adj_close, "Volume": 0}

        # Set the new row to the next day's date...
        new_row = pd.DataFrame(new_row, index=[next_date])
        # Append a place holder day for the prediction...
        df_plt_test_usd = pd.concat([df_plt_test_usd, new_row], axis=0)

        # If 'Close' is not in the columns, rename 'Adj Close' to 'Close'...
        if 'Close' not in df_plt_test_usd.columns:
            # Rename 'Adj Close' to 'Close' for the plot...
            df_plt_test_usd.rename(columns={'Adj Close': 'Close'}, inplace=True)

        title = (f'Ticker: {ticker} -- Period[ {self.period}] -- {self.dateStart_pred} to {last_date}\n'
                 f'Predictions High: {self.adj_pred_high[-1].round(2)}  Close: {self.adj_pred_close[-1].round(2)}  Low: {self.adj_pred_low[-1].round(2)}')
        kwargs = dict(type='candle', volume=True, figratio=(11, 6), figscale=2, warn_too_much_data=10000,
                      scale_padding=1, title=title)

        if hasattr(df_plt_test_usd.index[0], 'day') is False:
            df_plt_test_usd.set_index('Date', inplace=True, drop=True)

        # Trim the number of periods to plot if requested...
        min_len = min(len(df_plt_test_usd), last_candles)
        df_plt_test_usd = df_plt_test_usd.iloc[-min_len:].copy()
        min_len = min(len(df_plt_test_usd), len(self.adj_pred_close))
        preds = [mpf.make_addplot(self.adj_pred_close[-min_len:],
                                  type='line', panel=0, color='orange', secondary_y=False),
                 mpf.make_addplot(self.adj_pred_high[-min_len:],
                                  type='line', linestyle='-.', panel=0, color='blue', secondary_y=False),
                 mpf.make_addplot(self.adj_pred_low[-min_len:],
                                  type='line', linestyle='-.', panel=0, color='violet', secondary_y=False),
                 ]
        if self.seasonal_dec is not None:
            preds.append(mpf.make_addplot(self.seasonal_dec['_seasonal'][-min_len:], ylabel='Seasonality',
                                          type='line', linestyle='-', panel=2, color='blue', secondary_y=False))
            trend = self.seasonal_dec['_trend']
            # Find the first trend value from the end that is not NaN...
            for i in range(len(trend) - 1, 0, -1):
                if not np.isnan(trend.iloc[i]):
                    break
            # Shift the values that are non-NaN at i to the end of the array without
            # moving the Date field (tuple[0])...
            trend = np.roll(trend, len(trend) - i - 1)
            preds.append(mpf.make_addplot(trend[-min_len:], ylabel='Trend',
                                          type='line', linestyle='-.', panel=2, color='green', secondary_y=True))

        save_dict = dict(fname=file_path, dpi=300, pad_inches=0.25)

        df_plt_test_usd.ffill(inplace=True)
        fig = None
        if show_plot or save_plot:
            try:
                if save_plot:
                    fig, ax = mpf.plot(df_plt_test_usd[-min_len:], **kwargs,
                                       style='binance', addplot=preds, savefig=save_dict, returnfig=True)
                elif show_plot:
                    # For the interactive plot to show up, import mplfinance
                    # at the top of the script (or global level).
                    fig, ax = mpf.plot(df_plt_test_usd[-min_len:], **kwargs,
                                       style='binance', addplot=preds, returnfig=True)
            except Exception as e:
                self.logger.error(f"Error: Could not plot chart. {e}")

        return file_path, fig

    def save_prediction_data(self, file_path=None, last_candles=None):
        # Copy from the original data the OHLCV data for the prediction period...
        # df_ohlcv Will have an 'Date' as it's index...
        if 'Close' in self.orig_data.columns:
            df_ohlcv = pd.DataFrame(self.orig_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]).tail(
                len(self.pred) - 1)
        else:
            df_ohlcv = pd.DataFrame(self.orig_data[['Open', 'High', 'Low', 'Adj Close', 'Volume']]).tail(
                len(self.pred) - 1)
            # Duplicate the 'Adj Close' column as 'Close'...
            df_ohlcv['Close'] = df_ohlcv['Adj Close']

        # Add the date column if it is missing...
        if 'Date' not in df_ohlcv.columns and 'Date' not in df_ohlcv.index.names:
            # Handle the case where data is coming from cache...
            df_ohlcv.reset_index(inplace=True)
            dates_col = self.date_data[-len(df_ohlcv):].copy()
            dates_col.reset_index(drop=True, inplace=True)
            df_ohlcv['Date'] = dates_col
            df_ohlcv.set_index('Date', inplace=True, drop=True)
            df_ohlcv.drop(['index'], axis=1, inplace=True)

        # Get the last 'Date' of self.orig_data and add a day to it...
        last_date = df_ohlcv.index[-1]
        next_date = pd.to_datetime(last_date) + self.next_pd_DateOffset()
        last_close = df_ohlcv['Close'].iloc[-1]
        last_adj_close = df_ohlcv['Adj Close'].iloc[-1]
        new_row = {'Date': next_date,
                   "Open": last_close, "High": last_close, "Low": last_close, "Close": last_close,
                   "Adj Close": last_adj_close, "Volume": 0}
        # Set the new row to the next day's date...
        new_row = pd.DataFrame(new_row, index=[next_date])
        # Rename the index to 'Date'
        new_row.index.names = ['Date']
        # Append a place holder day for the prediction...
        df_ohlcv = pd.concat([df_ohlcv, new_row], axis=0)
        if 'Date' in df_ohlcv.columns:
            df_ohlcv.drop(columns=['Date'], inplace=True)
        # Gather up the prediction data...
        # Adjusted Predictions
        min_len = min(len(self.adj_pred), len(self.pred_class), len(df_ohlcv))
        df_adj_pred_close = pd.DataFrame(self.adj_pred[-min_len:, -3])
        df_adj_pred_high = pd.DataFrame(self.adj_pred[-min_len:, -2])
        df_adj_pred_low = pd.DataFrame(self.adj_pred[-min_len:, -1])

        # Unadjusted Predictions
        df_pred_class = pd.DataFrame(self.pred_class[-min_len:])
        df_pred_close = pd.DataFrame(self.pred_close[-min_len:])
        df_pred_high = pd.DataFrame(self.pred_high[-min_len:])
        df_pred_low = pd.DataFrame(self.pred_low[-min_len:])

        # df_ohlcv needs to match the length of the predictions...
        if len(df_pred_class) < len(df_ohlcv):
            df_ohlcv = df_ohlcv.iloc[-len(df_pred_class):].copy()
        df_ohlcv.reset_index(inplace=True)

        # Concatenate the columnar data into a single dataframe...
        df = pd.DataFrame()
        df = pd.concat([df_ohlcv,
                        df_pred_class, df_pred_close, df_pred_high, df_pred_low,
                        df_adj_pred_close, df_adj_pred_high, df_adj_pred_low
                        ], axis=1, ignore_index=True)
        if len(df.columns) == 15:
            df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
                          'junk',
                          'Pred Class', 'Pred Close', 'Pred High', 'Pred Low',
                          'Adj Pred Close', 'Adj Pred High', 'Adj Pred Low']
            df.drop(['junk'], axis=1, inplace=True)
        else:
            df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
                          'Pred Class', 'Pred Close', 'Pred High', 'Pred Low',
                          'Adj Pred Close', 'Adj Pred High', 'Adj Pred Low']
        df = df.round(2)

        file_paths = []

        # We can limit the number of candles to save_plot if requested...
        if last_candles is not None and len(df) > last_candles:
            df = df.tail(last_candles)

        # == Create the file-path to save_plot the prediction data...
        file_path = self.preds_dir + self.ticker + "_" + self.dateEnd_pred + ".csv"
        self.preds_path = file_path
        try:
            # Save the prediction data to a CSV file...
            df.to_csv(file_path, index=False)
        except Exception as e:
            self.logger.error(f"Error: Could not save_plot prediction data to:{file_path}")
            self.logger.error("Exception: {e}")

        self.preds_path = file_path
        file_paths.append(file_path)

        # === Analyze the prediction data...
        self.seasonality()
        self.prediction_analysis()
        # Save the analysis data...
        file_path = self.preds_dir + self.ticker + "_analysis_" + self.dateEnd_pred + ".json"
        try:
            # Save the analysis data to a json file...
            with open(file_path, 'w') as f:
                f.write(json.dumps(self.analysis))
        except Exception as e:
            self.logger.error(f"Error: Could not save_plot analysis data to: {file_path}")
            self.logger.error("Exception: {e}")
        self.analysis_path = file_path
        file_paths.append(file_path)

        return file_paths

    def _fetch_n_prep(self, ticker: str, date_start: str, date_end: str,
                      period: str = None, split_pcnt: float = 0.05,
                      backcandels: bool = None):

        # Handle the optional parameters
        if period is None:
            period = self.period
        else:
            if period not in PricePredict.PeriodValues:
                self.logger.error(f"period[{period}]: may only be \"{'", "'.join(PricePredict.PeriodValues)}\"")
                raise ValueError(f"period[{period}]: may only be \"{'", "'.join(PricePredict.PeriodValues)}\"")
            self.period = period
        if split_pcnt is None:
            split_pcnt = self.split_pcnt
        else:
            self.split_pcnt = split_pcnt
        if backcandels is None:
            backcandels = self.back_candles
        else:
            self.back_candles = backcandels

        # Load data from Yahoo Finance
        orig_data, features = self.fetch_data_yahoo(ticker, date_start, date_end, period)
        # Augment the data with technical indicators/features and targets
        aug_data, features, targets, dates_data = self.augment_data(orig_data, features)
        # Scale the data
        scaled_data, scaler = self.scale_data(aug_data)
        # Prepare the scaled data for model inputs
        X, y = self.prep_model_inputs(scaled_data, features)

        # Training split the X & y data into training and testing data
        splitlimit = int(len(X) * split_pcnt)
        self.split_limit = splitlimit

        # logger.info("lenX:",len(X), "splitLimit:",splitlimit)
        X_train, X_test = X[:splitlimit], X[splitlimit:]  # Training data, Test Data
        y_train, y_test = y[:splitlimit], y[splitlimit:]  # Training data, Test Data

        self.ticker = ticker
        self.orig_data = orig_data
        self.features = features
        self.targets = targets
        self.date_data = dates_data
        self.data_scaled = scaled_data
        self.dateStart_pred = date_start
        self.dateEnd_pred = date_end
        self.scaler = scaler
        self.X = X
        self.y = y
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.aug_data = aug_data
        self.data_scaled = scaled_data

        return X, y

    def fetch_and_predict(self, model_path: str = None,
                          date_start: str = None, date_end: str = None):
        """
        This is the main entry point for the class.
        Given a model_path and a date range, this method will load the model,
        load the data, augment the data,
        :param model_path:
        :param ticker:
        :param date_start:
        :param date_end:
        :return adj_predictions, pred_dates:
        """

        # TODO: Load a 'W'eekly model

        if model_path is None and self.model_path is None:
            self.logger.error("Error: The model_path parameter is required.")
            raise ValueError("Error: The model_path parameter is required.")
        else:
            model_path = self.model_path

        if date_start is None or date_end is None:
            self.logger.error("Error: The date_start and date_end parameters is required.")
            raise ValueError("Error: The date_start and date_end parameters is required.")

        model_path = model_path.replace('=', '~')

        # Extract the ticker from the model path
        ticker, period, dateStart_train, dateEnd_train = model_path.split('_')
        # Python does not like the = sign in filenames.
        # So, we restore the = sign to the ~ ticker symbol.
        ticker = ticker.replace('~', '=')
        if '/' in ticker:
            ticker = ticker.split('/')[-1]

        # Load the model
        # Python does not like the = sign in filenames.
        model_path = model_path.replace('=', '~')
        model = self.load_model(model_path)
        # Load the data period to predict
        orig_data, features = self.fetch_data_yahoo(ticker, date_start, date_end)
        # Augment the data
        aug_data, features, targets, dates = self.augment_data(orig_data, features)
        # Scale the data
        data_scaled, scaler = self.scale_data(aug_data)
        # Prepare the model inputs
        X, y = self.prep_model_inputs(data_scaled, features)
        # Predict the price
        orig_predictions = model.predict(X, verbose=self.keras_verbosity)
        # Restore the scale of the prediction
        rs_orig_predictions = self.restore_scale_pred(orig_predictions)

        # Hold on to the un-rescaled prediction data
        pred_class = rs_orig_predictions[:, 0]
        pred_close = rs_orig_predictions[:, 1]
        pred_high = rs_orig_predictions[:, 2]
        pred_low = rs_orig_predictions[:, 3]

        # Adjust the prediction
        adj_pred_close, adj_pred_high, adj_pred_low = self.adjust_prediction()

        # Get a list of the actual closing prices for the test period.
        closes = np.array(orig_data['Adj Close'])[:len(X) - 1]
        closes = np.append(closes, closes[-1])
        highs = np.array(orig_data['High'])[:len(X) - 1]
        highs = np.append(highs, highs[-1])
        lows = np.array(orig_data['Low'])[:len(X) - 1]
        lows = np.append(lows, lows[-1])

        self.dateStart_train = dateStart_train
        self.dateEnd_train = dateEnd_train
        self.dateStart_pred = date_start
        self.dateEnd_pred = date_end
        self.orig_data = orig_data
        self.data_scaled = data_scaled
        self.scaler = scaler
        self.X = X
        self.y = y
        self.target_close = closes
        self.target_high = highs
        self.target_low = lows

        self.pred = orig_predictions
        self.pred_rescaled = rs_orig_predictions
        self.pred_class = pred_class
        self.pred_close = pred_close
        self.pred_high = pred_high
        self.pred_low = pred_low

        pred_dates = self.date_data

        pred_dates = pd.Series._append(pred_dates, pd.Series(self.date_data.iloc[-1] + self.next_timedelta()))

        return self.adj_pred, pred_dates

    def fetch_train_and_predict(self, ticker,
                                train_date_start, train_date_end,
                                pred_date_start, pred_date_end,
                                force_training=False,
                                period=None, split_pcnt=None, backcandels=None,
                                use_curr_model=True, save_model=False):
        """
        Train and test the model.
        Does not load a model, only trains it.

        # Required Positional Parameters
        :param ticker:
        :param train_date_start:
        :param train_date_end:
        :param pred_date_start:
        :param pred_date_end:

        Optional Parameters:
        :param period=None:      # The period of the data, daily or weekly
        :param split_pcnt=None:  # The percentage of the data to use for training
        :param backcandels=None: # The number of candles to look back for each period
        :param save_model=False: # Save the model if True, and force training

        :return:
        """

        if force_training is None and self.force_training is not None:
            force_training = self.force_training
        else:
            force_training = False
            self.force_training = force_training

        if ticker is None or ticker == '':
            self.logger.error("Error: ticker is empty.")
            raise ValueError("Error: ticker is empty.")

        if period is None:
            period = self.period
        else:
            if period not in PricePredict.PeriodValues:
                self.logger.error(f"period[{period} is invalid. Must be \"{'", "'.join(PricePredict.PeriodValues)}\"")
                raise ValueError(f"period[{period}]: may only be \"{'", "'.join(PricePredict.PeriodValues)}\"")
            self.period = period
        if split_pcnt is None:
            split_pcnt = self.split_pcnt
        else:
            self.split_pcnt = split_pcnt
        if backcandels is None:
            backcandels = self.back_candles
        else:
            self.back_candles = backcandels

        model = None
        if use_curr_model and force_training is False:
            # If the self.model is None, load the latest model if it exists...
            if self.model is not None:
                # Load an existing model if it exists
                model_path = self.model_dir + ticker + f"_{period}_" + train_date_start + "_" + train_date_end + ".keras"
                if os.path.exists(model_path):
                    model = self.load_model(model_path)
                    self.logger.info(f">>> Model Loaded: {model_path}")
                else:
                    self.logger.info(f"=== Model Not Found: {model_path}")
            else:
                # Use the currently load loaded model.
                model = self.model

        if model is None:
            # Load training data and prepare the data
            X, y = self._fetch_n_prep(ticker, train_date_start, train_date_end,
                                      period=period, split_pcnt=0)

            # ============== Train the model
            # Use a small batch size and epochs to test the model training
            # Training split the X & y data into training and testing data
            # What is returned is the model, the prediction, and the mean squared error.
            model, y_pred, mse = self.train_model(X, y,
                                                  split_pcnt=split_pcnt, backcandels=backcandels)

            if len(y_pred) != 0:
                pcnt_nan = (len(y_pred) - np.count_nonzero(~np.isnan(y_pred))) / len(y_pred)
                if pcnt_nan > 0.1:
                    self.logger.info(f"\n*** NaNs in y_pred: {pcnt_nan}%")
                    # Throw a data exception if the model is not trained properly.
                    raise ValueError("Error: Prediction has too many NaNs. Check for Nans in the data?")

            # Load testing data and prepare the data
            X, y = self._fetch_n_prep(ticker, pred_date_start, pred_date_end,
                                      period=period, split_pcnt=0)
            # Predict the price for the test period
            y_pred = self.predict_price(X)

            pcnt_nan = (len(y_pred) - np.count_nonzero(~np.isnan(y_pred))) / len(y_pred)
            if pcnt_nan > 0.1:
                self.logger.info(f"\n*** NaNs in y_pred: {pcnt_nan}%")
                # Throw a data exception if the model is not trained properly.
                raise ValueError("Error: Prediction has too many NaNs. Check for Nans in the data?")

            adj_pred_close, adj_pred_high, adj_pred_low = self.adjust_prediction()
            # ============= End of training and testing the model
            # Save the model?
            if save_model:
                self.save_model(model, model_dir=self.model_dir,
                                ticker=ticker,
                                date_start=train_date_start, date_end=train_date_end)
                self.model = model

        self.dateStart_train = train_date_start
        self.dateEnd_train = train_date_end
        self.dateStart_pred = pred_date_start
        self.dateEnd_pred = pred_date_end
        self.X = X
        self.y = y

        return self.model

    def next_timedelta(self):
        time_delta = None

        if self.period == PricePredict.PeriodWeekly:
            time_delta = timedelta(days=7)
        elif self.period == PricePredict.PeriodDaily:
            time_delta = timedelta(days=1)
        elif self.period == PricePredict.Period1min:
            time_delta = timedelta(minutes=1)
        elif self.period == PricePredict.Period5min:
            time_delta = timedelta(minutes=5)
        # elif self.period == PricePredict.Period15min:
        #     time_delta = timedelta(minutes=15)
        # elif self.period == PricePredict.Period30min:
        #     time_delta = timedelta(minutes=30)
        elif self.period == PricePredict.Period1hour:
            time_delta = timedelta(hours=1)

        return time_delta

    def next_pd_DateOffset(self):
        pd_date_offset = None

        if self.period == PricePredict.PeriodWeekly:
            pd_date_offset = pd.DateOffset(weeks=1)
        elif self.period == PricePredict.PeriodDaily:
            pd_date_offset = pd.DateOffset(days=1)
        elif self.period == PricePredict.Period1min:
            pd_date_offset = pd.DateOffset(minutes=1)
        elif self.period == PricePredict.Period5min:
            pd_date_offset = pd.DateOffset(minutes=5)
        elif self.period == PricePredict.Period1hour:
            pd_date_offset = pd.DateOffset(hours=1)

        return pd_date_offset

    def prediction_analysis(self):
        """
        Analyze the prediction and generate a report.
        Report the MSE of the original prediction and the adjusted prediction.
        Report the trend of the prediction accuracy of the original and adjusted prediction.
        Report the percent accuracy of the prediction and the adjusted prediction
        with regard to the actual closing price.
        :return:
        """
        if self.pred is None:
            self.logger.error("Error: There is no prediction data yet.")
            return False

        # Convert back to dollar $values
        elements = len(self.adj_pred) - 1
        # logger.info(f"elements:{elements}")
        tot_deltas = 0
        tot_tradrng = 0
        for i in range(-1, -elements, -1):
            actual = self.orig_data['Adj Close'].iloc[i - 1]
            predval = self.adj_pred[i - 1][1]
            pred_delta = abs(predval - actual)
            tot_deltas += pred_delta
            trd_rng = abs(self.orig_data['High'].iloc[i] - self.orig_data['Low'].iloc[i])
            tot_tradrng += trd_rng
            self.logger.info(
                f"{i}: Close: {actual.round(2)}  Predicted: ${predval.round(2)}  Actual: ${actual.round(2)}  Delta: ${pred_delta.round(6)}  Trade Rng: ${trd_rng.round(2)}")

        self.logger.info("============================================================================")
        self.logger.info(f"Mean Trading Range: ${round(tot_tradrng / elements, 2)}")
        self.logger.info(f"Mean Delta (Actual vs Prediction): ${round((tot_deltas / elements), 2)}")
        self.logger.info("============================================================================")

        analysis = dict()
        analysis['actual_vs_pred'] = {
            'mean_trading_range': round(tot_tradrng / elements, 2),
            'mean_delta': round((tot_deltas / elements), 2)}

        elements = len(self.adj_pred_close)
        # logger.info(f"elements:{elements}")
        tot_deltas = 0
        tot_tradrng = 0
        for i in range(-1, -elements, -1):
            actual = self.orig_data['Adj Close'].iloc[i - 1]
            predval = self.adj_pred_close[i - 1]
            pred_delta = abs(predval - actual)
            tot_deltas += pred_delta
            trd_rng = abs(self.orig_data['High'].iloc[i] - self.orig_data['Low'].iloc[i])
            tot_tradrng += trd_rng
            self.logger.info(
                f"{i}: Close {actual.round(2)}  Predicted: ${predval.round(2)}  Actual: ${actual.round(2)}  Delta:${pred_delta.round(6)}  Trade Rng: ${trd_rng.round(2)}")

        self.logger.info("============================================================================")
        self.logger.info(f"Mean Trading Range: ${round(tot_tradrng / elements, 2)}")
        self.logger.info(f"Mean Delta (Actual vs Prediction): ${round((tot_deltas / elements), 2)}")
        self.logger.info("============================================================================")

        analysis['actual_vs_adj_pred'] = {
            'mean_trading_range': round(tot_tradrng / elements, 2),
            'mean_delta': round((tot_deltas / elements), 2)}

        if self.seasonal_dec is not None:
            # Analyze the trend corr between seasonality trend and the
            # predicted trend.
            sd_trend = self.seasonal_dec['_trend']
            # Find the first sd_trend value from the end that is not NaN...
            for i in range(len(sd_trend) - 1, 0, -1):
                if not np.isnan(sd_trend.iloc[i]):
                    break
            # Shift the values that are non-NaN at i to the end of the array without
            # moving the Date field (tuple[0])...
            sd_trends = np.roll(sd_trend, len(sd_trend) - i - 1)
            # Get up/down/flat trand days
            sd_trends = [1 if sd_trends[i] > sd_trends[i - 1] else -1 for i in range(len(sd_trends) - 1)]
            # Get deltas between days
            sd_deltas = [sd_trends[i] - sd_trends[i - 1] for i in range(1, len(sd_trends))]
            # Get up days vs down days
            self_data = self.orig_data
            # If the 'Close' column is not in the data, use the 'Adj Close' column
            if 'Close' not in self_data.columns:
                self_data['Close'] = self_data['Adj Close']
            try:
                self_trends = [1 if self_data['Close'].iloc[i] > self_data['Open'].iloc[i] else -1 for i in
                               range(len(self_data))]
            except Exception as e:
                self.logger.error(f"Exception: {e}")
                self.logger.error(f"Failed to collect trend data for self:{self.ticker}, for seasonality corr.")
            # Make sure that the lengths of self_trends and sd_trends are the same
            min_len = min(len(self_trends), len(sd_trends))
            self_trends = self_trends[-min_len:]
            sd_trends = sd_trends[-min_len:]

            # Calculate the corr
            corr_list = [self_trends[i] + sd_trends[i] for i in range(len(self_trends))]
            total_days = len(corr_list)
            correlated_days = corr_list.count(2) + corr_list.count(-2)
            uncorrelated_days = corr_list.count(0)
            pct_corr = correlated_days / total_days
            pct_uncorr = uncorrelated_days / total_days
            self.season_corr = pct_corr

            # Get the ranking of the last sd_delta value
            ranking = np.digitize(sd_deltas, np.histogram(sd_trends, bins=10)[1])
            # rank of the last value
            sd_rank = ranking[-1]

            sd_trend_sign = 1 if sd_deltas[-1] > 0 else -1

            self.season_last_delta = sd_trends[-1]
            # Invert the rank so that longs are positive and shorts are negative
            self.season_rank = (sd_trend_sign * sd_rank) * -1

            self.logger.info("Seasonal Trend Correlation...")
            self.logger.info(
                f"Days: {total_days} Correlated Days: {correlated_days}  Uncorrelated Days: {uncorrelated_days}")
            self.logger.info(f"Correlated Days: {pct_corr}%  Uncorrelated Days: {pct_uncorr}%")

            analysis['seasonal_trend_corr'] = {
                'total_days': f'{total_days}',
                'correlated_days': f'{correlated_days}',
                'uncorrelated_days': f'{uncorrelated_days}',
                'pct_corr': f'{self.season_corr}',
                'pct_uncorr': f'{pct_uncorr}'}

            analysis['pred_rankings'] = {
                'season_last_delta': f'{self.season_last_delta}',
                'season_rank': f'{self.season_rank}',
                'pred_last_delta': f'{self.pred_last_delta}',
                'pred_rank': f'{self.pred_rank}'}

            self.pred_strength = round((self.pred_rank + (self.season_rank * self.season_corr)) / 20, 4)
            analysis['pred_strength'] = {
                'strength': f'{self.pred_strength}'}

        self.analysis = analysis

        return analysis

    def model_report(self):
        """
        Print a report of the model.
        :return True: # If the model is loaded and the report is printed.
        """
        if self.model is None:
            self.logger.error("Error: Model is not loaded.")
            return False

        self.logger.info("Model Report...")
        self.logger.info(self.model.summary())
        plot_model(self.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

        return True

    def plot_pred_results(self,
                          target_close, target_high, target_low,
                          adj_pred_close, adj_pred_high, adj_pred_low,
                          title=''):
        # Plot the scaled test and predicted data.
        plt.figure(figsize=(16, 8))
        # Acj Close
        if target_close is not None:
            plt.plot(target_close, color='black', label='Target Close', linestyle='-')
        # Original Prediction
        if target_high is not None:
            plt.plot(target_high[1:], color='yellow', label='Target High', linestyle='-')
        if target_low is not None:
            plt.plot(target_low[1:], color='red', label='Target Low', linestyle='-')
        if adj_pred_close is not None:
            plt.plot(adj_pred_close[1:], color='orange', label='Adj Close', linestyle='-')
        if adj_pred_high is not None:
            plt.plot(adj_pred_high[1:], color='blue', label='Adj High', linestyle='-')
        if adj_pred_low is not None:
            plt.plot(adj_pred_low[1:], color='pink', label='Adj Low', linestyle='-')
        if title != '':
            plt.title(title)
        plt.legend()
        plt.grid()
        plt.show()
        return plt

    def periodic_correlation(self, ppo, pc_period_len: int = None, interolate: bool = True):
        """
        Calculate the corr of the predicted prices with the actual prices.
        - Pearson Correlation (Raw and Normalized Correlation)
        - Spearman Correlation (Trend Correlation)
        - Kendall Correlation (Trend Correlation)
        - Cointegration Test (Check for Cointegration)
        - Advanced Dickey-Fuller Test (Stationarity Test)

        Stocks that are both Cointegrated and Stationary are good candidates for Pairs Trading.

        # Required Parameters
        :param ppo:
            A PricePredict object to correlate with self
        # Optional Parameters
        :param pc_period_len:
            Length of the period to calculate the correlation ending with the last date.
        :param interpolate:
            If True, the data will be interpolated to the same length.
            Else, the originally downloaded data will be back-filled and forward-filled.
        :return:
            A dictionary containing the correlation data.
        """
        # PPO must be a PricePredict object
        if not isinstance(ppo, PricePredict):
            e_txt = f"The ppo parameter must be a PricePredict object."
            self.logger.error(e_txt)
            raise ValueError(e_txt)

        # Verify that the periods align.
        if self.period != ppo.period:
            e_txt = f"PPOs [{self.ticker}/{self.period}] [{ppo.ticker}/{ppo.period}] must have the same period."
            self.logger.error(e_txt)
            raise ValueError(e_txt)

        # Check that self has enough data points
        min_data_points = 7
        if self.orig_data is None or len(self.orig_data) < min_data_points:
            e_txt = f"self[{self.ticker}] has less than {min_data_points} data points."
            raise ValueError(e_txt)
        # Check that ppo has enough data points
        if ppo.orig_data is None or len(ppo.orig_data) < min_data_points:
            e_txt = f"ppo[{ppo.ticker}] has less than {min_data_points} data points."
            raise ValueError(e_txt)

        if interolate:
            # Get the original data from both objects
            self_data = self.orig_data
            ppo_data = ppo.orig_data
        else:
            # Get the latest start date and the earliest end date
            best_start_date = max(self.orig_downloaded_data.index[0], ppo.orig_downloaded_data.index[0])
            best_end_date = min(self.orig_downloaded_data.index[-1], ppo.orig_downloaded_data.index[-1])
            # Create a data frome that contains all business days.
            if self.period == self.PeriodWeekly:
                all_days = pd.date_range(start=best_start_date, end=best_end_date, freq='W')
            else:
                all_days = pd.date_range(start=best_start_date, end=best_end_date, freq='B')
            # Add rows into self_data that are not in ppo_data, such that fields other than 'Date' are NaN
            self_data = self.orig_downloaded_data.reindex(all_days).bfill().ffill()
            # Add rows into ppo_data that are not in self_data, such that fields other than 'Date' are NaN
            ppo_data = ppo.orig_downloaded_data.reindex(all_days).bfill().ffill()

        # Get the end date for each _date set
        self_end_date = self_data.index[-1]
        ppo_end_date = ppo_data.index[-1]
        # Get the smaller of the end dates between self and ppo
        best_end_date = min(self_end_date, ppo_end_date)
        # Truncate the data to the best end date
        self_data = self_data[:best_end_date]
        ppo_data = ppo_data[:best_end_date]
        # Get the best start_date
        self_start_date = self_data.index[0]
        ppo_start_date = ppo_data.index[0]
        # Truncate the start of the data to the best end date
        self_data = self_data[self_start_date:]
        ppo_data = ppo_data[ppo_start_date:]
        # Trunkate the data to pc_period_len if it is not None
        if pc_period_len is not None:
            self_data = self_data[-pc_period_len:]
            ppo_data = ppo_data[-pc_period_len:]
        # Once again, make sure that we have enough data points
        if len(self_data) < min_data_points:
            e_txt = f"self[{self.ticker}] has less than {min_data_points} data points."
            raise ValueError(e_txt)
        if len(ppo_data) < min_data_points:
            e_txt = f"ppo[{ppo.ticker}] has less than {min_data_points} data points."
            raise ValueError(e_txt)
        # Check that both data sets have the same length
        if len(self_data) != len(ppo_data):
            e_txt = f"Data lengths do not match: self[{len(self_data)}] != ppo[{len(ppo_data)}]"
            raise ValueError(e_txt)
        # Check that both datasets begin on the same day
        if self_data.index[0] != ppo_data.index[0]:
            e_txt = f"Data start dates do not match: self[{self_data.index[0]}] != ppo[{ppo_data.index[0]}]"
            raise ValueError(e_txt)
        # Check that both datasets end on the same day
        if self_data.index[-1] != ppo_data.index[-1]:
            e_txt = f"Data end dates do not match: self[{self_data.index[-1]}] != ppo[{ppo_data.index[-1]}]"
            raise ValueError(e_txt)

        # Save the start and end dates and period length of the corr
        corr_start_date = self_data.index[0].strftime("%Y-%m-%d %H:%M:%S")
        corr_end_date = self_data.index[-1].strftime("%Y-%m-%d %H:%M:%S")
        corr_period_len = len(self_data)

        # Get up days vs down days
        close_col = 'Close'
        self_closes = self_data[close_col]
        self_closes = self_closes.bfill().ffill()
        if close_col not in self_data.columns:
            close_col = 'Adj Close'
        self_trends = [1 if self_data[close_col].iloc[i] > self_data['Open'].iloc[i] else -1 for i in
                       range(len(self_data))]
        close_col = 'Close'
        ppo_closes = ppo_data[close_col]
        ppo_closes = ppo_closes.bfill().ffill()
        if close_col not in ppo_data.columns:
            close_col = 'Adj Close'
        ppo_trends = [1 if ppo_data[close_col].iloc[i] > ppo_data['Open'].iloc[i] else -1 for i in range(len(ppo_data))]

        """
         Calculate the correlations:
            1. Pearson: Calculate Pearson correlations using raw closing pricess and normalized closing prices.
            2. Spearman: Calculate Spearman correlations trands.
            3. Kendall: Calculate Kendall correlations trands.   
        """
        try_tracker = 'Before Correlation Functions'
        try:
            corr_list = [self_trends[i] + ppo_trends[i] for i in range(len(self_trends))]
            # Concatinage self_trends with ppo_trends into one dataframe whose columns are stock_a and stock_b
            corr_trends_df = pd.DataFrame({'stock_a': self_trends, 'stock_b': ppo_trends})
            corr_close_df = pd.DataFrame({'stock_a': self_closes, 'stock_b': ppo_closes})
            corr_close_df = corr_close_df.bfill().ffill()
            normed_close_df = (corr_trends_df - corr_trends_df.mean()) / corr_trends_df.std()
            # Perform Pearson Correlation on the raw closing prices and the normalized closing prices
            try_tracker = 'Person Correlation (Raw Closes)'
            pearson_corr_raw_matrix = corr_close_df.corr(method='pearson')
            try_tracker = 'Person Correlation (Raw Closes)'
            pearson_corr_nrm_matrix = normed_close_df.corr(method='pearson')
            # Perform Spearman and Kendall Correlation on the trends for the same timeframe
            try_tracker = 'Spearman Correlation'
            spearman_corr_matrix = corr_trends_df.corr(method='spearman')
            try_tracker = 'Kendall Correlation'
            kendall_corr_matrix = corr_trends_df.corr(method='kendall')
            try_tracker = 'After Kendall Correlation'
            # Get the Pearson corr values
            pearson_raw_corr = pearson_corr_raw_matrix.loc['stock_a']['stock_b']
            pearson_nrm_corr = pearson_corr_nrm_matrix.loc['stock_a']['stock_b']
            # Get the Spearman and Kendall corr values
            spearman_corr = spearman_corr_matrix.loc['stock_a']['stock_b']
            kendall_corr = kendall_corr_matrix.loc['stock_a']['stock_b']
            # Perform a coinegration test on the closing prices of the two stocks
            # Coint() returns coint_test(t-stat, p-value, [crit_values])
            try_tracker = 'Cointegration Test'
            coint_test = coint(self_closes, ppo_closes)
            try_tracker = 'After Cointegration Test'
            # Get the coinegration test result values
            # If this number is zero or greater, the two series are cointegrated.
            is_cointegrated = False
            if coint_test[0] < min(coint_test[2]) and coint_test[1] < 0.05:
                is_cointegrated = True
            coint_dict = {'is_cointegrated': is_cointegrated, 't_stat': coint_test[0],
                          'p_val': coint_test[1], 'crit_val': list(coint_test[2])}
            # Perform an ADF test on the spread between the two stocks.
            # Augmented Dickey Fuller Test: This is a test for stationarity in the spread data between 2 stocks.
            # This is required for Pairs Trading. We want the combination of Coinegration and Stationarity.
            # - Get the spread between the two stocks via self_closes and ppo_closes
            spread = self_closes - ppo_closes
            spread = spread.bfill().ffill()
            # Perform the ADF test on the spread data
            try_tracker = 'ADF Test'
            adf_result = sm.tsa.stattools.adfuller(spread, store=True, regresults=False)
            try_tracker = 'AFter ADF Test'
            # Gather the ADF test results
            is_stationary = False
            if adf_result[0] < min(adf_result[2].values()) and adf_result[1] < 0.05:
                is_stationary = True
            adf_dict = {'is_stationary': is_stationary, 'adf_stat': adf_result[0], 'p_val': adf_result[1],
                        'crit_val': adf_result[2]}
        except Exception as e:
            err_msg = (f"Error: In periodic_correlation(): {try_tracker} - " +
                       f"self.ticker[{self.ticker}] ppo.ticker[{ppo.ticker}] period[{self.period}]\n{e}")
            self.logger.error(err_msg)
            raise ValueError(f"Error: {err_msg}")

        total_days = len(corr_list)
        correlated_days = corr_list.count(2) + corr_list.count(-2)
        uncorrelated_days = corr_list.count(0)
        pct_corr = correlated_days / total_days
        pct_uncorr = uncorrelated_days / total_days
        # self.logger.info(f"Days: {total_days} Correlated Days: {correlated_days}  Uncorrelated Days: {uncorrelated_days}")
        # self.logger.info(f"Correlated Days: {pct_corr}%  Uncorrelated Days: {pct_uncorr}%")
        coint_stationary = False
        if is_cointegrated and is_stationary:
            coint_stationary = True
        ret_dict = {'total_days': total_days,
                    'correlated_days': correlated_days,
                    'uncorrelated_days': uncorrelated_days,
                    'pct_corr': pct_corr,
                    'pct_uncorr': pct_uncorr,
                    'pearson_raw_corr': pearson_raw_corr,
                    'pearson_nrm_corr': pearson_nrm_corr,
                    'spearman_corr': spearman_corr,
                    'kendall_corr': kendall_corr,
                    'avg_corr': (pearson_raw_corr + pearson_nrm_corr + spearman_corr + kendall_corr) / 4,
                    'coint_stationary': coint_stationary,
                    'coint_test': coint_dict,
                    'adf_test': adf_dict,
                    'corr_period_len': corr_period_len,
                    'start_date': corr_start_date,
                    'end_date': corr_end_date,
                    }

        return ret_dict

    def groq_sentiment(self):

        if self.ticker[0] == '^':
            # Indexes can't support sentiment analysis.
            self.sentiment_json = {}
            self.sentiment_text = f'Indexes [{self.ticker}] do not support sentiment analysis.'
            return

        stock = self.yf.Ticker(self.ticker)
        # balance_sheet = stock.balance_sheet
        balance_sheet = stock.quarterly_balance_sheet
        income_statement = stock.financials
        # print("=== Balance Sheets ===")
        # print(balance_sheet)
        # print("\n\n=== Income Statements ===")
        # print(income_statement)

        # Send an API call to GROQ LLM to perform sentiment analysis on the balance sheet data
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f'''
                     <promptInputs>
                         <Purpose>
                         Please perform a critical analyze the following balance sheets and income statements and 
                         give me a review of the company from a financial perspective and be critical of values from period to
                         period and consider if missing values indicate mis-reporting of data. And, add a summary of sentiment
                         analysis of the company from the viewpoint of board members, from the viewpoint of shareholders, and 
                         from the viewpoint of short sellers. Finally, create a sentiment analysis score for the company from 1 to 5,
                         where 1 is very negative and 5 is very positive. Separate each section into its json attribute.
                         Place the JSON output between the "<JsonOutputFormat>" and "</JsonOutputFormat>" tags, at the start of the response.
                         Place the sentiment text output between the "<sentimentTextOutput>" and "</sentimentTextOutput>" tags, at a the end of the response.
                         </Purpose>
                         <BalanceSheet>
                         {balance_sheet}
                         </BalanceSheet>
                         <IncomeStatements> 
                         {income_statement}
                         </IncomeStatements>
                     </promptInputs>
                     <exampleOutput>
                         <sentimentTextOutput> 
                         Here's a breakdown of the analysis:

                         **Balance Sheet Analysis**

                         * Treasury shares number: No change, as NaN values are present.
                         * Ordinary shares number: Stable, with no significant changes.
                         * Net debt: Increased significantly, which may be a concern.
                         * Cash and cash equivalents: Increased significantly, indicating sufficient liquidity.

                         **Income Statement Analysis**

                         * Net income from continuing operation net minority interest: Increased significantly, a positive sign.
                         * EBITDA: Consistently positive, indicating a healthy operating performance.
                         * Interest expense: Decreasing, which is a positive trend.
                         * Research and development expenses: Increasing, which may be a strategic investment for future growth.

                         **Critical Analysis**

                         * Missing values: Some incomplete financial statements, which may indicate a lack of transparency or mis-reporting of data.
                         * Debt level: Increasing, which may be a concern for investors and creditors.
                         * Tangible book value: No information provided, which makes it difficult to assess the company's financial health.
                         * Cash convertibility: Sufficient liquidity, as indicated by the increased cash and cash equivalents.

                         **Sentiment Analysis**

                         * Board members: 3 (neutral), as they may be concerned about the increasing debt level but optimistic about the company's growth prospects.
                         * Shareholders: 3 (neutral), as they may be pleased with the increasing net income but concerned about the debt level and missing values.
                         * Short sellers: 2 (somewhat negative), as they may be skeptical about the company's ability to sustain its growth and concerned about the increasing debt level.

                         **Overall Sentiment Score**

                         * 2.67 (somewhat positive), as the company's financial performance is generally positive, but concerns about debt level and missing values exist.
                         </sentimentTextOutput> 
                         <JsonOutputFormat> 
                         {{
                          "balance_sheet_analysis": {{
                            "treasury_shares_number": "increased significantly",
                            "ordinary_shares_number": "increasing",
                            "net_debt": "increased significantly",
                            "cash_and_cash_equivalents": "increased significantly"
                          }},
                          "income_statement_analysis": {{
                            "net_income_from_continuing_operation_net_minority_interest": "significant losses",
                            "ebitda": "consistently negative",
                            "interest_expense": "increasing",
                            "research_and_development_expenses": "increasing"
                          }},
                          "critical_analysis": {{
                            "missing_values": "incomplete financial statements",
                            "debt_level": "increasing",
                            "ebitda": "negative",
                            "net_income": "significant losses"
                          }},
                          "sentiment_analysis": {{
                            "board_members": 2,
                            "shareholders": 2,
                            "short_sellers": 4
                          }},
                          "overall_sentiment_score": 2.33
                         }}  
                         </JsonOutputFormat> 
                     </exampleOutput>
                     '''
                }
            ],
            model="llama3-70b-8192",
        )

        try:
            content = chat_completion.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error: Could call on Groq for sentiment on {self.ticker}.\n{e}")
            return

        # Extract the JSON response from the content using a regular expression
        jsn_matches = re.match(r'.*(\<JsonOutputFormat\>)\n(.*)\n(\<\/JsonOutputFormat\>)\n(.*)',
                               content, re.DOTALL | re.MULTILINE)
        txt_matches = re.match(r'.*(\<sentimentTextOutput\>)\n(.*)\n(\<\/sentimentTextOutput\>)',
                               content, re.DOTALL | re.MULTILINE)
        if jsn_matches is not None and txt_matches is not None:
            self.sentiment_text = ''
            self.sentiment_json = {}
            try:
                jsn_str = jsn_matches.group(2)
                txt_str = txt_matches.group(2)
                self.sentiment_text = txt_str.strip()
                self.sentiment_json = json.loads(jsn_str.strip())
            except Exception as e:
                self.sentiment_json = {}
                self.logger.warn(f"Failed to parse JSON response from Groq for sentiment on {self.ticker}.\n{e}")
                self.sentiment_text = content.strip()
        else:
            self.sentiment_json = {}
            self.sentiment_text = ''

    def seasonality(self, save_chart: bool = False,
                    show_chart: bool = False,
                    sd_period_len: int = 30):
        """
        Analyze the seasonality of the data in orig_data.
        Use statsmodels.api to analyze the seasonality of the data.
        If we don't have enough data, to perform the seasonality analysis,
        we will download more data.

        Seasonality requires at enough data/observations for the given period,
        such that there are at least 2 cycles for the given observation period.
        ie. observations:00 / period:50 = 2 cycles.

        The seasonality analysis requires enough data to perform the analysis.
        Thus, it typically requires at least 2 years of data. And, it needs
        to block, for data download. So we should call it from the
        cache_training_data() method, and have the that process always
        download at least 2 years of data.

        *** We should also add database storage for the data, so we can that
            we can minimize the data downloads and only download the latest data
            as needed to keep the data fresh.

        :return:
        """

        if self.orig_data is None or len(self.orig_data) < int(sd_period_len) * 2:
            # Fetch data from Yahoo Finance if needed...
            self.logger.info(f"*** Not enough data to perform seasonality analysis. sd_period_len[{sd_period_len}]")
            self.logger.info(f"*** Downloading more data to perform the analysis.")
            # Set date_start to 2 years before today's date aligned to a Monday
            self.fetch_data_yahoo(self.ticker, self.date_start, self.date_end, self.period)

        data = self.orig_data

        if 'Date' in data.columns and 'Date' not in data.index:
            data.set_index('Date', inplace=True)
            # Add an additional row to data for the next day's prediction
            last_date = data.index[-1]
            next_date = last_date + self.next_timedelta()
        else:
            # Add an additional row to data for the next day's prediction
            last_date = self.date_data[self.date_data.index[-1]]
            next_date = last_date + self.next_timedelta()

        close_col = 'Close'
        if close_col not in data.columns:
            close_col = 'Adj Close'
        new_row = {"Date": next_date,
                   "Open": data[close_col].iloc[-1], "High": data[close_col].iloc[-1],
                   "Low": data[close_col].iloc[-1], "Close": data[close_col].iloc[-1],
                   "Adj Close": data['Adj Close'].iloc[-1], "Volume": 0}
        # Set the new row to the next day's date...
        new_row = pd.DataFrame(new_row, index=[next_date])
        # Append a place holder day for the prediction...
        data = pd.concat([data, new_row], axis=0)

        data.ffill(inplace=True)

        try:
            seasonal_dec = sm.tsa.seasonal_decompose(data[close_col], model='additive', period=sd_period_len)
        except Exception as e:
            self.logger.error(
                f"Error: Could not perform seasonal decomposition for [{self.ticker}] [{self.period}]\n{e}")
            return None

        # Save the seasonal decomposition data
        # Copy it out of the seasonal_dec object, so it can be pickled
        self.seasonal_dec = seasonal_dec.__dict__.copy()

        if save_chart or show_chart:
            fig = seasonal_dec.plot(observed=True, seasonal=True,
                                    trend=True, weights=True, resid=True)
            plt.dpi = 600
            if save_chart:
                chart_path = self.chart_dir + f"{self.ticker}_seasonality_p({sd_period_len})_{self.period}_{self.date_end}.png"
                fig.savefig(chart_path, dpi=600)
                self.seasonal_chart_path = chart_path

            if show_chart:
                seasonal_dec.plot().show(fig)

        return seasonal_dec

    @staticmethod
    def serialize(obj):
        # Compress the object
        return lzma.compress(dill.dumps(obj))

    @staticmethod
    def unserialize(obj):
        # Decompress the object
        return dill.loads(lzma.decompress(obj))

    def serialize_me(self):
        # Compress the object
        return PricePredict.serialize(self)
