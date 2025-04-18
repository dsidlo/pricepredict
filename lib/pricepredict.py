"""
Class: PricPredict

When instantiated, this class will till stock date from Yahoo Finance, augment the data with technical indicators,
and train an LSTM model to predict the price of a stock.

See tests for examples of how to use this class.

"""
import datetime
import time
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import os
import pandas as pd
import pandas_ta as ta
import re
import sys
import tensorflow as tf
import yfinance as yf
import yfinance_cache as yfc
import logging
import statsmodels.api as sm
import json
import lzma
import dill
import pymc as pm
import arviz as az
import tf_keras as keras

from typing import Callable
from decimal import Decimal
from io import StringIO
from tf_keras.layers import Dense, LSTM, Input, Activation
from tf_keras.models import Model
from tf_keras.optimizers import Adam
from tf_keras.callbacks import CSVLogger, EarlyStopping, TensorBoard
from tf_keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
from groq import Groq
from bayes_opt import BayesianOptimization
from silence_tensorflow import silence_tensorflow
from statsmodels.tsa.stattools import coint

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

"""
Notes: Regarding training...
Training the model should requires at least 2 years of prior data for a daily period. 
For other periods, that's from 600 to 800 prior periods.

Notes: Regarding prediction...
To perform a prediction, the model requires enough data to fill the back_candles period,
and enough data to fill have data for any added technical indicators.

Note: Following are some functions that are used as Keras model metrics.
      These functions require the decoration @keras.saving.register_keras_serializable()  
      that allows keras to find these functions upon deserialization of associated models
      or a PricePredict object.  
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
        self.train_split: Decimal = None
        self.X: list[Decimal] = []
        self.y: list[Decimal] = []


"""
Use for a custom metric where we are looking for a model that more accurately
predicts the trend direction.

----- Custom gradient based trend correlation Metric Functions -----
"""
def correlation_loss(y_true, y_pred):
    # Compute correlation
    mean_true = tf.reduce_mean(y_true)
    mean_pred = tf.reduce_mean(y_pred)
    cov = tf.reduce_mean((y_true - mean_true) * (y_pred - mean_pred))
    std_true = tf.math.reduce_std(y_true)
    std_pred = tf.math.reduce_std(y_pred)
    corr = cov / (std_true * std_pred)

    # Convert correlation to loss (maximize correlation means minimize loss)
    return 1 - corr  # High correlation gives low loss


# ----- Combined Loss Function  mae and trend correlation -----
@keras.saving.register_keras_serializable(package='pricepredict', name='trend_loss')
def trend_loss(y_true, y_pred):
    # Example: A simple trend loss where we want predictions to follow the trend of true data
    # Here, we're checking if the direction of change matches
    diff_true = y_true[:, 1:] - y_true[:, :-1]
    diff_pred = y_pred[:, 1:] - y_pred[:, :-1]

    # Sign of the difference gives us direction
    sign_true = tf.sign(diff_true)
    sign_pred = tf.sign(diff_pred)

    # Count correct trend predictions
    correct_trends = tf.reduce_sum(tf.cast(tf.equal(sign_true, sign_pred), tf.float32))

    # Total number of trend comparisons
    total_trends = tf.cast(tf.size(diff_true), tf.float32)

    # Calculate the proportion of correct trend predictions
    return 1.0 - (correct_trends / tf.maximum(total_trends, 1.0))


@keras.saving.register_keras_serializable(package='pricepredict', name='trend_corr_mae_loss')
def trend_corr_mae_loss(y_true, y_pred):
    # Compute both losses
    mse = tf.keras.losses.MeanSquaredError()
    price_loss = mse(y_true, y_pred)
    trend_loss_value = trend_loss(y_true, y_pred)

    # You can weight these losses based on their importance
    alpha = 0.5  # Example weight for price loss
    beta = 0.5  # Example weight for trend loss

    return tf.cast(alpha * price_loss + beta * trend_loss_value, tf.float32)


"""
------------ End of Custom Metric Functions ------------ 
"""


# ================================
# This is the PricePredict class.
# ================================
@keras.saving.register_keras_serializable(package='pricepredict', name='PricePredict')
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
                 period=PeriodDaily,  # The period for the data (D, W)
                 # Directories
                 model_dir='./models/',  # The directory where the model is saved
                 chart_dir='./charts/',  # The directory where the charts are saved
                 preds_dir='./predictions/',  # The directory where the predictions are saved
                 ppo_dir='./ppo/',  # The directory where the partial prediction optimization data is saved
                 # Training Parameters
                 back_candles=15,  # The number of candles to look back for each period.
                 train_split=0.8,  # The value for splitting data into training and testing.
                 batch_size=30,  # The batch size for training the model.
                 epochs=50,  # The number of epochs for training the model.
                 shuffle=True,  # Shuffle the data for training the model.
                 force_training=False,  # Force training the model
                 # Logging and Debugging
                 keras_callbacks: [Callable] = None,  # An array of keras callbacks for model.fit(().
                 tf_logs_dir='./logs/fit/',  # The keras fit logs
                 tf_profiler=False,  # Use the tensorflow profiler
                 keras_log='PricePredict_keras.log',  # The keras log file
                 verbose=True,  # This Class' Print debug information
                 keras_verbosity=0, # The keras verbosity level
                 logger=None,  # The mylogger for this object
                 logger_file_path=None,  # The path to the log file
                 log_level=None,  # The logging level
                 # yfinance
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
        :param train_split:
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
        self.ppo_dir = ppo_dir  # The directory where the partial prediction optimization data is saved
        self.ppo_file = None  # The file where the partial prediction optimization data is saved
        self.tf_logs_dir = tf_logs_dir  # The keras fit logs
        self.tf_profiler = tf_profiler  # Use the tensorflow profiler
        self.tf_summary_step = 0  # The step for the tf.summary.experimental.scalar()
        # -------------------------------
        self.period = period  # The period for the data (D, W)
        self.model = None  # The current loaded model
        # ---- Bayesian Optimization ----
        self.bayes_best_loss = None  # The best loss for prediction from the bayesian optimization
        self.bayes_best_trend = None  # The best trend from the bayesian optimization
        self.bayes_best_pred_model = None  # The best bayesian optimized model, temp holder
        self.bayes_best_pred_hp = None  # The best hyperparameters from the bayesian optimization
        self.bayes_best_pred_analysis = None  # The best analysis from the bayesian optimization
        self.bayes_opt_hypers = {}  # The optimized hyperparameters in ['max'] and other data.
        self.scaler = None  # The current models scaler
        self.verbose = verbose  # Print debug information
        self.ticker = ticker  # The ticker for the data
        self.ticker_data = None  # The data for the ticker (Long Name, last price, etc.)
        self.date_start = ''  # The start date for the data
        self.date_end = ''  # The end date for the data
        self.orig_data = None  # Save the originally downloaded data for later use.
        self.orig_downloaded_data = None  # Save the originally downloaded data for later use.
        self.cached_data = None  # Interpolated cached data
        self.missing_rows_analysis = None  # Save the missing rows analysis for later review.
        self.date_data = None  # Save the date data for later use.
        self.force_training = force_training  # Force training the model
        # Training Parameters
        self.shuffle = shuffle  # Shuffle the data for training the model.
        self.train_split = train_split  # The validation split used during training.
        self.split_limit = None  # Derived as the len(data_) * train_split(%) before training
        self.batch_size = batch_size  # The batch size for training the model.
        self.epochs = epochs  # The number of epochs for training the model.
        self.keras_callbacks = keras_callbacks  # The keras callbacks for training the model.
        # Training and Data/Results
        self.X = None  # 3D array of training data.
        self.y = None  # Target values (Adj Close)
        self.X_train = None  # Training data
        self.X_test = None  # Test data
        self.y_test = None  # Test data
        self.y_test_closes = None  # Test data closes
        self.y_train = None  # Training data
        self.y_pred = None  # The prediction unscaled from model.predict()
        self.mean_squared_error = None  # The mean squared error for the model
        self.target_close = None  # The target close
        self.target_high = None  # The target high
        self.target_low = None  # The target low
        self.dateStart_train = None  # The start date for the training period
        self.dateEnd_train = None  # The end date for the training period
        self.aug_data = None  # Save the augmented data for later use.
        self.features = None  # The number of features in the data
        self.targets = None  # The number of targets (predictions) in the data
        self.data_scaled = None  # Save the scaled data for later use.
        self.pred = None  # The rescaled predictions (3 columns)
        self.dateStart_pred = None  # The start date for the prediction period
        self.dateEnd_pred = None  # The end date for the prediction period
        self.pred_class = None  # The prediction class
        self.pred_close = None  # The adjusted prediction close
        self.pred_high = None  # The adjusted prediction close
        self.pred_low = None  # The adjusted prediction close
        self.pred_dates = None  # The Data series that is aligned to pred(s) and adj_pred(s)
        self.adj_pred = None  # The adjusted predictions (3 columns)
        self.adj_pred_class = None  # The adjusted prediction class
        self.adj_pred_close = None  # The adjusted prediction close
        self.adj_pred_high = None  # The adjusted prediction high
        self.adj_pred_low = None  # The adjusted prediction low
        self.dateStart_train = None  # The start date for the training period
        self.analysis = None  # The analysis of the model
        self.analysis_path = None  # The path to the analysis file
        self.trends_close = None  # The trends in the close
        self.trends_adj_close = None  # The trends in the adjusted predictions
        self.trends_corr = None  # The correlation of the trends (actual close vs. adj predictions)
        self.back_candles = back_candles  # The number of candles to look back for each period.
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
        self.data_fetch_count = 0  # The data fetch count
        self.cache_fetch_count = 0  # The cache fetch count
        self.cache_update_count = 0  # The cache update count
        self.spread_analysis = {}  # Spread analysis against other tickers
        self.logger = None  # The mylogger for this object

        # Create a mylogger for this object.
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        if self.tf_profiler and (self.tf_logs_dir is None or self.keras_callbacks is None):
            e_txt = f"Error: if tf_profiler is True, you must set tf_logs_dir and keras_callbacks."
            self.logger.error(e_txt)
            raise ValueError(e_txt)

        silence_tensorflow()

        # # Set the objects logging to stdout by default for testing.
        # # The mylogger and the logging-level can be overridden by the calling program
        # # once the object is instantiated.
        # if self.mylogger.hasHandlers():
        #     self.mylogger.handlers.clear()
        # if tf.get_logger().hasHandlers():
        #     tf.get_logger().handlers.clear()

        # # Turn off this objects logging to stdout
        # self.mylogger.propagate = False
        # # Turn off tensorflow logging to stdout
        # tf.get_logger().propagate = False

        # # Set the mylogger to stdout by default.
        if logger_file_path is not None:
            self.logger.addHandler(logging.FileHandler(filename=logger_file_path))
            tf.get_logger().addHandler(logging.FileHandler(filename=logger_file_path))

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

    def get_config(self):
        pass

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

    def invalidate_cache(self):
        """
        Invalidate the data caches.

        Caches:
        - self.original_data: The original data from Yahoo Finance.
        - self.cached_data: The interpolated data.
        - self.orig_data: The latest interpolated data from yahoo or from the cache
        - self.orig_downloaded_data: The unaggregated data from the cache (if period is weekly)
        - self.missing_rows_analysis: The missing rows analysis from the cache.

        """
        self.orig_data = None
        self.orig_downloaded_data = None
        self.cached_data = None
        self.orig_downloaded_data = None
        self.missing_rows_analysis = None

    def fetch_data_yahoo(self, ticker: str = None,
                         period: str = None,
                         date_start: str = None, date_end: str = None,
                         force_fetch=False):
        """
        Load data from Yahoo Finance.
        Caveats: Yahoo download data may not be the same as the original data from the exchange.

        Note: self.date_start and self.date_end are set based on parameters of the last
              call to this function. They are not set by the data in the self.orig_data.

        - Required Parameters:
        :param ticker:
        :param date_start:
        :param date_end:

        - Optional
        :option period:      # The period for the data (D, W)
        :parameter force_fetch:       # Force the fetch of the data from Yahoo Finance

        - Returns:
        :return:
            data,           # An array of data Open, High, Low, Close, Adj Close, Volume
            feature_cnt:    # The number of features in the data
        """

        xyf = self.yf

        if ticker is None:
            ticker = self.ticker

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
                raise ValueError(f"period[{period}]: may only be \"{'", "'.join(PricePredict.PeriodValues)}\"")
            self.period = period

        if self.verbose:
            self.logger.info(f"Loading Data for: {ticker}  from: {date_start} to {date_end}, Period: {self.period}")

        if force_fetch:
            self.logger.info(f"Force Fetch: Loading Data for: {ticker}  from: {date_start} to {date_end}, Period: {self.period}")
            self.logger.info("Force Fetch: Clearing the caches.")
            self.orig_data = None
            self.orig_downloaded_data = None
            self.cached_data = None
            self.missing_rows_analysis = None

        if period is not None:
            self.period = period

        # ------ Load data from Cache if available ------

        # See if the requested data is in the cache...
        # - Does self.cached_data data data?
        # - Check the start and end dates to see if the requested data is in self.orig_data.
        # - If the requested data is in self.cached_data, then return the data from self.cached_data.
        # - Place data requested data from the cache into self.orig_data (which always contains the latest
        #   requested data).
        dt_start = datetime.strptime(date_start, '%Y-%m-%d')
        dt_end = datetime.strptime(date_end, '%Y-%m-%d')
        # TODO: Adjust dt_start and dt_end for Weekly data.
        if self.cached_data is not None and force_fetch is False:
            if self.cached_data.index[0] <= dt_start and self.cached_data.index[-1] >= dt_end:
                if self.period == self.PeriodDaily:
                    # Return Data from the cache.
                    # Get the data from the self.orig_data using the date range.
                    self.date_start = date_start
                    self.date_end = date_end
                    self.orig_data = self.cached_data.loc[date_start:date_end]
                    self.date_data = pd.Series(self.orig_data.index)
                    self.cache_fetch_count += 1
                    return self.orig_data, self.orig_data.shape[1]
                elif self.period == self.PeriodWeekly:
                    # Return Weekly Data from the cache.
                    # Get the data from the self.orig_data using the date range.
                    # The date aggregation returns data indexed by the Sunday date.
                    # Calculate the Sunday for the date_start.
                    date_obj = datetime.strptime(date_start, '%Y-%m-%d')
                    date_start_sunday = date_obj - timedelta(days=date_obj.weekday())
                    w_date_start = date_start_sunday.strftime('%Y-%m-%d')
                    # Calculate the Sunday date_end.
                    date_obj = datetime.strptime(date_end, '%Y-%m-%d')
                    date_end_sunday = date_obj + timedelta(days=6) - timedelta(days=date_obj.weekday())
                    w_date_end= date_end_sunday.strftime('%Y-%m-%d')

                    wkly_data = self.cached_data.loc[w_date_start:w_date_end]

                    # Set the objects date_start and date_end properties to the last fetch.
                    self.date_start = date_start
                    self.date_end = date_end
                    self.orig_data = wkly_data
                    self.cache_fetch_count += 1
                    return wkly_data, wkly_data.shape[1]

        # ------ Fetch the data from Yahoo Finance ------

        # Remove "Test-" from the start of the ticker (used for model testing)
        f_ticker = re.sub(r'^Test-', '', ticker)
        retry_cnt = 0
        retry_needed = False
        retry_success = False
        net_data = []
        self.data_fetch_count += 1
        while True:
            try:
                if self.period in [PricePredict.PeriodWeekly, PricePredict.PeriodDaily]:
                    net_data = xyf.download(tickers=[f_ticker], start=date_start, end=date_end)
                else:
                    net_data = xyf.download(tickers=f_ticker, start=date_start, end=date_end, interval=self.period)
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

        if len(net_data) == 0:
            self.logger.error(f"Error: No data for {ticker} from {date_start} to {date_end}")
            return None, None

        # if 'Date' in data.index.names:
        #     # Reset the index to release the date from the index.
        #     data.reset_index()

        # If the column is a tuple, then we only want the first part of the tuple.
        if len(net_data) > 0:
            cols = net_data.columns
            if type(cols[0]) == tuple:
                cols = [col[0] for col in cols]
                net_data.columns = cols

        if self.yf_cached and 'Adj Close' not in net_data.columns:
            net_data['Adj Close'] = net_data['Close']
            net_data = net_data[['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']]

        # ----- Date Cleanup after data fetch -----

        data_start_date = net_data.index[0]
        data_end_date = net_data.index[-1]
        # Create a data frome that contains all business days.
        all_days = pd.date_range(start=data_start_date, end=data_end_date, freq='B')
        # Add rows into self_data that are not in ppo_data, such that fields other than 'Date' are NaN
        data_nan_filled = net_data.copy(deep=True)
        data_nan_filled.index = pd.to_datetime(data_nan_filled.index)
        data_nan_filled = data_nan_filled.reindex(all_days)
        # Interpolate the missing data in the dataframe, given the existing data.
        interpolated_data = data_nan_filled.interpolate(method='time')
        # Missing rows analysis on self_data
        # Step 1: Identify missing rows (where all data columns are NaN)
        missing_rows_mask = data_nan_filled.isnull().all(axis=1)
        # Step 2: Count the number of missing rows
        missing_rows = missing_rows_mask.sum()
        # Group by month to get the distribution
        missing_distribution = missing_rows_mask.groupby(pd.Grouper(freq='ME')).sum()
        # Make sure that the 'Adj Close' column exists.
        if 'Adj Close' not in interpolated_data.columns:
            interpolated_data['Adj Close'] = interpolated_data['Close']

        # Normalize the datetime index.
        if self.period not in [PricePredict.PeriodWeekly, PricePredict.PeriodDaily]:
            net_data.index = net_data.index.tz_localize(None)
            interpolated_data.index = interpolated_data.index.tz_localize(None)

        # Aggregate the data to a weekly period, if nd
        orig_data = net_data.copy(deep=True)
        if self.period == self.PeriodWeekly:
            interpolated_data = self.aggregate_data(interpolated_data, period=self.PeriodWeekly)
            # Make new copies of the data so that these separate dataframes don't share the same memory.

        # ------ At this point the final data to return is in "interpolated_data" ------

        if self.verbose:
            self.logger.info(f"data.len(): {len(orig_data)}  data.shape: {orig_data.shape}")

        # Data from clode in this function above...
        # - data: latest data just downloaded from Yahoo Finance.
        # - orig_data: "data" filled in and interpolated.

        # ----- Cache update logic -----

        # Check if the data lies byond the original cache.
        if self.cached_data is not None and force_fetch is False:
            cache_start_date = self.cached_data.index[0]
            cache_end_date = self.cached_data.index[-1]
            orig_start_date = self.orig_downloaded_data.index[0]
            orig_end_date = self.orig_downloaded_data.index[-1]
            intrp_data_start_date = interpolated_data.index[0]
            intrp_data_end_date = interpolated_data.index[-1]
            # Update the orig_downloaded_data and the cached_data.
            cache_updated = False
            data_start_date = data_start_date
            data_end_date = data_end_date
            if orig_start_date > data_start_date or orig_end_date < data_end_date:
                if cache_start_date > data_start_date and orig_start_date < data_end_date:
                    # If the latest data fully overlaps the cache, on both sides,
                    # then replace orig_downloaded_data with the latest data
                    # and replace cached_data with the latest interpolated_data.
                    self.orig_downloaded_data = orig_data
                    self.cached_data = interpolated_data
                    cache_updated = True
                else:
                    if orig_start_date > data_start_date:
                        # Append that backend unaltered data to orig_downloaded_data
                        # Append the backend interpolated_data to cached_data
                        # - Get the backend of the unaltered data that does not exist in
                        #   orig_downloaded_data.
                        be_net_data = net_data[net_data.index < orig_start_date].copy(deep=True)
                        # - Append the backend of the unaltered data to orig_downloaded_data.
                        if len(be_net_data) > 0:
                            self.orig_downloaded_data = pd.concat([be_net_data, self.orig_downloaded_data])
                            cache_updated = True
                    if orig_start_date < data_end_date:
                        # Append that frontend unaltered data to orig_downloaded_data
                        # Append the frontend interpolated_data to cached_data
                        # - Get the frontend of the unaltered data that does not exist in
                        #   orig_downloaded_data.
                        fe_net_data = net_data.loc[net_data.index > orig_start_date].copy(deep=True)
                        # - Append the frontend of the unaltered data to orig_downloaded_data.
                        if len(fe_net_data) > 0:
                            self.orig_downloaded_data = pd.concat([self.orig_downloaded_data, fe_net_data])
                            cache_updated = True
            # Update the cached_data if the data is within the cache.
            if cache_start_date > intrp_data_start_date or cache_end_date < intrp_data_end_date:
                if cache_start_date > intrp_data_start_date and cache_end_date < intrp_data_start_date:
                    # If the latest data fully overlaps the cache, on both sides,
                    # then replace orig_downloaded_data with the latest data
                    # and replace cached_data with the latest interpolated_data.
                    self.orig_downloaded_data = orig_data
                    self.cached_data = interpolated_data
                    cache_updated = True
                else:
                    if cache_start_date > intrp_data_start_date:
                        # Append that backend unaltered data to orig_downloaded_data
                        # Append the backend interpolated_data to cached_data
                        # - Get the backend of the unaltered data that does not exist in
                        #   orig_downloaded_data.
                        be_int_data = interpolated_data[interpolated_data.index > cache_start_date].copy(deep=True)
                        # - Append the backend of the unaltered data to orig_downloaded_data.
                        if len(be_int_data) > 0:
                            self.cached_data = pd.concat([be_int_data, self.cached_data])
                            cache_updated = True
                    if cache_end_date < intrp_data_start_date:
                        # Append that frontend unaltered data to orig_downloaded_data
                        # Append the frontend interpolated_data to cached_data
                        # - Get the frontend of the unaltered data that does not exist in
                        #   orig_downloaded_data.
                        fe_int_data = interpolated_data.loc[interpolated_data.index < cache_end_date].copy(deep=True)
                        # - Append the frontend of the unaltered data to orig_downloaded_data.
                        if len(fe_int_data) > 0:
                            self.cached_data = pd.concat([self.cached_data, fe_int_data])
                            cache_updated = True
            if cache_updated:
                self.cache_update_count += 1

        # Hold on the missing rows analysis for later review.
        self.missing_rows_analysis = {'missing_rows': missing_rows, 'missing_distribution': missing_distribution}

        # Data normally of use to the user and to this object.
        self.ticker = ticker
        self.date_start = date_start
        self.date_end = date_end
        self.orig_downloaded_data = net_data
        self.orig_data = interpolated_data
        self.cached_data = interpolated_data
        self.date_data = pd.Series(interpolated_data.index)

        return interpolated_data, interpolated_data.shape[1]

    def aggregate_data(self, data, period):
        """
        Aggregate the data to a weekly period.
        :param data:
        :return:
        """
        try:
            data = data.resample(period).agg(
                {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Adj Close': 'last', 'Volume': 'sum'})
        except Exception as e:
            e_txt = f"Error: Ticker[{self.ticker}:{self.period}] resample() failed for period [{period}].\n{e}"
            raise RuntimeError(e_txt)

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

        # mylogger.info("= Before Adding Indicators ========================================================")
        # mylogger.info(data.tail(10))

        # Make a copy of the data to that we don't modify orig_data
        aug_data = data.copy(deep=True)

        aug_data['RSI'] = ta.rsi(aug_data.Close, length=3);
        feature_cnt += 1
        # aug_data['EMAF']=ta.ema(aug_data.Close, length=3); feature_cnt += 1
        # aug_data['EMAM']=ta.ema(aug_data.Close, length=6); feature_cnt += 1
        aug_data['EMAS'] = ta.ema(aug_data.Close, length=9)
        feature_cnt += 1
        aug_data['DPO3'] = ta.dpo(aug_data.Close, length=3, lookahead=False, centered=False)
        feature_cnt += 1
        aug_data['DPO6'] = ta.dpo(aug_data.Close, length=6, lookahead=False, centered=False)
        feature_cnt += 1
        aug_data['DPO9'] = ta.dpo(aug_data.Close, length=9, lookahead=False, centered=False)
        feature_cnt += 1

        # mylogger.info("= After Adding DPO2 ========================================================")
        # mylogger.info(aug_data.tail(10))
        #
        # On Balance Volume
        if aug_data['Volume'].iloc[-1] > 0:
            aug_data = aug_data.join(ta.aobv(close=aug_data.Close, volume=aug_data.Volume,
                                             fast=3, slow=6, min_lookback=3, max_lookback=9))
            feature_cnt += 7  # ta.aobv adds 7 columns

        # mylogger.info("= After Adding APBV ========================================================")
        # mylogger.info(aug_data.tail(10))

        # Target is the difference between the adjusted close and the open price.
        aug_data['Target'] = aug_data['Adj Close'] - aug_data.Open
        feature_cnt += 1
        aug_data['TargetH'] = aug_data.High - aug_data.Open
        feature_cnt += 1
        aug_data['TargetL'] = aug_data.Low - aug_data.Open
        feature_cnt += 1
        # Shift the target up by one day.Target is the difference between the adjusted close and the open price.
        # That is, the target is the difference between the adjusted close and the open price.
        # Our model will predict the target close for the next day. So we shift the target up by one day.
        aug_data['Target'] = aug_data['Target'].shift(-1);
        aug_data['TargetH'] = aug_data['TargetH'].shift(-1)
        aug_data['TargetL'] = aug_data['TargetL'].shift(-1)

        # 1 if the price goes up, 0 if the price goes down.
        # Not a feature: Needed to test prediction accuracy.
        aug_data['TargetClass'] = [1 if aug_data['Target'].iloc[i] > 0 else 0 for i in range(len(aug_data))]
        target_cnt = 1

        # The TargetNextClose is the adjusted close price for the next day.
        # This is the value we want to predict.
        # Not a feature: Needed to test prediction accuracy.
        aug_data['TargetNextClose'] = aug_data['Adj Close'].shift(-1)
        target_cnt += 1
        # TargetNextHigh and TargetNextLow are the high and low prices for the next day.
        aug_data['TargetNextHigh'] = aug_data['High'].shift(-1)
        target_cnt += 1
        # TargetNextLow are the low prices for the next day.
        aug_data['TargetNextLow'] = aug_data['Low'].shift(-1)
        target_cnt += 1

        # Before scaling the aug_data, we need to use the last good value for rows that have NaN values.
        aug_data.ffill(inplace=True)

        # # Reset the index of the dataframe.
        # aug_data.reset_index(inplace=True)
        #
        # if 'Date' not in aug_data.columns and aug_data['index'].dtype == '<M8[ns]':
        #     # Rename the 'index' column to 'Date'
        #     aug_data.rename(columns={'index': 'Date'}, inplace=True)
        # if 'Date' not in aug_data.columns:
        #     raise ValueError("Error: 'Date' column not found in aug_data.")

        # Save the date aug_data for later use.
        if 'Date' in aug_data.columns:
            # Copy the 'Date' Datetime column to the dates_data as a DataFrame.
            dates_data = aug_data['Date'].copy()
            dates_data.set_index('Date', inplace=True)
            aug_data = aug_data.drop(['Date'], axis=1)
            feature_cnt -= 1
        elif 'Date' in data.index.names or 'Datetime' in data.index.names:
            # Copy the index, which is a DatetimeIndex to the dates_data as a DataFrame.
            aug_data.reset_index(inplace=True)
            # Copy the 'Date' Datetime column to the dates_data as a DataFrame.
            # dates_data is a DataFrame with the 'Date' column as the index, and no other columns.
            if 'Date' in aug_data.columns:
                # Normally, directly from yfinance, the column is named 'Date'.
                dates_data = aug_data['Date'].copy()
                aug_data = aug_data.drop(['Date'], axis=1)
                feature_cnt -= 1
            elif 'Datetime' in aug_data.columns:
                # After caching the data, the column is named 'Datetime'.
                dates_data = aug_data['Datetime'].copy()
                aug_data = aug_data.drop(['Datetime'], axis=1)
                feature_cnt -= 1
        elif aug_data.index.dtype == '<M8[ns]':
            # If the index is a DatetimeIndex, then we use the index as the date data.
            # This does not remove a feature
            dates_data = pd.Series(aug_data.index)
            aug_data.index.rename('Date', inplace=True)
            aug_data.reset_index(inplace=True)
            aug_data.drop(['Date'], axis=1, inplace=True)
            # Date col was never a feature, so we don't decrement feature_cnt.
            feature_cnt -= 1

        # Drop the 'Close' column.
        aug_data.drop(['Close'], axis=1, inplace=True);
        feature_cnt -= 1

        # Add one more row to the data file, this will be our next day's prediction.
        aug_data = pd.concat([aug_data, aug_data[-1:]])
        # # And, reindex the dataframe.
        # aug_data.reset_index(inplace=True)

        self.aug_data = aug_data.copy(deep=True)
        self.features = feature_cnt
        self.targets = target_cnt
        self.date_data = dates_data
        self.logger.debug(f'*~*~*~> self.data_data: [{type(self.date_data)}]')


        return aug_data, feature_cnt, target_cnt, self.date_data

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

        # Pull the Target Columns (Predictions) from the restored data.
        pred_restored_scale = pred_restored_scale[:, -self.targets:]
        self.pred = pred_restored_scale
        self.pred_class = pred_restored_scale[:, -4]
        self.pred_close = pred_restored_scale[:, -3]
        self.pred_high = pred_restored_scale[:, -2]
        self.pred_low = pred_restored_scale[:, -1]

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

        if self.verbose:
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

        if dateStart is None:
            dateStart = self.date_start
        if dateEnd is None:
            dateEnd = self.date_end

        # Verify that the modelDir exists.
        if not os.path.exists(modelDir):
            self.logger.error(f"Error: Model directory [{modelDir}] does not exist.")
            raise ValueError(f"Error: Model directory [{modelDir}] does not exist.")

        model_file = ticker + f"_{self.period}_" + dateStart + "_" + dateEnd + ".keras"
        # Python does not like the = sign in filenames.
        model_file = model_file.replace('=', '~')
        model_path = modelDir + model_file
        # Load the model
        model = keras.models.load_model(model_path)
        if model is None:
            self.logger.error(f"Error: load_model returned None: {model_path}")
            raise ValueError(f"Error: load_model returned None: {model_path}")
        self.logger.debug(f"Model Loaded: {model_path}")
        if self.verbose:
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
            # Match files that start with the ticker and end with .keras
            model_files = [f for f in os.listdir(self.model_dir) if f.split('_')[0] == ticker and f.endswith('.keras')]
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
        X, y = self.fetch_n_prep(symbol, dateStart_, dateEnd_,
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
        X, y = self.fetch_n_prep(symbol, dateStart_, dateEnd_,
                                 period=period, train_split=1)

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

        self.logger.debug(f"=== Started: Training and Predicting for [{self.ticker}:{self.period}] using cached data...")
        if self.model is None or force_training is True:
            tc = self.cached_train_data
            if tc is None:
                self.logger.error(
                    f"Error: No training data cached for [{self.ticker}:{self.period}]. Cached training data was expected.")
                raise ValueError(
                    f"Error: No training data cached for [{self.ticker}:{self.period}]. Cached training data was expected.")
            self.ticker = tc.symbol
            self.dateStart_train = tc.dateStart
            self.dateEnd_train = tc.dateEnd
            self.period = tc.period
            self.train_split = tc.train_split
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
            self.logger.info(f"Training [{self.ticker}:{self.period}] using cached data...")
            # Train a new model using the cached training data.
            model, y_pred, mse = self.train_model(self.X, self.y)
            if model is None:
                self.logger.error("Error: Could not train the model.")
                raise ValueError("Error: Could not train the model.")
            else:
                # Save the model
                self.model = model
                self.logger.info(f"Using existing model for [{self.ticker}:{self.period}], file-path {self.model_path}...")
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
                f"Exception Error: No prediction data cached for [{self.ticker}:{self.period}]. Cached prediction data was expected.")
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
            self.logger.error(f"Exception Error: Could not load prediction data for [{self.ticker}:{self.period}]: {e}")
            return

        # Make Predictions on all the data
        self.train_split = 1.0
        # Perform the prediction
        # This call will also save_plot the prediction data to this object.
        self.predict_price(self.X)

        if no_report is False:
            """
            - Produce a prediction chart.
            - Save the prediction data to a file or database.
            - Save to weekly or daily data to a file or database.
            - Save up/down corr data to a file or database.
            - Perform Seasonality Decomposition.
            - Save the Seasonality Decomposition to a file or database.
            """
            self.logger.info(f"Performing price prediction for [{self.ticker}:{self.period}] using cached data...")
            try:
                self.gen_prediction_chart(last_candles=75, save_plot=save_plot, show_plot=show_plot)
            except Exception as e:
                self.logger.error(f"Exception Error: Could not generate prediction chart for [{self.ticker}:{self.period}]: {e}")

            self.save_prediction_data()

        # Save current datetime of the last analysis.
        self.last_analysis = datetime.now()
        self.logger.debug(f"=== Completed: Training and Predicting for [{self.ticker}:{self.period}] using cached data...Done.")

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
        # mylogger.info(data_set_scaled[0].size)
        # data_set_scaled=data_set.values
        # mylogger.info(data_set_scaled.shape[0])

        # Create a 3D array of the data. X[features][periods][candles]
        # Where candles is the number of candles that rolls by 1 period for each period.
        for j in range(feature_cnt):  # data_set_scaled[0].size):# last 2 columns are target not X
            X.append([])
            for i in range(backcandles, data_set_scaled.shape[0]):  # backcandles+2
                X[j].append(data_set_scaled[i - backcandles:i, j])

        # mylogger.info("X.shape:", np.array(X).shape)
        X = np.array(X)
        # Move axis from 0 to position 2
        X = np.moveaxis(X, [0], [2])
        # mylogger.info("X.shape:", X.shape)

        # The last 4 columns are the Targets.
        # The ML model will learn to predict the 4 Target columns.
        X, yi = np.array(X), np.array(data_set_scaled[backcandles:, -self.targets:])
        y = np.reshape(yi, (len(yi), self.targets))

        self.logger.debug(f"data_set_scaled.shape: {data_set_scaled.shape}  X.shape: {X.shape}  y.shape: {y.shape}")

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

    def train_model(self, X, y,
                    train_split: Decimal = 0.8,
                    backcandles: int = 15,
                    shuffle: bool = True,
                    use_bayes_opt: bool = True,
                    # -- Hyperparameters
                    lstm_units: int = None,
                    lstm_dropout: float = None,
                    adam_learning_rate: float = None,
                    epochs: int = None,
                    batch_size: int = None,
                    hidden_layers: int = None,
                    hidden_layer_units: [int] = None,
                    ):
        """
        Train the model.
        Given the training data, split it into training and testing data.
        Train the model and return the model and the prediction.
        Adjust the prediction are not made here.

        :param X:
        :param y:
        # --- Optional Parameters
        :param train_split:
        :param backcandles:
        # --- Optimize Hyperparameters
        :param use_bayes_opt:
        :param bayes_opt:
        # --- Hyperparameters
        :param shuffle:
        :param epochs:
        :param adam_learning_rate:
        :param batch_size:
        :param lstm_units:
        :param lstm_dropout:
        :param hidden_layers:
        :param hidden_layer_units:

        :return model, y_pred, mse:

        """

        self.logger.debug(f"=== Training Model [{self.ticker}:{self.period}]...")

        # Get Hyperparameters from last Bayesian optimization, if available.
        if self.bayes_opt_hypers is not None and 'params' in self.bayes_opt_hypers:
            if batch_size is None:
                if 'batch_size' in self.bayes_opt_hypers['params']:
                    batch_size = int(round(self.bayes_opt_hypers['params']['batch_size']))
            if epochs is None:
                if 'epochs' in self.bayes_opt_hypers['params']:
                    epochs = int(round(self.bayes_opt_hypers['params']['epochs']))
            if lstm_units is None:
                if 'lstm_units' in self.bayes_opt_hypers['params']:
                    lstm_units = int(round(self.bayes_opt_hypers['params']['lstm_units']))
            if lstm_dropout is None:
                if 'lstm_dropout' in self.bayes_opt_hypers['params']:
                    lstm_dropout = self.bayes_opt_hypers['params']['lstm_dropout']
            if adam_learning_rate is None:
                if 'adam_learning_rate' in self.bayes_opt_hypers['params']:
                    adam_learning_rate = self.bayes_opt_hypers['params']['adam_learning_rate']
            if hidden_layers is None:
                if 'hidden_layers' in self.bayes_opt_hypers['params']:
                    hidden_layers = int(round(self.bayes_opt_hypers['params']['hidden_layers']))
                else:
                    hidden_layers = 0
            if hidden_layer_units is None:
                hul_arry = ['hidden_layer_units_0', 'hidden_layer_units_1', 'hidden_layer_units_2',]
                hidden_layer_units = []
                for i in range(hidden_layers):
                    if hul_arry[i] in self.bayes_opt_hypers['params']:
                        hidden_layer_units.append(int(round(self.bayes_opt_hypers['params'][hul_arry[i]])))

        # If the hyperparameters are still None, then set them to the default values.
        if batch_size is None:
            batch_size = 128
        if epochs is None:
            epochs = 100
        if lstm_units is None:
            lstm_units = 200
        if lstm_dropout is None:
            lstm_dropout = 0.2
        if adam_learning_rate is None:
            adam_learning_rate = 0.035
        if hidden_layers is None or hidden_layers == 0:
                hidden_layers = 0
                hidden_layer_units = []

        # Handle the optional parameters
        if train_split is None:
            train_split = self.train_split
        validation_split = 1 - train_split
        split_limit = int(len(X) * train_split)

        self.split_limit = split_limit

        self.back_candles = backcandles

        if shuffle is None and self.shuffle is not None:
            shuffle = self.shuffle

        data_set = self.data_scaled
        feature_cnt = self.features

        if use_bayes_opt is True:
            # Fetch optimized hyperparameters, if available.
            self.fetch_opt_hyperparameters(self.ticker)

        # Split the scaled data into training and testing
        # mylogger.info("lenX:",len(X), "splitLimit:",splitlimit)
        X_train, X_test = X[:split_limit], X[split_limit:]  # Training data, Test Data
        y_train, y_test = y[:split_limit], y[split_limit:]  # Training data, Test Data

        if len(hidden_layer_units) != hidden_layers:
            e_txt = (f"Error: Ticker[{self.ticker}:{self.period}] Hidden layers [{hidden_layers}]"
                     + f" and hidden_layer_units [{len(hidden_layer_units)}] do not match.")
            self.logger.error(e_txt)
            raise ValueError(e_txt)

        # lstm_units: int = 200,
        # lstm_dropout: Decimal = 0.2,
        # adam_learning_rate: Decimal = 0.035,
        # epochs: int = 100,
        # batch_size: int = 32,
        # hidden_layers: int = 0,
        # hidden_layer_units: [int] = None,

        # Create the LSTM model
        if self.model is None:
            self.logger.debug("Creating a new model...")
            lstm_input = Input(shape=(backcandles, feature_cnt), name='lstm_input')
            return_seq = hidden_layers > 1
            inputs = LSTM(lstm_units, return_sequences=return_seq,
                          name='lstm_layer_0', dropout=lstm_dropout)(lstm_input)
            # inputs = LSTM(lstm_units, name='lstm_layer_0', return_sequences=return_seq)(lstm_input)
            self.logger.debug(f'\nAdded LSTM Layer: 0  Units: {lstm_units}  Return Seq: {return_seq}')
            # Add LSTM layers hidden layers, as needed
            # - When there is only 1 LSTM Layer set return_seq to False.
            # - When we add hidden LSTM Layers, set return_seq to True for all but the last layer.
            for i in range(0, hidden_layers):
                # Only the last layer set set to False
                return_seq = (i + 1) < hidden_layers
                units = hidden_layer_units[i]
                inputs = LSTM(units, return_sequences=return_seq,
                                  name=f'lstm_hidden_{i}', dropout=lstm_dropout)(inputs)
                self.logger.debug(f'Added LSTM Layer: {i}  Units: {units}  Return Seq: {return_seq}')
            inputs = Dense(self.targets, name='dense_layer')(inputs)
            output = Activation('linear', name='output')(inputs)
            model = Model(inputs=lstm_input, outputs=output)
            adam = Adam(learning_rate=adam_learning_rate)
            model.compile(optimizer=adam, loss='mse', metrics=['mae', 'mse', 'accuracy'])
            self.logger.debug(f'Created new model: {model.summary()}')
        else:
            self.logger.debug("Using existing self.model...")
            model = self.model

        # Define the CSV mylogger
        csv_logger = CSVLogger('PricePred_keras_training_log.csv')

        try:
            # Train the model
            model.fit(x=X_train, y=y_train,
                      batch_size=batch_size, epochs=epochs,
                      shuffle=shuffle, validation_split=validation_split,
                      callbacks=[csv_logger],
                      verbose=self.keras_verbosity)
        except Exception as e:
            self.logger.error(f"Error: Training model [{self.ticker}:{self.period}].\n{e}")
            return None, None, None

        if len(X_test) > 0:
            y_pred = model.predict(X_test, verbose=self.keras_verbosity)
            self.y_pred = y_pred
            fy_pred = np.array(pd.DataFrame(y_pred).replace({np.nan: 0}))
            fy_test = np.array(pd.DataFrame(y_test).replace({np.nan: 0}))
            mse = mean_squared_error(fy_test, fy_pred)
            if self.verbose:
                self.logger.info(f"Mean Squared Error: {mse}")

            # Restore the scale of the prediction
            # - Save the rescaled prediction to this object (self.pred_xxx)
            pred_rescaled = self.restore_scale_pred(y_pred.reshape(-1, self.targets))
            # Adjust the prediction
            # - The prediction is the prior closing price + the delta of the prediction's
            #   last closing prices.
            self.adjust_prediction()
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
        self.target_close = np.array(self.orig_data['Adj Close'].iloc[backcandles + split_limit - 1:])
        self.target_high = np.array(self.orig_data['High'].iloc[backcandles + split_limit - 1:])
        self.target_low = np.array(self.orig_data['Low'].iloc[backcandles + split_limit - 1:])
        self.mean_squared_error = mse
        self.dateStart_train = self.date_data.iloc[0].strftime("%Y-%m-%d")
        self.dateEnd_train = self.date_data.iloc[-1].strftime("%Y-%m-%d")
        self.dateStart_pred = self.date_data.iloc[split_limit + 1].strftime("%Y-%m-%d")
        self.dateEnd_pred = self.date_data.iloc[-1].strftime("%Y-%m-%d")
        self.model = model

        self.logger.debug(f"=== Model Training Completed [{self.ticker}:{self.period}]...")

        return model, y_pred, mse


    def bayes_train_model(self, X, y,
                          train_split: float = 0.8, backcandles: int = 15,
                          shuffle: bool = True, validation_split: float = 0.2,
                          # --- Hyperparameters
                          lstm_units: int = 256,
                          lstm_dropout: float = 0.2,
                          adam_learning_rate: float = 0.035,
                          epochs: int = 100,
                          batch_size: int = 128,
                          hidden_layers: int = 0,
                          hidden_layer_units: [int] = None):
        """
        Train the model.
        Given the training data, split it into training and testing data.
        Train the model and return the model and the prediction.
        Adjust the prediction are not made here.

        :param X:
        :param y:
        :param train_split:
        :param backcandles:
        :param shuffle:
        :param validation_split:
        # --- Hyperparameters
        :param lstm_units:
        :param lstm_dropout:
        :param adam_learning_rate:
        :param epochs:
        :param batch_size:
        :param hidden_layers:
        :param hidden_layer_units:

        :return:
        """

        # Convert int hyperparameters to int
        # as the optimizer passes them as floats.
        lstm_units = int(lstm_units)
        epochs = int(epochs)
        batch_size = int(batch_size)
        hidden_layers = int(hidden_layers)
        hidden_layer_units = [int(hlu) for hlu in hidden_layer_units]

        # self.logger.debug(f"=== Training Model [{self.ticker}:{self.period}]...")
        # self.logger.debug(f"Split Percentage: {train_split}  Back Candles: {backcandles}")
        # self.logger.debug(f"Shuffle: {shuffle}  Validation Split: {validation_split}")
        # self.logger.debug(f"LSTM Units: {lstm_units}  LSTM Dropout: {lstm_dropout}")
        # self.logger.debug(f"Adam Learning Rate: {adam_learning_rate}")
        # self.logger.debug(f"Epochs: {epochs}  Batch Size: {batch_size}")
        self.logger.debug(f"Hidden Layers: {hidden_layers}  Hidden Layer Units: {hidden_layer_units}")
        # self.logger.debug(f"-----------------------------------------------------------------")
        # Handle the optional parameters
        if train_split is None:
            train_split = self.train_split
        split_limit = int(len(X) * train_split)

        self.split_limit = split_limit
        if backcandles is None:
            backcandles = self.back_candles

        data_set = self.data_scaled
        feature_cnt = self.features

        # self.logger.debug("Creating a new model...")

        if hidden_layers is None or hidden_layers == 0:
            hidden_layers = 0
            hidden_layer_units = []

        lstm_input = Input(shape=(backcandles, feature_cnt), name='lstm_input')
        return_seq = hidden_layers >= 1
        inputs = LSTM(lstm_units, return_sequences=return_seq,
                      name='lstm_layer_0', dropout=lstm_dropout)(lstm_input)

        # Add LSTM layers hidden layers, as needed
        i = None
        units = None
        return_seq = None
        try:
            for i in range(0, hidden_layers):
                # Only the last layer set set to False
                return_seq = (i + 1) < hidden_layers
                units = hidden_layer_units[i]
                inputs = LSTM(units, return_sequences=return_seq,
                              name=f'lstm_hidden_{i}', dropout=lstm_dropout)(inputs)
                self.logger.debug(f'Added LSTM Layer: {i}  Units: {units}  Return Seq: {return_seq}')
        except Exception as e:
            e_txt = f"Error: Ticker[{self.ticker}:{self.period}] Adding Hidden layer [{i}] units [{units}] return_seq [{return_seq}].\n{e}"
            self.logger.error(e_txt)
            raise ValueError(e_txt)

        inputs = Dense(self.targets, name='dense_layer')(inputs)
        output = Activation('linear', name='output')(inputs)
        model = Model(inputs=lstm_input, outputs=output)
        adam = Adam(learning_rate=adam_learning_rate)
        # Use Mean Squared Error as the loss metric.
        # model.compile(optimizer=adam, loss='mse', metrics=['mae', 'mse', 'accuracy', trend_direction_accuracy])
        # Use the custom metric for trend direction accuracy.
        model.compile(optimizer=adam, loss=trend_corr_mae_loss, metrics=['mae', 'mse', 'accuracy', trend_loss, trend_corr_mae_loss])
        self.logger.debug(f'Created new model: {model.summary()}')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_limit, random_state=42)
        # Callback: Define the CSV mylogger
        csv_logger = CSVLogger('PricePred_keras_training_log.csv')
        # Callback: Early Stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001)

        # Add callbacks for model.fit()
        model_fit_callbacks = [early_stopping, csv_logger]
        if self.keras_callbacks is not None:
            for cb in self.keras_callbacks:
                model_fit_callbacks.append(cb)

        # Train the model
        model.fit(x=X_train, y=y_train,
                batch_size=batch_size, epochs=epochs,
                shuffle=shuffle, validation_split=validation_split,
                validation_data=(X_test, y_test),
                callbacks=model_fit_callbacks,
                verbose=self.keras_verbosity)

        loss = model.evaluate(X_test, y_test, verbose=self.keras_verbosity)
        loss_fit = loss[0]  # Best loss metric: mse
        loss_mae = loss[1]  # Mean Absolute Error
        loss_mse = loss[2]  # Mean Squared Error
        loss_trd = loss[4]  # Trend tred_loss
        less_tcm = loss[5]  # Trend Correlation MAE Loss

        # Save the model with the best prediction prediction accuracy
        if self.bayes_best_loss is None or abs(loss_fit) < abs(self.bayes_best_loss):
            best_model = True
            # Save the best loss metric
            self.bayes_best_loss = loss_fit
            # Hold onto the best pred model found thus far.
            self.bayes_best_pred_model = model
            # Saving the best prediction hyperparameters
            self.bayes_best_pred_hp = {'hp_lstm_units': lstm_units,
                                       'hp_lstm_dropout': lstm_dropout,
                                       'hp_adam_learning_rate': adam_learning_rate,
                                       'hp_epochs': epochs,
                                       'hp_batch_size': batch_size,
                                       'hp_hidden_layers': hidden_layers,
                                       'hp_hidden_layer_units': hidden_layer_units,
                                       'fit_loss': loss_fit,
                                       'fit_mae': loss_mae,
                                       'fit_mse': loss_mse,
                                       'fit_trend_loss': loss_trd,
                                       'fit_trend_corr_mae_loss': less_tcm}
            # Hold on the values required for prediction
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
            self.X = X
            self.y = y

        # Loss is flipped because we want to maximize the loss.
        return -loss_fit

    def bayesian_optimization(self, X, y,
                              keras_callbacks=None,
                              # Bayesian Optimization Parameters
                              opt_max_init = 10,
                              opt_max_iter = 20,
                              opt_csv=None,                  # File to save the optimization results
                              # Hyperparameter Ranges
                              pb_lstm_units: tuple = None,            # Default 256
                              pb_lstm_dropout: tuple = None,          # Default 0.5
                              pb_adam_learning_rate: tuple = None,    # Default 0.035
                              pb_epochs: tuple = None,                # Default 300
                              pb_batch_size: tuple = None,            # Default 128
                              pb_hidden_layers: tuple = None,         # Default None or 0
                              pb_hidden_layer_units_1: tuple = None,  # Default None
                              pb_hidden_layer_units_2: tuple = None,  # Default None
                              pb_hidden_layer_units_3: tuple = None,  # Default None
                              pb_hidden_layer_units_4: tuple = None,  # Default None
                              ):
        """
        Perform Bayesian Optimization on the model.
        """

        # =======================================================================
        # === This is the inner function where we try various hyperparameters ===
        def optimize_model(lstm_units: int = 200,
                           lstm_dropout: float = 0.2,
                           adam_learning_rate: float = 0.635,
                           epochs: int = 100,
                           batch_size: int = 32,
                           hidden_layers: int = 0,
                           hidden_layer_units_1: int = None,
                           hidden_layer_units_2: int = None,
                           hidden_layer_units_3: int = None,
                           hidden_layer_units_4: int = None):

            # Force the hidden_layers for testing here...
            # hidden_layers = 4

            # Round up and intify integer parameters.
            # Setup the hidden_layer_units list
            hidden_layer_units = []
            if hidden_layer_units_1 is not None:
                hidden_layer_units = [int(round(hidden_layer_units_1))]
            if hidden_layer_units_2 is not None:
                hidden_layer_units.append(int(round(hidden_layer_units_2)))
            if hidden_layer_units_3 is not None:
                hidden_layer_units.append(int(round(hidden_layer_units_3)))
            if hidden_layer_units_4 is not None:
                hidden_layer_units.append(int(round(hidden_layer_units_4)))
            if hidden_layers is not None and hidden_layer_units is not None:
                # Truncate the hidden_layer_units list to the number of hidden_layers.
                hidden_layer_units = hidden_layer_units[:int(hidden_layers)]

            if self.keras_callbacks is not None and self.tf_profiler:
                tf.profiler.experimental.start(self.tf_logs_dir)

            # Train the mode: loss returns [loss, mae, mse, accuracy, trend_direction_accuracy]
            loss = self.bayes_train_model(X, y,
                                          lstm_units=int(lstm_units),
                                          lstm_dropout=lstm_dropout,
                                          adam_learning_rate=adam_learning_rate,
                                          epochs=int(epochs),
                                          batch_size=int(batch_size),
                                          hidden_layers=int(hidden_layers),
                                          hidden_layer_units=hidden_layer_units)

            # Turn on TensorFlow Profiler
            if self.keras_callbacks is not None and self.tf_profiler:
                tf.summary.trace_on(graph=True, profiler=True)
                writer = tf.summary.create_file_writer(self.tf_logs_dir)
                self.tf_summary_step += 1
                tf.summary.experimental.set_step(self.tf_summary_step)
                with writer.as_default():
                    # Create a summary of the model.
                    tf.summary.trace_export(
                        name="bayes_train_model",
                        step=0,
                        profiler_outdir=self.tf_logs_dir)
                    # Log the hyperparameters
                    tf.summary.scalar('tf_step', tf.summary.experimental.get_step())
                    tf.summary.scalar('lstm_units', lstm_units)
                    tf.summary.scalar('lstm_dropout', lstm_dropout)
                    tf.summary.scalar('adam_learning_rate', adam_learning_rate)
                    tf.summary.scalar('epochs', epochs)
                    tf.summary.scalar('batch_size', batch_size)
                    tf.summary.scalar('hidden_layers', hidden_layers)
                    i = 0
                    for val in hidden_layer_units:
                        tf.summary.scalar(f'hidden_layer_units[{i}]', hidden_layer_units[i])
                        i += 1
                    # Log the trend direction accuracy
                    tf.summary.scalar('loss', loss)

            return loss

        # === End of the function to optimize ===
        # ======================================================

        # Build Huperparameter's dictionary
        pbounds_dict: {tuple} = {}
        if pb_lstm_units is not None:
            pbounds_dict['lstm_units'] = pb_lstm_units
        if pb_lstm_dropout is not None:
            pbounds_dict['lstm_dropout'] = pb_lstm_dropout
        if pb_adam_learning_rate is not None:
            pbounds_dict['adam_learning_rate'] = pb_adam_learning_rate
        if pb_epochs is not None:
            pbounds_dict['epochs'] = pb_epochs
        if pb_batch_size is not None:
            pbounds_dict['batch_size'] = pb_batch_size
        if pb_hidden_layers is not None:
            pbounds_dict['hidden_layers'] = pb_hidden_layers
        if pb_hidden_layer_units_1 is not None:
            pbounds_dict['hidden_layer_units_1'] = pb_hidden_layer_units_1
        if pb_hidden_layer_units_2 is not None:
            pbounds_dict['hidden_layer_units_2'] = pb_hidden_layer_units_2
        if pb_hidden_layer_units_3 is not None:
            pbounds_dict['hidden_layer_units_3'] = pb_hidden_layer_units_3
        if pb_hidden_layer_units_4 is not None:
            pbounds_dict['hidden_layer_units_4'] = pb_hidden_layer_units_4

        # Prepare the Bayesian Optimization
        optimizer = BayesianOptimization(f=optimize_model, pbounds=pbounds_dict)
        # Optimize...
        optimizer.maximize(init_points=opt_max_init, n_iter=opt_max_iter)

        # Save the best hyperparameters to self...
        self.bayes_opt_hypers = optimizer.max

        self.logger.debug(f"Ticker: [{self.ticker}:{self.period}] Bayesian Optimization: {optimizer.max}")

        """
        -------------- Analyze the Prediction Model -------------- 
        """
        # Make this PP objects models the best model found by the optimizer.
        self.model = self.bayes_best_pred_model
        # Run the last best prediction
        self.predict_price(self.X)
        # Restore the scale of the prediction
        # - Save the rescaled prediction to this object (self.pred_xxx)
        self.restore_scale_pred(self.y_pred.reshape(-1, self.targets))
        # Adjust the prediction
        # - The prediction is the prior closing price + the delta of the prediction's
        #   last closing prices.
        self.adjust_prediction()
        # Analyze the prediction
        self.bayes_best_pred_analysis = self.prediction_analysis()

        if opt_csv is not None:
            # Append the results to a CSV file.
            with open(opt_csv, 'a') as f:
                f.write(f"{{ Ticker: [{self.ticker}::{self.period}]: {optimizer.max} }}\n")

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
        max_tries = 10
        while i <= max_tries:
            i += 1
            try:
                # Save the model...
                self.logger.debug(f"Saving model to [{model_path}]")
                model.save(model_path, save_format='keras_v3')
            except Exception as e:
                if i <= max_tries:
                    self.logger.warn(f"Warning: Failed to Save model [{i}] [{model_path}]\n{e}, will retry...")
                    time.sleep(2)
                    continue
                else:
                    self.logger.error(f"Error: Saving model [{model_path}] Tried [{i}] times\n{e}")
                    raise ValueError(f"Error: Saving model [{model_path}] Tried [{i}] times\n{e}")

        self.ticker = ticker
        self.model_path = model_path
        self.last_analysis = datetime.now()

        return model, model_path

    def predict_price(self, X_data: [] = None, model_path: str = None,
                      start_date: str = None, end_date: str = None):
        """
        Predict the next price.
        If  X_data is None, then fetch the required data from Yahoo Finance and pre-process it.
        After predicting the price, adjust the prediction and perform an analysis on the prediction.

        :parameters:
            An array of prior price and indicator data with placeholders for the predicted values.
            This data is passed into
        :param X_data:
        :param model_path:
        :param start_date:
        :param end_date:
        :return:
        """
        self.logger.debug(f"=== Predicting Price for [{self.ticker}:{self.period}]...")

        if self.model is None:
            e_txt = f"Error: Ticker[{self.ticker}] Model is None. Model must be trained before making predictions."
            self.logger.error(e_txt)
            raise ValueError(e_txt)

        if model_path is None and self.model_path is None:
            self.logger.error("Error: The model_path parameter is required.")
            raise ValueError("Error: The model_path parameter is required.")
        else:
            model_path = self.model_path

        if model_path is not None and model_path != '':
            model_path = model_path.replace('=', '~')
            # Extract the ticker from the model path
            ticker, period, dateStart_train, dateEnd_train = model_path.split('_')
            # Python does not like the = sign in filenames.
            # So, we restore the = sign to the ~ ticker symbol.
            ticker = ticker.replace('~', '=')
            if '/' in ticker:
                ticker = ticker.split('/')[-1]

            # Verify that the model ticker matches the object ticker.
            if ticker != self.ticker:
                e_txt = f"Error: Ticker[{self.ticker}] does not match the model ticker[{ticker}] in model file: {model_path}"
                self.logger.error(e_txt)
                raise ValueError(e_txt)

        if start_date is not None:
            # if we have a start_date, load the data from Yahoo Finance.
            # else just use the last prediction dates in self.dateStart_pred and self.dateEnd_pred.
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            self.dateStart_pred = start_date
            self.dateEnd_pred = end_date

            if start_date is None or end_date is None:
                self.logger.error("Error: The date_start and date_end parameters is required.")
                raise ValueError("Error: The date_start and date_end parameters is required.")

        if X_data is None:
            # Load training data and prepare the data
            X, y = self.fetch_n_prep(self.ticker, start_date, end_date,
                                     train_split=0)
            X_data = X
            # Get a list of the actual closing prices for the test period.
            closes = np.array(self.orig_data['Adj Close'])[:len(X) - 1]
            closes = np.append(closes, closes[-1])
            highs = np.array(self.orig_data['High'])[:len(X) - 1]
            highs = np.append(highs, highs[-1])
            lows = np.array(self.orig_data['Low'])[:len(X) - 1]
            lows = np.append(lows, lows[-1])
            self.target_close = closes
            self.target_high = highs
            self.target_low = lows
            # Make Predictions on all the data
            self.train_split = 1.0

        # By this point we should have the X_data to make the prediction.
        # But check to make sure.
        if X_data is None:
            e_txt = f"Error: Ticker[{self.ticker}] X_data is None. X_data must be provided."
            self.logger.error(e_txt)
            raise ValueError(e_txt)

        try:
            # Perform the prediction
            y_pred = self.model.predict(X_data, verbose=self.keras_verbosity)
            self.y_pred = y_pred
        except Exception as e:
            self.logger.error(f"Error: Predicting Price: {e}")
            return None
        else:  # try's else
            # if self.split_limit is None:
            #     self.split_limit = int(len(X_data) * self.train_split)
            self.split_limit = int(len(X_data) * self.train_split)

            # Rescaled the predicted values to dollars...
            data_set_scaled_y = self.data_scaled[-(self.back_candles + self.split_limit):, :].copy()
            # Replace the last columns 4 in data_set_scaled_y with the predicted column values...
            min_len = min(len(y_pred), len(data_set_scaled_y))
            data_set_scaled_y[-min_len:, -self.targets:] = y_pred[-min_len:]

            # Restore the scale of the prediction
            # - Save the rescaled prediction to this object (self.pred_xxx)
            pred_rescaled = self.restore_scale_pred(y_pred.reshape(-1, self.targets))

            self.logger.debug(f"=== Price Prediction Completed [{self.ticker}:{self.period}]...")

            # Perform data alignment/adjustment on the prediction data.
            # - We take the deltas of the prediction and apply them to the prior actual values.
            # - The expectation is that the predictions deltas are the valuable information.
            # - The abs(deltas) are also ranked from 1 to 10 info self.pred_rank.
            # - We should check if the prediction rank can be applied to an HMM model.
            # Doing so makes use the the prediction deltas rather than the actual values.
            # Detail re self.adj_xxx are stored to this object in the adjust_prediction method.
            self.adjust_prediction()

            if start_date is not None:
                # From the prediction call or current date_start
                self.dateStart_pred = start_date
            elif self.date_data is not None:
                self.dateStart_pred = self.date_data.iloc[0].strftime('%Y-%m-%d')

            if end_date is not None:
                # From the prediction call or current date_end
                self.dateEnd_pred = end_date
            elif self.date_data is not None:
                self.dateEnd_pred = self.date_data.iloc[-1].strftime('%Y-%m-%d')

            pred_dates = self.date_data
            pred_dates = pd.concat([pred_dates, pd.Series(self.date_data.iloc[-1] + self.next_timedelta())])
            self.pred_dates = pred_dates

            # Perform an analysis of the prediction
            self.prediction_analysis()

        return self.y_pred

    def adjust_prediction(self):
        """
        The adjusted prediction leverages the deltas between the predicted values
        and pins the delta to the prior actual close, high, and low values, rather
        than pining the prediction to the prior predicted value.
        This results in predictions that do not wander from the actual price action.

        Updates self.adj_pred with the adjusted predictions.

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
        # -- Adjust Predicted Close: Applies deltas of the predicted close to the prior actual close.
        pred_delta_c = [pred_close[i - 1] - pred_close[i] for i in range(1, len(pred_close))]
        min_len = min(len(target_close), len(pred_close))
        target_close = target_close[-min_len:]
        pred_adj_close = [target_close[i] + pred_delta_c[i] for i in range(0, len(pred_delta_c))]

        # -- Adjust Predicted High: Applies deltas of the predicted high to the prior actual high.
        pred_delta_h = [pred_high[i - 1] - pred_high[i] for i in range(1, len(pred_high))]
        min_len = min(len(target_high), len(pred_high))
        target_high = target_high[-min_len:]
        pred_adj_high = [target_high[i] + abs(pred_delta_h[i]) for i in range(0, len(pred_delta_h))]


        #    -- Adjusted Close Prediction should not be higher than Adjusted High Prediction
        pred_adj_close = [pred_adj_close[i] if pred_adj_close[i] < pred_adj_high[i] else pred_adj_high[i]
                          for i in range(0, len(pred_adj_close))]

        # -- Adjust Predicted low: Applies deltas of the predicted low to the prior actual low.
        pred_delta_l = [pred_low[i - 1] - pred_low[i] for i in range(1, len(pred_low))]
        min_len = min(len(target_low), len(pred_low))
        target_low = target_low[-min_len:]
        pred_adj_low = [target_low[i] - abs(pred_delta_l[i]) for i in range(0, len(pred_delta_l))]

        #    -- Adjusted Close Prediction should not be lower than Adjusted Low Prediction
        pred_adj_close = [pred_adj_close[i] if pred_adj_close[i] > pred_adj_low[i] else pred_adj_low[i]
            for i in range(0, len(pred_adj_close))]

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
        # self.pred_rank = (pred_sign * pred_rank) * -1
        # Stash all prediction rankings.
        self.pred_rank = ranking

        self.target_close = target_close
        self.target_high = target_high
        self.target_low = target_low

        self.adj_pred = adj_pred  # The adjusted predictions
        self.adj_pred_class = pred_class  # Does not get adjusted
        self.adj_pred_close = pred_adj_close  # Adjusted close
        self.adj_pred_high = pred_adj_high  # Adjusted high
        self.adj_pred_low = pred_adj_low  # Adjusted low

        return pred_adj_close, pred_adj_high, pred_adj_low

    def mcmc_process_data(self, data, plot_trace=False):
        """
        MCMC:  Markov Chain Monte Carlo
        This function preprocesses the data and defines the Bayesian AR(1) model.
        """
        # Step 2: Preprocess Data
        data['LogReturn'] = np.log(data['Close'] / data['Close'].shift(1))
        data.dropna(inplace=True)

        # Step 3: Define Bayesian AR(1) Model
        # AR(1) Autoregressive Model of order 1
        with pm.Model() as model:
            # Priors for the parameters
            mu = pm.Normal('mu', mu=0, sigma=0.1)
            phi = pm.Normal('phi', mu=0, sigma=1)
            sigma = pm.HalfNormal('sigma', sigma=0.1)

            # Observed data
            start_obs = -7
            Y_t = data['LogReturn'].values[:start_obs]  # Current returns
            Y_tm1 = data['LogReturn'].values[:-1]  # Previous returns

            # Expected value of the current return
            mu_Y_t = mu + phi * (Y_tm1 - mu)

            # Likelihood: Model the current return given the previous return
            Y_obs = pm.Normal('Y_obs', mu=mu_Y_t, sigma=sigma, observed=Y_t)

        # Step 4: Sample from Posterior
        with model:
            trace = pm.sample(2000, tune=1000, target_accept=0.95, return_inferencedata=True)

        if plot_trace:
            # Check Trace Plots
            az.plot_trace(trace)
            plt.show()

        return data, Y_t, Y_tm1, mu_Y_t, trace

    def mcmc_forecast_returns(self, trace, last_return, steps=30):
        """
        MCMC:  Markov Chain Monte Carlo
        This function forecasts future returns and prices using the Bayesian AR(1) model.
        AR(1): Autoregressive Model of order 1
        """
        # Step 5: Forecast Future Returns and Prices
        mu_samples = trace.posterior['mu'].values.flatten()
        phi_samples = trace.posterior['phi'].values.flatten()
        sigma_samples = trace.posterior['sigma'].values.flatten()

        n_samples = len(mu_samples)
        forecasts = np.zeros((n_samples, steps))

        for i in range(n_samples):
            mu_i = mu_samples[i]
            phi_i = phi_samples[i]
            sigma_i = sigma_samples[i]
            ret = last_return
            for t in range(steps):
                eps = np.random.normal(0, sigma_i)
                ret = mu_i + phi_i * (ret - mu_i) + eps
                forecasts[i, t] = ret
        return forecasts

    def mcmc_forecast_prices(self, show_plot=False):
        """
        MCMC:  Markov Chain Monte Carlo
        AR(1): Autoregressive Model of order 1
        This function forecasts future stock prices using the Bayesian AR(1) model.
        """
        data, features = self.fetch_data_yahoo()
        data, Y_t, Y_tm1, mu_Y_t, trace = self.mcmc_process_data(data)
        last_return = data['LogReturn'].values[-1]
        forecast_steps = 30
        forecasts = self.mcmc_forecast_returns(trace, last_return, steps=forecast_steps)

        # Convert to Prices
        last_price = data['Close'].values[-1]
        price_forecasts = last_price * np.exp(np.cumsum(forecasts, axis=1))
        median_forecast = np.median(price_forecasts, axis=0)
        # Get the 94% Highest Density Interval, given the probability of hdi_prob.
        # hpd_interval = az.hdi(price_forecasts, hdi_prob=0.94)
        hpd_interval = az.hdi(price_forecasts, hdi_prob=0.30)

        # Step 6: Visualize Forecast
        forecast_dates = pd.date_range(start=data.index[-1], periods=forecast_steps + 1, freq='B')[1:]
        if show_plot:
            plt.figure(figsize=(12, 6))
            plt.plot(data.index[-100:], data['Close'].values[-100:], label='Historical Prices')
            plt.plot(forecast_dates, median_forecast, label='Median Forecast')
            plt.fill_between(forecast_dates, hpd_interval[:, 0], hpd_interval[:, 1],
                             color='gray', alpha=0.5, label='94% Credible Interval')
            plt.title('Forecasted Stock Prices for {}'.format(symbol))
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.show()

        # Return data that will be used to generate the forecast chart
        return forecast_dates, median_forecast, hpd_interval

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

        split_limit = int(len(self.date_data) * self.train_split)
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
        if self.pred is None:
            self.logger.error("Error: No prediction data to save.")
            raise ValueError("Error: No prediction data to save.")

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

    def fetch_n_prep(self, ticker: str, date_start: str, date_end: str,
                     period: str = None, train_split: float = 0.05,
                     backcandles: bool = None):

        # Handle the optional parameters
        if period is None:
            period = self.period
        else:
            if period not in PricePredict.PeriodValues:
                self.logger.error(f"period[{period}]: may only be \"{'", "'.join(PricePredict.PeriodValues)}\"")
                raise ValueError(f"period[{period}]: may only be \"{'", "'.join(PricePredict.PeriodValues)}\"")
            self.period = period
        if train_split is None:
            train_split = self.train_split
        else:
            self.train_split = train_split
        if backcandles is None:
            backcandles = self.back_candles
        else:
            self.back_candles = backcandles

        # Load data from Yahoo Finance
        orig_data, features = self.fetch_data_yahoo(ticker=ticker, period=period, date_start=date_start, date_end=date_end)
        # Augment the data with technical indicators/features and targets
        aug_data, features, targets, dates_data = self.augment_data(orig_data, features)
        # Scale the data
        scaled_data, scaler = self.scale_data(aug_data)
        # Prepare the scaled data for model inputs
        X, y = self.prep_model_inputs(scaled_data, features)

        # Training split the X & y data into training and testing data
        splitlimit = int(len(X) * train_split)
        self.split_limit = splitlimit

        # mylogger.info("lenX:",len(X), "splitLimit:",splitlimit)
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

    def fetch_train(self, ticker,
                     train_date_start, train_date_end,
                     force_training=False,
                     period=None, train_split=None, backcandles=None,
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
        :param train_split=None:  # The percentage of the data to use for training
        :param backcandles=None: # The number of candles to look back for each period
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
        if train_split is None:
            train_split = self.train_split
        else:
            self.train_split = train_split
        if backcandles is None:
            backcandles = self.back_candles
        else:
            self.back_candles = backcandles

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
            X, y = self.fetch_n_prep(ticker, train_date_start, train_date_end,
                                     period=period, train_split=0)

            # ============== Train the model
            # Use a small batch size and epochs to test the model training
            # Training split the X & y data into training and testing data
            # What is returned is the model, the prediction, and the mean squared error.
            model, y_pred, mse = self.train_model(X, y,
                                                  train_split=train_split, backcandles=backcandles)

            if len(y_pred) != 0:
                pcnt_nan = (len(y_pred) - np.count_nonzero(~np.isnan(y_pred))) / len(y_pred)
                if pcnt_nan > 0.1:
                    self.logger.info(f"\n*** NaNs in y_pred: {pcnt_nan}%")
                    # Throw a data exception if the model is not trained properly.
                    raise ValueError("Error: Prediction has too many NaNs. Check for Nans in the data?")

                if model is not None:
                    self.model = model

        return model

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

        self.predict_price(start_date=date_start, end_date=date_end)

        return self.adj_pred, self.pred_dates

    def fetch_train_and_predict(self, ticker,
                                train_date_start, train_date_end,
                                pred_date_start, pred_date_end,
                                force_training=False,
                                period=None, train_split=None, backcandles=None,
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
        :param train_split=None:  # The percentage of the data to use for training
        :param backcandles=None: # The number of candles to look back for each period
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
        if train_split is None:
            train_split = self.train_split
        else:
            self.train_split = train_split
        if backcandles is None:
            backcandles = self.back_candles
        else:
            self.back_candles = backcandles

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
            X, y = self.fetch_n_prep(ticker, train_date_start, train_date_end,
                                     period=period, train_split=0)

            # ============== Train the model
            # Use a small batch size and epochs to test the model training
            # Training split the X & y data into training and testing data
            # What is returned is the model, the prediction, and the mean squared error.
            model, y_pred, mse = self.train_model(X, y,
                                                  train_split=train_split, backcandles=backcandles)

            if len(y_pred) != 0:
                pcnt_nan = (len(y_pred) - np.count_nonzero(~np.isnan(y_pred))) / len(y_pred)
                if pcnt_nan > 0.1:
                    self.logger.info(f"\n*** NaNs in y_pred: {pcnt_nan}%")
                    # Throw a data exception if the model is not trained properly.
                    raise ValueError("Error: Prediction has too many NaNs. Check for Nans in the data?")

            self.predict_price(start_date=pred_date_start, end_date=pred_date_end)

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
        # mylogger.info(f"elements:{elements}")
        tot_deltas = 0
        tot_tradrng = 0
        if self.verbose:
            # Output an Analysis of the prediction
            for i in range(-1, -elements, -1):
                actual = self.orig_data['Adj Close'].iloc[i - 1]
                predval = self.adj_pred[i - 1][1]
                pred_delta = abs(predval - actual)
                tot_deltas += pred_delta
                trd_rng = abs(self.orig_data['High'].iloc[i] - self.orig_data['Low'].iloc[i])
                tot_tradrng += trd_rng
                self.logger.debug(f"{i}: Close: {actual.round(2)}  Predicted: ${predval.round(2)}  Actual: ${actual.round(2)}" +
                                 f"  Delta: ${pred_delta.round(6)}  Trade Rng: ${trd_rng.round(2)}")

        # Get up days vs down days
        self_data = self.orig_data
        self_adj = self.adj_pred
        self_prd = self.pred
        close_col = 'Adj Close'
        close_idx = 0
        open_idx = 0
        self_close_trends = [1 if self_data[close_col].iloc[i] > self_data['Open'].iloc[i] else -1 for i in range(len(self_data))]
        self_prd_trends = [1 if self_prd[i - 1][close_idx] > self_prd[i][open_idx] else -1 for i in range(1, len(self_prd))]
        self_adj_trends = [1 if self_adj[i - 1][close_idx] > self_adj[i][open_idx] else -1 for i in range(1, len(self_adj))]
        min_len = min(len(self_close_trends), len(self_adj_trends))
        self_close_trends = self_close_trends[-min_len:]
        self_adj_trends = self_adj_trends[-min_len:]

        # Get the min length betweem se;f_close_trends self_adj_trends and self_pred_trends
        min_len = min(len(self_close_trends), len(self_adj_trends), len(self_prd_trends))
        # Truncate the lists to the min length
        self_close_trends = self_close_trends[-min_len:]
        self.trends_close = self_close_trends
        self_adj_trends = self_adj_trends[-min_len:]
        self.trends_adj = self_adj_trends
        self_prd_trends = self_prd_trends[-min_len:]

        # Calculate the corr for adjusted predictions
        corr_adj_list = [self_close_trends[i] + self_adj_trends[i] for i in range(len(self_close_trends))]
        total_days = len(corr_adj_list)
        corr_adj_days = corr_adj_list.count(2) + corr_adj_list.count(-2)
        uncorr_adj_days = corr_adj_list.count(0)
        pct_adj_corr = round(corr_adj_days / total_days * 100, 4)
        pct_adj_uncorr = round(uncorr_adj_days / total_days * 100, 4)

        # Calculate the corr for original predictions
        corr_prd_list = [self_close_trends[i] + self_prd_trends[i] for i in range(len(self_close_trends))]
        self.trends_corr = corr_prd_list
        total_days = len(corr_prd_list)
        corr_prd_days = corr_prd_list.count(2) + corr_prd_list.count(-2)
        uncorr_prd_days = corr_prd_list.count(0)
        pct_prd_corr = round(corr_prd_days / total_days * 100, 4)
        pct_prd_uncorr = round(uncorr_prd_days / total_days * 100, 4)

            # Calculate a df pf trends of 'Adj Close' as 1-up 0-flat -1-down

        self.logger.debug("============================================================================")
        self.logger.debug(f"Mean Trading Range: ${round(tot_tradrng / elements, 2)}")
        self.logger.debug(f"Mean Delta (Actual vs Prediction): ${round((tot_deltas / elements), 2)}")
        self.logger.debug("============================================================================")

        analysis = dict()
        analysis['actual_vs_pred'] = {'mean_trading_range': round(tot_tradrng / elements, 2),
                                      'mean_delta': round((tot_deltas / elements), 2),
                                      'corr_day_cnt': corr_prd_days,
                                      'corr_days': corr_prd_days,
                                      'uncorr_days': uncorr_prd_days,
                                      'pct_corr': round(pct_prd_corr, 4),
                                      'pct_uncorr': round(pct_prd_uncorr, 4)}

        elements = len(self.adj_pred_close)
        # mylogger.info(f"elements:{elements}")
        tot_deltas = 0
        tot_tradrng = 0
        for i in range(-1, -elements, -1):
            actual = self.orig_data['Adj Close'].iloc[i - 1]
            predval = self.adj_pred_close[i - 1]
            pred_delta = abs(predval - actual)
            tot_deltas += pred_delta
            trd_rng = abs(self.orig_data['High'].iloc[i] - self.orig_data['Low'].iloc[i])
            tot_tradrng += trd_rng
            self.logger.debug(
                f"{i}: Close {actual.round(2)}  Predicted: ${predval.round(2)}  Actual: ${actual.round(2)}" +
                f"  Delta:${pred_delta.round(6)}  Trade Rng: ${trd_rng.round(2)}")

        self.logger.debug("============================================================================")
        self.logger.debug(f"Mean Trading Range: ${round(tot_tradrng / elements, 2)}")
        self.logger.debug(f"Mean Delta (Actual vs Prediction): ${round((tot_deltas / elements), 2)}")
        self.logger.debug("============================================================================")

        analysis['actual_vs_adj_pred'] = {'mean_trading_range': round(tot_tradrng / elements, 2),
                                          'mean_delta': round((tot_deltas / elements), 2),
                                          'corr_days': corr_adj_days,
                                          'uncorr_days': uncorr_adj_days,
                                          'pct_corr': round(pct_adj_corr, 4),
                                          'pct_uncorr': round(pct_adj_uncorr , 4)}

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
                self.logger.error(f"Failed to collect trend data for self:[{self.ticker}:{self.period}], for seasonality corr.")
            # Make sure that the lengths of self_trends and sd_trends are the same
            min_len = min(len(self_trends), len(sd_trends))
            self_trends = self_trends[-min_len:]
            sd_trends = sd_trends[-min_len:]

            # Calculate the corr
            corr_adj_list = [self_trends[i] + sd_trends[i] for i in range(len(self_trends))]
            total_days = len(corr_adj_list)
            correlated_days = corr_adj_list.count(2) + corr_adj_list.count(-2)
            uncorrelated_days = corr_adj_list.count(0)
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
                'pct_corr': f'{round(self.season_corr * 100, 4)}',
                'pct_uncorr': f'{round(pct_uncorr * 100, 4)}'}

            analysis['pred_rankings'] = {
                'season_last_delta': f'{self.season_last_delta}',
                'season_rank': f'{self.season_rank}',
                'pred_last_delta': f'{round(self.pred_last_delta, 4)}',
                'pred_rank': f'{self.pred_rank}'}

            self.pred_strength = np.round((self.pred_rank + (self.season_rank * self.season_corr)) / 20, 4)
            analysis['pred_strength'] = {
                'strength': f'{self.pred_strength}'}
            if str(type(self.pred_strength)) == "<class 'numpy.ndarray'>":
                self.logger.debug(f"pred_strength is a numpy array: {self.pred_strength[:3]}")
                self.pred_strength = self.pred_strength[0]

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
        plt.figure(figsize=(16, 8), facecolor='grey')
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

    def periodic_correlation(self, ppo,
                             start_date: str = None,
                             period_len: int = None,
                             min_data_points: int = None):
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
        :param period_len:
            Length of the period to calculate the correlation ending with the last date.
        :param start_date:
            If set grab data from start_date to the end of the data or for period_len days.
        :param min_data_points:
            The minimum number of data points to calculate the correlation.
        :return:
            A dictionary containing the correlation data.
        """

        # PPO must be a PricePredict object
        if type(ppo).__name__ != type(self).__name__:
            e_txt = (f"The ppo parameter must be a {type(self).__name__} object." +
                     f" Incoming ppo type: {type(ppo).__name__}")
            self.logger.error(e_txt)
            raise ValueError(e_txt)
        if ppo.period != self.period:
            e_txt = f"PPOs [{self.ticker}:{self.period}] [{ppo.ticker}:{ppo.period}] must have the same period."
            self.logger.error(e_txt)
            raise ValueError(e_txt)

        if start_date is not None:
            # Initially assume that we will use all the data
            # - Convert start_date to a datetime object
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_date = None
            if period_len is None:
                end_date = datetime.now().strftime("%Y-%m-%d").strftime("%Y-%m-%d")
            else:
                # Calculate the end date based on the period_len
                if self.period == PricePredict.PeriodDaily:
                    end_date = (start_dt + timedelta(days=period_len)).strftime("%Y-%m-%d")
                elif self.period == PricePredict.PeriodWeekly:
                    end_date = (start_dt + timedelta(weeks=period_len)).strftime("%Y-%m-%d")
            # Load the data from Yahoo Finance
            self_data, feature_cnt = self.fetch_data_yahoo(ticker=self.ticker, date_start=start_date, date_end=end_date)
            ppo_data, feature_cnt = ppo.fetch_data_yahoo(ticker=ppo.ticker, date_start=start_date, date_end=end_date)
            if self_data is None or ppo_data is None:
                e_txt = f"Error: Could not load the data for correlation analysis."
                self.logger.error(e_txt)
                raise ValueError(e_txt)
        else:
            self_data = self.orig_data
            ppo_data = ppo.orig_data

        if period_len is not None and min_data_points is not None and period_len < min_data_points:
            min_data_points = period_len

        if start_date is None:
            # Get the require dataset for self and the incoming ppo.
            needed_end_dt = datetime.now().strftime("%Y-%m-%d")
            needed_start_dt = (datetime.now() - timedelta(days=300)).strftime("%Y-%m-%d")
            self_start_dt = self.date_start
            self_end_dt = self.date_end
            ppo_start_dt = ppo.date_start
            ppo_end_dt = ppo.date_end
            if self_data is None or needed_start_dt != self_start_dt or needed_end_dt != self_end_dt:
                self_data, feature_cnt = self.fetch_data_yahoo(ticker=self.ticker, date_start=needed_start_dt, date_end=needed_end_dt)
                self_data = self.orig_data
            if ppo_data is None or needed_start_dt != ppo_start_dt or needed_end_dt != ppo_end_dt:
                ppo_data, feature_cnt = ppo.fetch_data_yahoo(ticker=ppo.ticker, date_start=needed_start_dt, date_end=needed_end_dt)
                ppo_data = ppo.orig_data
            self_start_dt = self.date_start
            self_end_dt = self.date_end
            ppo_start_dt = ppo.date_start
            ppo_end_dt = ppo.date_end

        # Use as much data as possible for correlation analysis
        # Get the end date for each _date set
        self_end_date = self_data.index[-1]
        ppo_end_date = ppo_data.index[-1]
        best_end_date = min(self_end_date, ppo_end_date)
        # Truncate the data to the best end date
        self_data = self_data[:best_end_date]
        ppo_data = ppo_data[:best_end_date]
        # Get the best start_date
        self_start_date = self_data.index[0]
        ppo_start_date = ppo_data.index[0]
        best_start_date = max(self_start_date, ppo_start_date)
        # Truncate the start of the data to the best end date
        self_data = self_data[best_start_date:]
        ppo_data = ppo_data[best_start_date:]
        # Trunkate the data to period_len if it is not None
        if period_len is not None:
            self_data = self_data[-period_len:]
            ppo_data = ppo_data[-period_len:]

        # Once again, make sure that we have enough data points
        if min_data_points is not None and (self_data) < min_data_points:
            e_txt = f"self[{self.ticker}:{self.period}] has less than {min_data_points} data points [{len(self_data)}]."
            raise ValueError(e_txt)
        if min_data_points is not None and len(ppo_data) < min_data_points:
            e_txt = f"ppo[{ppo.ticker}:{ppo.period}] has less than {min_data_points} data points [{len(ppo_data)}]."
            raise ValueError(e_txt)
        # Check that both data sets have the same length
        if len(self_data) != len(ppo_data):
            e_txt = f"Data lengths do not match [{self.ticker}:{self.period}] [{ppo.ticker}:{ppo.period}]: self[{len(self_data)}] != ppo[{len(ppo_data)}]"
            raise ValueError(e_txt)
        # Check that both datasets begin on the same day
        if self_data.index[0] != ppo_data.index[0]:
            e_txt = f"Data start dates do not match [{self.ticker}:{self.period}] [{ppo.ticker}:{ppo.period}]: self[{self_data.index[0]}] != ppo[{ppo_data.index[0]}]"
            raise ValueError(e_txt)
        # Check that both datasets end on the same day
        if self_data.index[-1] != ppo_data.index[-1]:
            e_txt = f"Data end dates do not match [{self.ticker}:{self.period}] [{ppo.ticker}:{ppo.period}]: self[{self_data.index[-1]}] != ppo[{ppo_data.index[-1]}]"
            raise ValueError(e_txt)

        # Save the start and end dates and period length of the corr
        corr_start_date = self_data.index[0].strftime("%Y-%m-%d %H:%M:%S")
        corr_end_date = self_data.index[-1].strftime("%Y-%m-%d %H:%M:%S")
        corr_period_len = len(self_data)

        # Get up days vs down days
        close_col = 'Adj Close'
        self_closes = self_data[close_col]
        self_closes = self_closes.bfill().ffill()
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
            # Concatenate self_trends with ppo_trends into one dataframe whose columns are stock_a and stock_b
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
        # self.mylogger.info(f"Days: {total_days} Correlated Days: {correlated_days}  Uncorrelated Days: {uncorrelated_days}")
        # self.mylogger.info(f"Correlated Days: {pct_corr}%  Uncorrelated Days: {pct_uncorr}%")
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
            self.sentiment_text = f'Indexes [{self.ticker}:{self.period}] do not support sentiment analysis.'
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
                         give me a review of the company from a financial perspective and be critical of values from 
                         period to period and consider if missing values indicate mis-reporting of data. And, add a
                         summary of sentiment analysis of the company from the viewpoint of board members, from the
                         viewpoint of shareholders, and from the viewpoint of short sellers. Finally, create a
                         sentiment analysis score for the company from 1 to 5, where 1 is very negative and 5 is
                         very positive. Separate each section into its json attribute.
                         Please, place the correctly formatted JSON output between the "<JsonOutputFormat>" and
                         "</JsonOutputFormat>" tags, at the start of the response.
                         Please, place the sentiment text output between the "<sentimentTextOutput>" and 
                         "</sentimentTextOutput>" tags, at a the end of the response.
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
            self.logger.error(f"Error: Could call on Groq for sentiment on [{self.ticker}:{self.period}].\n{e}")
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
                self.logger.warn(f"Failed to parse JSON response from Groq for sentiment on [{self.ticker}:{self.period}].\n{e}")
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
            self.fetch_data_yahoo(ticker=self.ticker, period=self.period, date_start=self.date_start, date_end=self.date_end)

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
                f"Error: Could not perform seasonal decomposition for [{self.ticker}:{self.period}]\n{e}")
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
    def unserialize(obj, ignore: bool = False):

        if ignore is None and ignore is False:
            # But default use the current PricePredict Module
            # vs the one packaged in the dill object.
            ignore = False
            PricePredict.__module__ = '__main__'

        # Decompress the object
        return dill.loads(lzma.decompress(obj), ignore=ignore)

    def serialize_me(self):
        # Compress the object
        return PricePredict.serialize(self)

    def store_me(self) -> str:
        # Clear properties that may have Tensorflow objects
        # as they won't pickle correctly.
        self.keras_callbacks = None
        self.tf_profiler = None
        # Store this object into a compressed dill object *.dilz
        filepath = self.ppo_dir + f"{self.ticker}_{self.period}_{self.date_end}.dilz"
        self.ppo_file = filepath
        with open(filepath, 'wb') as f:
            f.write(self.serialize_me())
        return filepath

    @staticmethod
    def restore(filepath, ignore: bool = None):
        # Restore this object from a compressed dill object *.dilz
        with open(filepath, 'rb') as f:
            obj = f.read()
        return PricePredict.unserialize(obj, ignore=ignore)