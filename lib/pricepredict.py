
import asyncio
import datetime
import time
import ipywidgets as widgets
import keras
import math
import matplotlib.pyplot as plt
import mplfinance as mfp
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
import logging
import statsmodels.api as sm
import json
import jsonify
import pydantic

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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from IPython.display import Image
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union

from sympy.physics.quantum.shor import period_find

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
Notes: Regarding training...
Training the model should requires at least 2 years of prior data for a daily period. 
For other periodicies, that's from 600 to 800 prior periods.

Notes: Regarding prediction...
To perform a prediction, the model requires enough data to fill the back_candles period,
and enough data to fill have data for any added technical indicators.
"""


# -------------------------------------------------
# Define the Cache class using Pydantic's BaseModel
# -------------------------------------------------
class DataCache(BaseModel):
    items: Dict[str, Union[str, int, float, List[Any], Dict[str, Any]]] = Field(default_factory=dict)

    # Method to set a cache item
    def set_item(self, key: str, value: Union[str, int, float, List[Any], Dict[str, Any]]) -> None:
        self.items[key] = value

    # Method to get a cache item
    def get_item(self, key: str) -> Optional[Union[str, int, float, List[Any], Dict[str, Any]]]:
        return self.items.get(key)

    # Method to invalidate a cache item
    def invalidate_item(self, key: str) -> None:
        if key in self.items:
            del self.items[key]

    # Method to clear all cache items
    def clear_cache(self) -> None:
        self.items.clear()


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
    PeriodValues = [PeriodDaily, PeriodWeekly]

    def __init__(self,
                 ticker='',                 # The ticker for the data
                 model_dir='./models/',       # The directory where the model is saved
                 chart_dir='./charts/',       # The directory where the charts are saved
                 preds_dir='./predictions/',  # The directory where the predictions are saved
                 period=PeriodDaily,        # The period for the data (D, W)
                 back_candles=15,           # The number of candles to look back for each period.
                 split_pcnt=0.8,            # The value for splitting data into training and testing.
                 batch_size=30,             # The batch size for training the model.
                 epochs=50,                 # The number of epochs for training the model.
                 shuffle=True,              # Shuffle the data for training the model.
                 val_split=0.1,             # The validation split used during training.
                 verbose=True,              # Print debug information
                 logger = None,              # The logger for this object
                 logger_file_path = None,   # The path to the log file
                 log_level = None,           # The logging level
                 force_training = False,    # Force training the model
                 keras_log = 'PricePredict_keras.log'  # The keras log file
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
        self.model_dir = model_dir        # The directory where the model is saved
        self.model_path = ''              # The path to the current loaded model
        self.preds_dir = preds_dir        # The directory where the predictions are saved
        self.preds_path = ''              # The path to the current predictions
        self.chart_dir = chart_dir        # The directory where the charts are saved
        self.chart_path = ''              # The path to the current chart
        self.seasonal_chart_path = ''     # The path to the seasonal decomposition chart
        self.period = period              # The period for the data (D, W)
        self.model = None                 # The current loaded model
        self.scaler = None                # The current models scaler
        self.Verbose = verbose            # Print debug information
        self.ticker = ticker              # The ticker for the data
        self.ticker_data = None           # The data for the ticker (Long Name, last price, etc.)
        self.date_start = ''              # The start date for the data
        self.date_end = ''                # The end date for the data
        self.orig_data = None             # Save the originally downloaded data for later use.
        self.unagg_data = None            # Save the unaggregated data for later use.
        self.date_data = None             # Save the date data for later use.
        self.aug_data = None              # Save the augmented data for later use.
        self.features = None              # The number of features in the data
        self.targets = None               # The number of targets (predictions) in the data
        self.data_scaled = None           # Save the scaled data for later use.
        self.force_training = force_training  # Force training the model
        self.X = None                     # 3D array of training data.
        self.y = None                     # Target values (Adj Close)
        self.X_train = None               # Training data
        self.X_test = None                # Test data
        self.y_test = None                # Test data
        self.y_test_closes = None         # Test data closes
        self.y_train = None               # Training data
        self.y_pred = None                # The prediction
        self.y_pred_rescaled = None       # The rescaled prediction
        self.mean_squared_error = None    # The mean squared error for the model
        self.target_close = None          # The target close
        self.target_high = None           # The target high
        self.target_low = None            # The target low
        self.pred = None                  # The predictions (4 columns)
        self.pred_rescaled = None         # The  predictions rescaled (4 columns)
        self.pred_class = None            # The adjusted prediction class
        self.pred_close = None            # The adjusted prediction close
        self.pred_high = None             # The adjusted prediction close
        self.pred_low = None              # The adjusted prediction close
        self.adj_pred = None              # The adjusted predictions (3 columns)
        self.adj_pred_class = None        # The adjusted prediction class
        self.adj_pred_close = None        # The adjusted prediction close
        self.adj_pred_high = None         # The adjusted prediction high
        self.adj_pred_low = None          # The adjusted prediction low
        self.dateStart_train = None       # The start date for the training period
        self.analysis = None              # The analysis of the model
        self.analysis_path = None         # The path to the analysis file
        self.dateEnd_train = None         # The end date for the training period
        self.dateStart_pred = None        # The start date for the prediction period
        self.dateEnd_pred = None          # The end date for the prediction period
        self.back_candles = back_candles  # The number of candles to look back for each period.
        self.split_pcnt = split_pcnt      # The value for splitting data into training and testing.
        self.split_limit = None           # The split limit for training and testing data.
        self.batch_size = batch_size      # The batch size for training the model.
        self.epochs = epochs              # The number of epochs for training the model.
        self.shuffle = shuffle            # Shuffle the data for training the model.
        self.val_split = val_split        # The validation split used during training.
        self.seasonal_dec = None          # The seasonal decomposition
        self.keras_log = keras_log        # The keras log file
        self.keras_verbosity = 0          # The keras verbosity level
        self.cached_train_data = None     # Cached training data
        self.cached_pred_data = None      # Cached prediction data
        # Analitics...
        self.last_analysis = None         # The last analysis
        self.preds_path = None            # The path to the predictions file
        self.pred_last_delta = None       # The last delta in the prediction
        self.pred_rank = None             # The rank of the prediction
        self.season_last_delta = None     # The last delta in the seasonal decomposition
        self.season_rank = None           # The rank of the seasonal decomposition
        self.season_corr = None           # The correlation of the seasonal decomposition
        self.pred_strength = None         # The strength of the prediction
        self.top10corr = None             # The top 10 correlations dict {'<Sym>': <Corr%>}
        self.top10xcorr = None            # The top 10 cross correlations dict {'<Sym>': <xCorr%>}

        # Create a logger for this object.
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        # # Set the objects logging to stdout by default for testing.
        # # The logger and the logging-level can be overridden by the calling program
        # # once the object is instantiated.
        # if self.logger.hasHandlers():
        #     self.logger.handlers.clear()
        # if tf.get_logger().hasHandlers():
        #     tf.get_logger().handlers.clear()

        # # Turn off this objects logging to stdout
        # self.logger.propagate = True
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
        # Check yahoo if symbol is valid...
        try:
            ticker_ = yf.Ticker(chk_ticker).history(period='5d', interval='1d')
            if len(ticker_) > 0:
                self.ticker = ticker
                ticker_data = yf.Ticker(chk_ticker).info
                self.ticker_data = ticker_data
        except:
            ticker_ = []
        if len(ticker_) == 0:
            ticker = None


        return ticker_data

    def fetch_data_yahoo(self, ticker, date_start, date_end, period=None):
        """
        Load data from Yahoo Finance.
        :param ticker:
        :param date_start:
        :param date_end:
        - Optional
        :param period:      # The period for the data (D, W)
        :return:
            data,           # An array of data Open, High, Low, Close, Adj Close, Volume
            feature_cnt:    # The number of features in the data
        """

        if period is None:
            period = self.period
        else:
            if period not in PricePredict.PeriodValues:
                self.logger.error(f"*** Exception: [{ticker}]: period[{period}] must be \"{'"| "'.join(PricePredict.PeriodValues)}\"")
                raise ValueError(f"period[{period}]: man only be \"{'", "'.join(PricePredict.PeriodValues)}\"")
            self.period = period

        if self.Verbose:
            self.logger.info(f"Loading Data for: {ticker}  from: {date_start} to {date_end}")

        # Remove "Test-" from the start of the ticker (used for model testing)
        f_ticker = re.sub(r'^Test-', '', ticker)
        try:
            data = yf.download(tickers=f_ticker, start=date_start, end=date_end)
        except Exception as e:
            self.logger.error(f"Error: Could not download data for: {ticker}")
            self.logger.error(f"Error: {e}")
            return None, None

        if len(data) == 0:
            self.logger.error(f"Error: No data for {ticker} from {date_start} to {date_end}")
            raise ValueError(f"Error: No data for {ticker} from {date_start} to {date_end}")

        # Aggregate the data to a weekly period, if nd
        unagg_data = None
        orig_data = data.copy(deep=True)
        if self.period == self.PeriodWeekly:
            wkl_data = self.aggregate_data(data, period=self.PeriodWeekly)
            unagg_data = orig_data.copy(deep=True)
            orig_data = wkl_data.copy(deep=True)
            data = wkl_data.copy(deep=True)

        offset = pd.Timedelta(days=-30)
        # Resample to 'W'eekly or 'ME'(Month End)
        # logic = {'Open'  : 'first',
        #          'High'  : 'max',
        #          'Low'   : 'min',
        #          'Close' : 'last',
        #          'Adj Close': 'last',
        #          'Volume': 'sum'}
        # data = data.resample('W', offset=offset).apply(logic)
        if self.Verbose:
            self.logger.info(f"data.len(): {len(data)}  data.shape: {data.shape}")

        feature_cnt_dl = data.shape[1]
        feature_cnt = feature_cnt_dl

        self.ticker = ticker
        self.date_start = date_start
        self.date_end = date_end
        self.orig_data = orig_data
        self.unagg_data = unagg_data

        return data, feature_cnt

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

        # logger.info("= After Adding Targets___ and ForwardFill ========================================================")
        # logger.info(data.tail(10))

        # Reset the index of the dataframe.
        data.reset_index(inplace=True)
        dates_data = data['Date'].copy()
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
        if len(args)==1 and args[0]!='':
            return self._fetch_model_1(args[0])
        if len(kwargs) and 'model_path' in kwargs.keys():
            return self._fetch_model_1(kwargs['model_path'])

        if len(args)>1 and args[0] != '':
            ticker = args[0]
            dateStart = args[1]
            dateEnd = args[2]
            model_dir = args[3]
            return self._fetch_model_2(ticker=ticker, dateStart=dateStart, dateEnd=dateEnd, modelDir=model_dir)
        if len(kwargs) > 0 and kwargs['ticker'] != '':
            return self._fetch_model_2(ticker=kwargs['ticker'], dateStart=kwargs['dateStart'], dateEnd=kwargs['dateEnd'], modelDir=kwargs['modelDir'])
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

    def _fetch_model_2(self, ticker='', dateStart='', dateEnd='',
                       modelDir='../models/'):
        """
        Load a model given a ticker, start date, and end date.
        :param ticker:
        :param dateStart:
        :param dateEnd:
        :return model:
        """
        # Build the filepath to the model
        model_file = ''
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
        if self.Verbose:
            self.logger.info(f"Model Loaded: {model_path}")

        self.ticker = ticker
        self.date_start = dateStart
        self.date_end = dateEnd
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
                # We found a model that matches, so save it off and return True.
                self.model_path = model_file
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
        self.logger.info(f"Cache Training Data:[{symbol}]  period:[{period}]  dateStart:[{dateStart_}]  dateEnd:[{dateEnd_}]")
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
            self.model = self.load_model(model_path=(self.model_dir + self.model_path))
            self.logger.info(f"Model Loaded: {self.model_path}. Training cache will not be loaded.")

        # Allways pull data for seasonality decomposition.
        # Load training data and prepare the data
        X, y = self._fetch_n_prep(symbol, dateStart_, dateEnd_,
                                  period=period)

        # Store the date data as a strings so that pydantic can serialize it.
        # It does not do a proper job if the date is a datetime object.
        str_datesData = []
        for item in self.date_data:
            str_datesData.append(item.strftime('%Y-%m-%d'))
        str_datesData = pd.Series(str_datesData)

        # Cache the data
        training_cache = DataCache()
        training_cache.set_item('symbol', symbol)
        training_cache.set_item('dateStart', dateStart_)
        training_cache.set_item('dateEnd', dateEnd_)
        training_cache.set_item('period', period)
        tc_orig_data = self.orig_data.copy(deep=True)
        tc_orig_data.reset_index(inplace=True)
        training_cache.set_item('data', tc_orig_data.to_json())
        training_cache.set_item('feature_cnt', self.features)
        training_cache.set_item('data_scaled', list(self.data_scaled))
        training_cache.set_item('target_cnt', self.targets)
        training_cache.set_item('dates_data', str_datesData.to_json())
        training_cache.set_item('X', list(self.X))
        training_cache.set_item('y', list(self.y))

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
        for item in self.date_data:
            str_datesData.append(item.strftime('%Y-%m-%d'))
        str_datesData = pd.Series(str_datesData)

        # Cache the data
        prediction_cache = DataCache()
        prediction_cache.set_item('symbol', symbol)
        prediction_cache.set_item('dateStart', dateStart_)
        prediction_cache.set_item('dateEnd', dateEnd_)
        prediction_cache.set_item('period', period)
        tc_orig_data = self.orig_data.copy(deep=True)
        tc_orig_data.reset_index(inplace=True)
        prediction_cache.set_item('data', tc_orig_data.to_json())
        prediction_cache.set_item('feature_cnt', self.features)
        prediction_cache.set_item('data_scaled', list(self.data_scaled))
        prediction_cache.set_item('target_cnt', self.targets)
        prediction_cache.set_item('dates_data', str_datesData.to_json())
        prediction_cache.set_item('X', list(X))
        prediction_cache.set_item('y', list(y))

        # Save the cached data into this object...
        self.cached_pred_data = prediction_cache

    def cached_train_predict_report(self):
        """
        Train the model, make a prediction, and output a report.
        This method uses the cached training data and the cached prediction data,
        to train the model and make a prediction. Separating the training and prediction
        process allows for training and prediction to run concurrently, while pulling
        the training data and prediction data and caching it can be done with blocking calls.
        :return boolean:  # Returns True if the training and prediction were successful.
        """
        self.logger.debug(f"=== Started: Training and Predicting for [{self.ticker}] using cached data...")
        if self.model is None or self.force_training is True:
            tc = self.cached_train_data
            if tc is None:
                self.logger.error(f"Error: No training data cached for {self.ticker}. Cached training data was expected.")
                raise ValueError(f"Error: No training data cached for {self.ticker}. Cached training data was expected.")
            self.ticker = tc.get_item('symbol')
            self.dateStart_train = tc.get_item('dateStart')
            self.dateEnd_train = tc.get_item('dateEnd')
            self.period = tc.get_item('period')
            self.split_pcnt = tc.get_item('split_pcnt')
            self.orig_data = pd.read_json(StringIO(tc.get_item('data')))
            self.features = tc.get_item('feature_cnt')
            self.data_scaled = np.array(tc.get_item('data_scaled'))
            self.targets = tc.get_item('target_cnt')
            str_datesData = pd.Series(json.loads(tc.get_item('dates_data')))
            self.date_data = pd.to_datetime(str_datesData, format='%Y-%m-%d')
            self.X = np.array(tc.get_item('X'))
            self.y = np.array(tc.get_item('y'))
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

        # Load the cached prediction
        pc = self.cached_pred_data
        if pc is None:
            self.logger.error(f"Exception Error: No prediction data cached for {self.ticker}. Cached prediction data was expected.")
            return
        try:
            self.ticker = pc.get_item('symbol')
            self.dateStart_pred = pc.get_item('dateStart')
            self.dateEnd_pred = pc.get_item('dateEnd')
            self.orig_data = pd.read_json(StringIO(pc.get_item('data')))
            self.features = pc.get_item('feature_cnt')
            self.data_scaled = np.array(pc.get_item('data_scaled'))
            self.targets = pc.get_item('target_cnt')
            str_datesData = pd.Series(json.loads(pc.get_item('dates_data')))
            self.date_data = pd.to_datetime(str_datesData, format='%Y-%m-%d')
            self.X = np.array(pc.get_item('X'))
            self.y = np.array(pc.get_item('y'))
        except Exception as e:
            self.logger.error(f"Exception Error: Could not load prediction data: {e}")
            return

        # Make Predictions on all the data
        self.split_pcnt = 1.0
        # Perform the prediction
        # This call will also save the prediction data to this object.
        self.predict_price(self.X)
        # Perform data alignment on the prediction data.
        # Doing so makes use the the prediction deltas rather than the actual values.
        self.adjust_prediction()
        """
        - Produce a prediction chart.
        - Save the prediction data to a file or database.
        - Save to weekly or daily data to a file or database.
        - Save up/down correlation data to a file or database.
        - Perform Seasonality Decomposition.
        - Save the Seasonality Decomposition to a file or database.
        """
        self.logger.info(f"Performing price prediction for [{self.ticker}] using cached data...")
        self.save_prediction_chart(last_candles=75)
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

        self.logger.debug(f"=== Training Model [{self.ticker}] [{self.period}]...")
        # Handle the optional parameters
        if split_pcnt is None:
            split_pcnt = self.split_pcnt
        splitlimit = int(len(X) * split_pcnt)

        self.split_limit = splitlimit
        if backcandels is None:
            backcandels = self.back_candles

        data_set = self.data_scaled
        feature_cnt = self.features

        # Split the scaled data into training and testing
        # logger.info("lenX:",len(X), "splitLimit:",splitlimit)
        X_train, X_test = X[:splitlimit], X[splitlimit:] # Training data, Test Data
        y_train, y_test = y[:splitlimit], y[splitlimit:] # Training data, Test Data

        # Get the model parameters from this object
        batch_size = self.batch_size
        epochs = self.epochs
        shuffle = self.shuffle
        validation_split = self.val_split

        lstm_input = Input(shape=(backcandels, feature_cnt), name='lstm_input')
        inputs = LSTM(200, name='first_layer')(lstm_input)
        inputs = Dense(self.targets, name='dense_layer')(inputs)
        output = Activation('linear', name='output')(inputs)
        model = Model(inputs=lstm_input, outputs=output)
        adam = optimizers.Adam()
        model.compile(optimizer=adam, loss='mse')

        # Define the CSV logger
        csv_logger = CSVLogger('PricePred_keras_training_log.csv')

        # Train the model
        model.fit(x=X_train, y=y_train,
                  batch_size=batch_size, epochs=epochs,
                  shuffle=shuffle, validation_split=validation_split,
                  callbacks=[csv_logger],
                  verbose=self.keras_verbosity)

        if len(X_test) > 0:
            y_pred = model.predict(X_test)
            fy_pred = np.array(pd.DataFrame(y_pred).replace({np.nan:0}))
            fy_test = np.array(pd.DataFrame(y_test).replace({np.nan:0}))
            mse = mean_squared_error(fy_test, fy_pred)
            if self.Verbose:
                self.logger.info(f"Mean Squared Error: {mse}")

            # Restore the scale of the prediction
            pred_rescaled = self.restore_scale_pred(y_pred.reshape(-1, self.targets))
        else:
            self.logger.info("*** Won't predict. No test data.")
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
        self.dateStart_pred = pd.to_datetime(self.date_data.iloc[splitlimit+1]).strftime("%Y-%m-%d")
        self.dateEnd_pred = pd.to_datetime(self.date_data.iloc[-1]).strftime("%Y-%m-%d")
        self.model = model

        self.logger.debug(f"=== Model Training Completed [{self.ticker}] [{self.period}]...")

        return model, y_pred, mse

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
                working_dir = '/home/dsidlo/lean-trading/ml-price-prediction'
                os.chdir(working_dir)
                # Save the model...
                model.save(model_path)
            except Exception as e:
                if i < 3:
                    self.logger.warning(f"Warning: Failed to Save model [{i}] [{model_path}]\n{e}, will retry...")
                    time.sleep(2)
                    continue
                else:
                    self.logger.error(f"Error: Saving model [{model_path}]\n{e}")
                    raise ValueError(f"Error: Saving model [{model_path}]\n{e}")

        self.ticker = ticker
        self.model_path = model_path

        return model, model_path

    def predict_price(self, X_data):

        self.logger.debug(f"=== Predicting Price for [{self.ticker}] [{self.period}]...")

        # Test the model
        try:
            y_pred = self.model.predict(X_data, verbose=self.keras_verbosity)
        except Exception as e:
            self.logger.error(f"Error: Predicting Price: {e}")
            raise ValueError(f"Error: Predicting Price: {e}")

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
        self.pred_class = y_pred_rs[:, 0]
        self.pred_close = y_pred_rs[:, 1]
        self.pred_high = y_pred_rs[:, 2]
        self.pred_low = y_pred_rs[:, 3]

        self.logger.debug(f"=== Price Prediction Completed [{self.ticker}] [{self.period}]...")

        return y_pred

    def adjust_prediction(self):
        """
        Calculate the delta between actual price and prediction
        Bring the prediction closer to the price based on the delta
        This is used on scaled and scale-restored data.
        :param y_test:       # The actual prices
        :param y_pred:       # The predicted prices
        :param avg_win:      # The moving average window
        :return y_p_adj,     # The adjusted prediction
                y_p_delta:   # The deltas between the actual price and the prediction
        """
        # Gather the predicted data for the test period.
        pred_class = self.pred_rescaled[:, 0]
        pred_close = self.pred_rescaled[:, 1]
        pred_high = self.pred_rescaled[:, 2]
        pred_low = self.pred_rescaled[:, 3]

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
        pred_adj_high = [pred_adj_close[i] if pred_adj_close[i] > pred_adj_high[i] else pred_adj_high[i] for i in
                         range(0, len(pred_delta_c))]

        # -- Adjust Predicted low
        pred_delta_l = [pred_low[i - 1] - pred_low[i] for i in range(1, len(pred_low))]
        min_len = min(len(target_low), len(pred_low))
        target_low = target_low[-min_len:]
        pred_adj_low = [target_low[i] - abs(pred_delta_l[i]) for i in range(0, len(pred_delta_l))]
        #    -- Adjusted Close Prediction should not be lower than Adjusted Low Prediction
        pred_adj_low = [pred_adj_close[i] if pred_adj_close[i] < pred_adj_low[i] else
                        pred_adj_low[i] for i in range(0, len(pred_delta_l))]

        adj_pred = np.array([pred_class[-len(pred_adj_close):], pred_adj_close, pred_adj_high, pred_adj_low])
        adj_pred = np.moveaxis(adj_pred, [0], [1])

        # Calculate the strength of the prediction vs. other predictions.
        abs_deltas = np.abs(pred_delta_c)
        # Determine the rank of the last prediction from 1 to 10.
        ranking = np.digitize(abs_deltas, np.histogram(abs_deltas, bins=10)[1])
        # rank of the last value
        pred_rank = ranking[-2]                  # Index to 2nd to last value, as the last value is a placeholder.

        self.pred_last_delta = pred_delta_c[-2]  # Index to 2nd to last value, as the last value is a placeholder.
        pred_sign = np.sign(pred_delta_c[-2])    # Index to 2nd to last value, as the last value is a placeholder.
        # Invert the rank so that longs are positive and shorts are negative
        self.pred_rank = (pred_sign * pred_rank) * -1

        self.target_close = target_close
        self.target_high = target_high
        self.target_low = target_low

        self.adj_pred = adj_pred               # The adjusted predictions
        self.adj_pred_class = pred_class       # Does not get adjusted
        self.adj_pred_close = pred_adj_close   # Adjusted close
        self.adj_pred_high = pred_adj_high     # Adjusted high
        self.adj_pred_low = pred_adj_low       # Adjusted low

        return pred_adj_close, pred_adj_high, pred_adj_low

    def save_prediction_chart(self, file_path=None, last_candles=50):

        if file_path is None or file_path == '':
            file_path = self.chart_dir + self.ticker + f"_{self.period}_" + self.dateEnd_pred + ".png"
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
        plt_ohlcv = self.orig_data.iloc[split_start:, [0, 1, 2, 3, 4, 5, 6]].copy()
        if len(plt_ohlcv) <= 0:
            # Handle the side effect of dealing with data from the prediction cache...
            plt_ohlcv = self.orig_data.iloc[:, [0, 1, 2, 3, 4, 5, 6]].copy()

        ohlcv = plt_ohlcv.copy()
        ohlcv.reset_index()

        ticker = self.ticker

        if 'Date' not in ohlcv.columns:
            df_plt_test_usd = pd.concat([df_plt_test_usd, ohlcv.set_axis(df_plt_test_usd.index)], axis=1)
        else:
            df_plt_test_usd = ohlcv.copy()

        # Append a row do df_plt_test_usd for the prediction period
        # where open, high, low, and close are all equal to the last close price.
        if 'Close' in df_plt_test_usd.columns:
            last_close = df_plt_test_usd['Close'].iloc[-1]
            last_adj_close = df_plt_test_usd['Adj Close'].iloc[-1]
            last_date = df_plt_test_usd['Date'].iloc[-1]
            next_date = pd.to_datetime(last_date) + pd.DateOffset(days=1)
            new_row = {"Date": next_date,
                       "Open": last_close, "High": last_close, "Low": last_close, "Close": last_close,
                       "Adj Close": last_adj_close, "Volume": 0}
        else:
            last_close = df_plt_test_usd['Adj Close'].iloc[-1]
            last_adj_close = df_plt_test_usd['Adj Close'].iloc[-1]
            last_date = df_plt_test_usd['Date'].iloc[-1]
            next_date = pd.to_datetime(last_date) + pd.DateOffset(days=1)
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

        title = (f'Ticker: {ticker} -- Period[ {self.period}] -- {self.dateStart_pred} to {self.dateEnd_pred}\n'
                 f'Predictions High: {self.adj_pred_high[-1].round(2)}  Close: {self.adj_pred_close[-1].round(2)}  Low: {self.adj_pred_low[-1].round(2)}')
        kwargs = dict(type='candle', volume=True, figratio=(11, 6), figscale=2, warn_too_much_data=10000, title=title)

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

        save = dict(fname=file_path, dpi=300, pad_inches=0.25)
        df_plt_test_usd.ffill(inplace=True)
        try:
            fig, axis = mpf.plot(df_plt_test_usd[-min_len:], **kwargs,
                                 style='binance', addplot=preds, savefig=save,
                                 returnfig=True)
        except Exception as e:
            self.logger.error(f"Error: Could not plot chart. {e}")

        return file_path

    def save_prediction_data(self, file_path=None, last_candles=None):
        # Copy from the original data the OHLCV data for the prediction period...
        # df_ohlcv Will have an 'Date' as it's index...
        if 'Close' in self.orig_data.columns:
            df_ohlcv = pd.DataFrame(self.orig_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]).tail(len(self.pred) - 1)
        else:
            df_ohlcv = pd.DataFrame(self.orig_data[['Open', 'High', 'Low', 'Adj Close', 'Volume']]).tail(len(self.pred) - 1)
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
        next_date = pd.to_datetime(last_date) + pd.DateOffset(days=1)
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

        # We can limit the number of candles to save if requested...
        if last_candles is not None and len(df) > last_candles:
            df = df.tail(last_candles)

        # == Create the file-path to save the prediction data...
        file_path = self.preds_dir + self.ticker + "_" + self.dateEnd_pred + ".csv"
        self.preds_path = file_path
        try:
            # Save the prediction data to a CSV file...
            df.to_csv(file_path, index=False)
        except Exception as e:
            self.logger.error(f"Error: Could not save prediction data to:{file_path}")
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
            self.logger.error(f"Error: Could not save analysis data to: {file_path}")
            self.logger.error("Exception: {e}")
        self.analysis_path = file_path
        file_paths.append(file_path)

        return file_paths

    def _fetch_n_prep(self, ticker: str, date_start: str, date_end: str,
                      period: str = None, split_pcnt: float =0.05,
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
        X_train, X_test = X[:splitlimit], X[splitlimit:] # Training data, Test Data
        y_train, y_test = y[:splitlimit], y[splitlimit:] # Training data, Test Data

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

    def fetch_train_and_predict(self, ticker,
                                train_date_start, train_date_end,
                                pred_date_start, pred_date_end,
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
        if use_curr_model and self.force_training is False:
            # Load an existing model if it exists
            model_path = self.model_dir + ticker + f"_{period}_" + train_date_start + "_" + train_date_end + ".keras"
            if os.path.exists(model_path):
                model = self.load_model(model_path)
                self.logger.info(f">>> Model Loaded: {model_path}")
            else:
                self.logger.info(f"=== Model Not Found: {model_path}")

        if model is None:
            # Load training data and prepare the data
            X, y = self._fetch_n_prep(ticker, train_date_start, train_date_end,
                                      period=period, split_pcnt=0)

            # ============== Train the model
            # Use a small batch size and epochs to test the model training
            # Training split the X & y data into training and testing data
            # What is returned is the model, the prediction, and the mean squared error.
            model, y_pred, mse = self.train_model(X, y,
                                                  split_pcnt=split_pcnt,backcandels=backcandels)

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

        if model_path is None:
            self.logger.error("Error: The model_path parameter is required.")
            raise ValueError("Error: The model_path parameter is required.")
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
        orig_predictions = model.predict(X)
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
        closes = np.array(orig_data['Adj Close'])[:len(X)-1]
        closes = np.append(closes, closes[-1])
        highs = np.array(orig_data['High'])[:len(X)-1]
        highs = np.append(highs, highs[-1])
        lows = np.array(orig_data['Low'])[:len(X)-1]
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
        pred_dates = pd.Series._append(pred_dates, pd.Series(self.date_data.iloc[-1] + timedelta(days=1)))

        return self.adj_pred, pred_dates

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
        elements = len(self.pred_rescaled) - 1
        # logger.info(f"elements:{elements}")
        tot_deltas = 0
        tot_tradrng = 0
        for i in range(-1, -elements, -1):
            actual = self.orig_data['Adj Close'].iloc[i - 1]
            predval = self.pred_rescaled[i - 1][1]
            pred_delta = abs(predval - actual)
            tot_deltas += pred_delta
            trd_rng = abs(self.orig_data['High'].iloc[i] - self.orig_data['Low'].iloc[i])
            tot_tradrng += trd_rng
            self.logger.info(f"{i}: Close: {actual.round(2)}  Predicted: ${predval.round(2)}  Actual: ${actual.round(2)}  Delta: ${pred_delta.round(6)}  Trade Rng: ${trd_rng.round(2)}")

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
            self.logger.info(f"{i}: Close {actual.round(2)}  Predicted: ${predval.round(2)}  Actual: ${actual.round(2)}  Delta:${pred_delta.round(6)}  Trade Rng: ${trd_rng.round(2)}")

        self.logger.info("============================================================================")
        self.logger.info(f"Mean Trading Range: ${round(tot_tradrng / elements, 2)}")
        self.logger.info(f"Mean Delta (Actual vs Prediction): ${round((tot_deltas / elements), 2)}")
        self.logger.info("============================================================================")

        analysis['actual_vs_adj_pred'] = {
            'mean_trading_range': round(tot_tradrng / elements, 2),
            'mean_delta': round((tot_deltas / elements), 2)}

        if self.seasonal_dec is not None:
            # Analyze the trend correlation between seasonality trend and the
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
            sd_trends = [1 if sd_trends[i] > sd_trends[i-1] else -1 for i in range(len(sd_trends)-1)]
            # Get deltas between days
            sd_deltas = [sd_trends[i] - sd_trends[i-1] for i in range(1, len(sd_trends))]
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
                self.logger.error(f"Failed to collect trend data for self:{self.ticker}, for seasonality correlation.")
            # Make sure that the lengths of self_trends and sd_trends are the same
            min_len = min(len(self_trends), len(sd_trends))
            self_trends = self_trends[-min_len:]
            sd_trends = sd_trends[-min_len:]

            # Calculate the correlation
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

            self.pred_strength = round((self.pred_rank + (self.season_rank * self.season_corr)) / 2, 4)
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
            plt.plot(target_close, color='black', label='Target Close', marker='.')
        # Original Prediction
        if target_high is not None:
            plt.plot(target_high[1:], color='blue', label='Target High', marker='1')
        if target_low is not None:
            plt.plot(target_low[1:], color='red', label='Target Low', marker='1')
        if adj_pred_close is not None:
            plt.plot(adj_pred_close[1:], color='green', label='Adj Close', linestyle='-.', marker='o')
        if adj_pred_high is not None:
            plt.plot(adj_pred_high[1:], color='violet', label='Adj High', linestyle='-.', marker='o')
        if adj_pred_low is not None:
            plt.plot(adj_pred_low[1:], color='orange', label='Adj Low', linestyle='-.', marker='o')
        if title != '':
            plt.title(title)
        plt.legend()
        plt.grid()
        plt.show()
        return plt

    def periodic_correlation(self, ppo, pc_period_len: int = None):
        """
        Calculate the correlation of the predicted prices with the actual prices.
        :param ppo:
        # Optional Parameters
        :param pc_period_len:
        :return:
        """
        # Verify that the periods align.
        if type(ppo).__name__ != 'PricePredict':
            self.logger.error(f"ppo must be a PricePredict object. ppo[{type(ppo).__name__}]")
            raise ValueError(f"ppo must be a PricePredict object. ppo[{type(ppo).__name__}]")

        min_data_points = 50

        if ppo.orig_data is None or len(ppo.orig_data) < min_data_points:
            self.logger.info(f"ppo[{ppo.ticker}] has less than 150 data points.")
            return None

        # Verify that self.date_end and ppo.date_end are
        # within 3 days of eachother.
        if abs((datetime.strptime(self.date_end, "%Y-%m-%d") - datetime.strptime(ppo.date_end, "%Y-%m-%d")).days) > 3:
            self.logger.info(f"End dates must be within 3 days of each other. self[{self.ticker} {self.date_end}] != ppd[{ppo.ticker} {ppo.date_end}]")
            return None

        # Get the smaller of the end dates between self and ppo
        target_end_date = min(self.date_end, ppo.date_end)
        # Get the difference in days between self and ppo
        days_diff = abs((datetime.strptime(self.date_end, "%Y-%m-%d") - datetime.strptime(ppo.date_end, "%Y-%m-%d")).days)
        if target_end_date == self.date_end:
            self_days_diff = 0
            ppo_days_diff = days_diff
        else:
            self_days_diff = days_diff
            ppo_days_diff = 0

        # # Verify that end dates are the same.
        # if self.date_end != ppo.date_end:
        #     raise ValueError(f"End dates must be the same. self[{self.date_end}] != ppd[{ppo.date_end}]")

        # Grab the last pc_period_len of the data
        self_len = len(self.orig_data)
        ppo_len = len(ppo.orig_data)
        min_len = min(self_len - self_days_diff - 1, ppo_len - ppo_days_diff - 1)
        if pc_period_len is None:
            pc_period_len = min_len
        elif pc_period_len < min_len:
            self.logger.warn(f"pc_period_len [{pc_period_len}] is less than the minimum length of the data [{min_len}]. self_len[{self_len}], ppd_len[{ppo_len}]")

        # Get data from each object
        self_data = self.orig_data.iloc[-(pc_period_len + self_days_diff):-(1 + self_days_diff)]
        ppo_data = ppo.orig_data.iloc[-(pc_period_len + ppo_days_diff):-(1 + ppo_days_diff)]

        # Get up days vs down days
        close_col = 'Close'
        if close_col not in self_data.columns:
            close_col = 'Adj Close'
        self_trends = [1 if self_data[close_col].iloc[i] > self_data['Open'].iloc[i] else -1 for i in range(len(self_data))]
        close_col = 'Close'
        if close_col not in ppo_data.columns:
            close_col = 'Adj Close'
        ppo_trends = [1 if ppo_data[close_col].iloc[i] > ppo_data['Open'].iloc[i] else -1 for i in range(len(ppo_data))]

        # Calculate the correlation
        try:
            corr_list = [self_trends[i] + ppo_trends[i] for i in range(len(self_trends))]
        except Exception as e:
            self.logger.error(f"Error: {e}")
            self.logger.error(f"self.ticker:{self.ticker}-len:{len(self_trends)} ppo.ticker:{ppo.ticker}-len:{len(ppo_trends.shape)}")
            raise ValueError(f"Error: {e}")

        total_days = len(corr_list)
        correlated_days = corr_list.count(2) + corr_list.count(-2)
        uncorrelated_days = corr_list.count(0)
        pct_corr = correlated_days / total_days
        pct_uncorr = uncorrelated_days / total_days
        self.logger.info(f"Days: {total_days} Correlated Days: {correlated_days}  Uncorrelated Days: {uncorrelated_days}")
        self.logger.info(f"Correlated Days: {pct_corr}%  Uncorrelated Days: {pct_uncorr}%")
        ret_dict = {'total_days': total_days,
                    'correlated_days': correlated_days,
                    'uncorrelated_days': uncorrelated_days,
                    'pct_corr': pct_corr,
                    'pct_uncorr': pct_uncorr}

        return ret_dict

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
            # Set date_end to today's date
            self.date_end = datetime.now().strftime("%Y-%m-%d")
            # Set date_start to 2 years before today's date aligned to a Monday
            self.date_start = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
            self.fetch_data_yahoo(self.ticker, self.date_start, self.date_end, self.period)

        data = self.orig_data

        if 'Date' in data.columns and 'Date' not in data.index:
            data.set_index('Date', inplace=True)
            # Add an additional row to data for the next day's prediction
            last_date = data.index[-1]
            next_date = last_date + timedelta(days=1)
        else:
            # Add an additional row to data for the next day's prediction
            last_date = self.date_data[self.date_data.index[-1]]
            next_date = last_date + timedelta(days=1)

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
            self.logger.error(f"Error: Could not perform seasonal decomposition for [{self.ticker}] [{self.period}]\n{e}")
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