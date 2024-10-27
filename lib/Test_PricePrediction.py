"""
File: Test_PricePredict.py

Notes:
  - Prediction tests run with fewer epochs and batch size to speed up testing.
    Thus, plots may not be as accurate as they could be.
"""
import os.path
import pytest
import numpy as np
import pandas as pd
import logging
import sys
import json

from unittest import TestCase
from pricepredict import PricePredict
from datetime import datetime, timedelta


class Test_PricePredict(TestCase):

    def __init__(self, *args, **kwargs):

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(logging.StreamHandler(stream=sys.stdout))
        TestCase.__init__(self, *args, **kwargs)

    @staticmethod
    def _create_test_model(this_test, pp, ticker, test_ticker, mdl_start_date, mdl_end_date):

        data, features = pp.fetch_data_yahoo(ticker, mdl_start_date, mdl_end_date)
        this_test.assertGreaterEqual(len(data), 1, "data: Wrong length")

        # Augment the data with additional indicators/features
        aug_data, features, targets, dates_data = pp.augment_data(data, features)
        this_test.assertGreater(features, 1, "features: Wrong count")
        this_test.assertGreater(len(dates_data), 1, "dates_data: Wrong length")
        this_test.assertGreater(len(aug_data), 1, "aug_data: Wrong length")

        # Scale the data so the model can use it more effectively
        scaled_data, scaler = pp.scale_data(aug_data)
        this_test.assertGreater(len(scaled_data), 1, "scaled_data: Wrong length")
        this_test.assertIsNotNone(scaler, "scaler: Wrong length")

        # Prepare the scaled data for model inputs
        X, y = pp.prep_model_inputs(scaled_data, features)
        this_test.assertEqual(len(aug_data) - pp.back_candles, len(X), "X: Wrong length")
        this_test.assertEqual(len(aug_data) - pp.back_candles, len(y), "y: Wrong length")

        # Use a small batch size and epochs to test the model training
        pp.epochs = 3
        pp.batch_size = 1
        # Train the model
        model, y_pred, mse = pp.train_model(X, y)
        this_test.assertIsNotNone(model, "model: Wrong length")

        # Save the model (with a test marker)
        model, model_path = pp.save_model(ticker=test_ticker)
        # Checks to verify that we have a saveable model object
        this_test.assertIsNotNone(model, "model: Is None")
        save_op = getattr(model, 'save', None)
        this_test.assertTrue(callable(save_op), "model: 'save' method not found")
        # Verify that the _Test_ model file was created
        this_test.assertTrue(os.path.isfile(model_path), "model_path: File does not exist")

        return model_path

    def test_init(self):
        pp = PricePredict(model_dir='../models/', chart_dir='../charts/', preds_dir='../predictions/')
        has_meth = callable(getattr(pp, 'fetch_data_yahoo', None))
        self.assertTrue(has_meth, "fetch_data_yahoo: Method not found")

    def test_chk_yahoo_ticker(self):
        # Create an instance of the price prediction object
        pp = PricePredict(model_dir='../models/', chart_dir='../charts/', preds_dir='../predictions/')

        ticker = "AAPL"
        ticker_data = pp.chk_yahoo_ticker(ticker)
        self.assertEqual("AAPL", pp.ticker_data.get('symbol'), f"ticker[{pp.ticker_data.get('symbol')}]: Ticker should be AAPL")

        ticker = "Test-AAPL"
        ticker_data = pp.chk_yahoo_ticker(ticker)
        self.assertEqual("AAPL", pp.ticker_data.get('symbol'), f"ticker[{pp.ticker_data.get('symbol')}]: Ticker should be AAPL")

        ticker = "yy043xx"
        ticker = pp.chk_yahoo_ticker(ticker)
        self.assertEqual(None, ticker, f"ticker[{ticker_data}]: Ticker returned should be None")

    def test_fetch_data_yahoo(self):
        # Create an instance of the price prediction object
        pp = PricePredict(model_dir='../models/', chart_dir='../charts/', preds_dir='../predictions/')

        # Load data from Yahoo Finance
        ticker = "AAPL"
        start_date = "2020-01-01"
        end_date = "2023-12-31"
        # Load data from Yahoo Finance
        data, features = pp.fetch_data_yahoo(ticker, start_date, end_date)
        self.assertGreaterEqual(len(data), 1, "Wrong data length")
        self.assertEqual(pp.ticker, ticker, "Wrong ticker")

        # Now try loading a different ticker
        ticker = "TSLA"
        data, features = pp.fetch_data_yahoo(ticker, start_date, end_date)
        self.assertGreaterEqual(len(data), 1, "Wrong data length")
        self.assertEqual(pp.ticker, ticker, "Wrong ticker")

        # Now try loading a different ticker
        ticker = "TSLA"
        pp.period = PricePredict.Period1min
        # We need at least @2000 inputs to train the model
        # - We can only get 7 Days of data for 1 minute period
        # - We can get 35 days of data for 5 minute period
        # - 15min 107 days: Fails - Only pull 35 days (1006 rows)
        # - 1hour 428 days: Passes
        days_to_load = int(3000 / PricePredict.PeriodMultiplier[pp.period])
        # Set the start date to 7 days ag
        start_date = (datetime.now() - timedelta(days=days_to_load)).strftime("%Y-%m-%d")
        # Set the end date to today
        end_date = datetime.now().strftime("%Y-%m-%d")
        data, features = pp.fetch_data_yahoo(ticker, start_date, end_date)
        self.assertGreaterEqual(len(data), 1000, "Wrong data length")
        self.assertEqual(pp.ticker, ticker, "Wrong ticker")

    def test_check_for_recent_model(self):
        """
        Place some test files into the model directory and used
        them to test the check_for_model method.
        - Create a positive test case where the model file exists
        - Create a negative test case where the model file does not exist
        :return:
        """
        # Create an instance of the price prediction object
        pp = PricePredict(model_dir='../models/', chart_dir='../charts/', preds_dir='../predictions/')

        # Generate a list of 10 random uppercase symbols that
        # represent tickers of 2 to 4 uppercase alpha characters...
        tickers = []
        for i in range(10):
            ticker = ""
            for j in range(3):
                ticker += chr(np.random.randint(ord('A'), ord('Z')))
            tickers.append(ticker)
        # Generate a list of 10 random dates between 2015 and 2020-07-30
        start_dates = []
        for i in range(10):
            year = np.random.randint(2015, 2020)
            month = np.random.randint(1, 12)
            day = np.random.randint(1, 28)
            date = f"{year}-{month:02d}-{day:02d}"
            start_dates.append(date)
        # For each date in start_dates, create an end_date that is greater
        # than the start_date and ends between 2024-01-01 and 2024-07-30.
        end_dates = []
        for i in range(10):
            year = np.random.randint(2020, 2024)
            month = np.random.randint(1, 12)
            day = np.random.randint(1, 28)
            date = f"{year}-{month:02d}-{day:02d}"
            end_dates.append(date)
        # Create a list of 10 period that are either 'D' or 'W'
        periods = []
        for i in range(10):
            period = 'D' if np.random.randint(0, 2) == 0 else 'W'
            periods.append(period)
        # Combine the list of 10 tickers, start_dates, end_dates and periods
        # into a list of file paths that start with the 'models/' directory
        model_files = []
        for i in range(10):
            model_files.append(f"{pp.model_dir}/{tickers[i]}_{periods[i]}_{start_dates[i]}_{end_dates[i]}.keras")
        # Create the model files
        for i in range(10):
            with open(model_files[i], 'w') as f:
                f.write("Test file")

        # Test the check_for_model method

        # Negative test case
        # Generate a list of dickers that do not exist in the tickers list
        # where the ticker is from 2 to 4 uppercase alpha characters...
        # Create the tickers that are not in the ticker list
        xtickers = []
        for i in range(10):
            while True:
                ticker = ""
                for j in range(3):
                    ticker += chr(np.random.randint(ord('A'), ord('Z')))
                if ticker not in xtickers:
                    xtickers.append(ticker)
                    break
        for i in range(10):
            with open(model_files[i], 'w') as f:
                f.write("Test file")
        # Check for the model files that should not exist
        for i in range(10):
            model_found = pp.check_for_recent_model(xtickers[i], start_dates[i], end_dates[i], periods[i])
            self.assertFalse(model_found, f"Model file [{xtickers[i]}] should not exist")
        # Check for the model files that exist
        for i in range(10):
            model_found = pp.check_for_recent_model(xtickers[i], start_dates[i], end_dates[i], periods[i])
            self.assertFalse(model_found, f"Model file [{xtickers[i]}] should not exist")
        # Delete the test model files
        for i in range(10):
            if os.path.isfile(model_files[i]):
                os.remove(model_files[i])

    def test_cache_training_data(self):
        # Create an instance of the price prediction object
        pp = PricePredict(model_dir='../models/', chart_dir='../charts/', preds_dir='../predictions/')

        # =========================================
        # First we need to create a model file
        ticker = "NEE"
        test_ticker = "Test-" + ticker
        # Data download dates
        start_date = "2024-06-01"
        end_date = "2024-07-30"
        period = pp.PeriodDaily

        model_path = Test_PricePredict._create_test_model(self, pp, ticker, test_ticker, start_date, end_date)
        # Create a Model that can be loaded...
        # =========================================

        # This test find a recent model file and loads the model
        # - Does not download data from Yahoo Finance
        pp.cache_training_data(test_ticker, start_date, end_date, period)
        self.assertIsNotNone(pp.model, "pp.model: is None, but should be loaded")
        self.assertIsNotNone(pp.cache_training_data, "pp.cache_training_data: should be None, but has data")

        # Now delete the model file that we created
        if os.path.isfile(model_path):
            os.remove(model_path)

        # This test does not find a recent model file and downloads data from Yahoo Finance
        pp.cache_training_data(test_ticker, start_date, end_date, period)
        self.assertIsNone(pp.model, "pp.model: should be None, but has been loaded")
        self.assertIsNotNone(pp.cache_training_data, "pp.cache_training_data: is None, but should have data")

        tc = pp.cached_train_data
        sym = tc.get_item('symbol')
        self.assertEqual(test_ticker, sym, f"pp.cache_training_data['symbol']: Wrong ticker [{sym}] expected [{test_ticker}]")
        dateStart_train = tc.get_item('dateStart')
        self.assertEqual(start_date, dateStart_train, f"pp.cache_training_data['dateStart']: Wrong date [{dateStart_train}] expected [{start_date}]")
        dateEnd_train = tc.get_item('dateEnd')
        self.assertEqual(end_date, dateEnd_train, f"pp.cache_training_data['dateEnd']: Wrong date [{dateEnd_train}] expected [{end_date}]")
        period = tc.get_item('period')
        self.assertEqual(pp.PeriodDaily, period, f"pp.cache_training_data['period']: Wrong period [{period}] expected [{pp.PeriodDaily}]")
        orig_data = pd.read_json(tc.get_item('data'))
        self.assertGreater(len(orig_data), 1,"orig_data: Wrong length")
        feature_cnt = tc.get_item('feature_cnt')
        self.assertEqual(19, feature_cnt, f"pp.cache_training_data['feature_cnt']: Wrong count [{feature_cnt}] expected [19]")
        data_scaled = np.array(tc.get_item('data_scaled'))
        self.assertGreater(len(data_scaled), 1, "data_scaled: Wrong length")
        target_cnt = tc.get_item('target_cnt')
        self.assertEqual(4, target_cnt, f"pp.cache_training_data['target_cnt']: Wrong count [{target_cnt}] expected [4]")
        dates_data = pd.DataFrame(list(json.loads(tc.get_item('dates_data')).items()),
                                  columns=['index', 'Date']).set_index('index')
        self.assertGreater( len(dates_data), 1, "dates_data: Wrong length")
        X = np.array(tc.get_item('X'))
        self.assertGreater(len(X), 1, "X: Wrong length")
        y = np.array(tc.get_item('y'))
        self.assertGreater(len(y), 1, "y: Wrong length")

        # TODO: Add a test for the existence of the seasonality chart
        sd_file_exists = os.path.isfile(pp.seasonal_chart_path)
        # By default we don't save_plot the seasonal chart file
        self.assertFalse(sd_file_exists, f"Seasonal chart file [{pp.seasonal_chart_path}] does not exist")
        if sd_file_exists:
            # Delete the seasonal chart file
            os.remove(pp.seasonal_chart_path)

        # Delete the model file that we crated and loaded
        if os.path.isfile(model_path):
            os.remove(model_path)
        # End of the load_model test
        # =========================================

    def test_cache_prediction_data(self):
        # Create an instance of the price prediction object
        pp = PricePredict(model_dir='../models/', chart_dir='../charts/', preds_dir='../predictions/')

        ticker = "JD"
        test_ticker = "Test-" + ticker
        # Data download dates
        start_date = "2024-06-01"
        end_date = "2024-07-30"
        period = pp.PeriodDaily

        # This test does not find a recent model file and downloads data from Yahoo Finance
        pp.cache_prediction_data(ticker, start_date, end_date, period)
        self.assertIsNotNone(pp.cache_prediction_data, "pp.cache_pred_data: is None, but should have data")
        pc = pp.cached_pred_data
        sym = pc.get_item('symbol')
        self.assertEqual(ticker, sym, f"pp.cache_training_data['symbol']: Wrong ticker [{sym}] expected [{ticker}]")
        dateStart_train = pc.get_item('dateStart')
        self.assertEqual(start_date, dateStart_train, f"pp.cache_training_data['dateStart']: Wrong date [{dateStart_train}] expected [{start_date}]")
        dateEnd_train = pc.get_item('dateEnd')
        self.assertEqual(end_date, dateEnd_train, f"pp.cache_training_data['dateEnd']: Wrong date [{dateEnd_train}] expected [{end_date}]")
        period = pc.get_item('period')
        self.assertEqual(pp.PeriodDaily, period, f"pp.cache_training_data['period']: Wrong period [{period}] expected [{pp.PeriodDaily}]")
        orig_data = pd.read_json(pc.get_item('data'))
        self.assertGreater(len(orig_data), 1,"orig_data: Wrong length")
        feature_cnt = pc.get_item('feature_cnt')
        self.assertEqual(19, feature_cnt, f"pp.cache_training_data['feature_cnt']: Wrong count [{feature_cnt}] expected [19]")
        data_scaled = np.array(pc.get_item('data_scaled'))
        self.assertGreater(len(data_scaled), 1, "data_scaled: Wrong length")
        target_cnt = pc.get_item('target_cnt')
        self.assertEqual(4, target_cnt, f"pp.cache_training_data['target_cnt']: Wrong count [{target_cnt}] expected [4]")
        dates_data = pd.DataFrame(list(json.loads(pc.get_item('dates_data')).items()),
                                  columns=['index', 'Date']).set_index('index')
        self.assertGreater( len(dates_data), 1, "dates_data: Wrong length")
        X = np.array(pc.get_item('X'))
        self.assertGreater(len(X), 1, "X: Wrong length")
        y = np.array(pc.get_item('y'))
        self.assertGreater(len(y), 1, "y: Wrong length")

        pass

    def test_cached_train_predict_report(self):
        # Create an instance of the price prediction object
        pp = PricePredict(model_dir='../models/', chart_dir='../charts/', preds_dir='../predictions/')

        ticker = "BA"
        test_ticker = "Test-" + ticker
        # Data download dates
        trn_start_date = "2020-07-01"
        trn_end_date = "2024-07-30"
        prd_start_date = "2024-07-15"
        prd_end_date = "2024-08-22"
        period = pp.PeriodDaily

        # Pull training data from Yahoo Finance
        pp.cache_training_data(test_ticker, trn_start_date, trn_end_date, period)
        # Pull prediction data from Yahoo Finance
        pp.cache_prediction_data(test_ticker, prd_start_date, prd_end_date, period)
        # Perform the training and prediction and reporting
        pp.cached_train_predict_report()

        # Delete the resulting data...
        if os.path.isfile(pp.model_path):
            os.remove(pp.model_path)
        # Delete the resulting predictions file...
        if os.path.isfile(pp.preds_path):
            os.remove(pp.preds_path)
        # Delete the resulting analysis file...
        if os.path.isfile(pp.analysis_path):
            os.remove(pp.analysis_path)

        pass

    def test_aggregate_data (self):
        # Create an instance of the price prediction object
        pp = PricePredict(period='W', model_dir='../models/', chart_dir='../charts/', preds_dir='../predictions/')

        # Set the ticker and date range (what we want to pull from yahoo finance)
        ticker = "AAPL"
        start_date = "2020-01-01"
        end_date = "2023-12-31"

        # Load data from Yahoo Finance
        data, features = pp.fetch_data_yahoo(ticker, start_date, end_date)
        self.assertGreaterEqual(len(data), 1, "data: Wrong length")

        # Aggregate the data
        agg_data = pp.aggregate_data(data, pp.period)

        # Check the data
        self.assertEqual(209, agg_data.shape[0], "agg_data: Wrong length")
        sum_1 = data.sum()
        sum_2 = agg_data.sum()
        # Volume is the only aggregate value that can be checked as the rest
        # are not sums, but first, last, max and min values.
        self.assertEqual(sum_1['Volume'], sum_2['Volume'], "agg_data: Data is incorrect")

    def test_augment_data(self):
        pp = PricePredict(model_dir='../models/', chart_dir='../charts/', preds_dir='../predictions/')
        start_date = "2020-01-01"
        end_date = "2023-12-31"
        data, features = pp.fetch_data_yahoo("AAPL", start_date, end_date)
        self.assertGreaterEqual(len(data), 1, "data: Wrong length")

        aug_data, features, targets, dates_data = pp.augment_data(data, features)
        self.assertEqual(features, 19, "features: Wrong count")
        self.assertEqual(targets, 4, "features: Wrong count")
        self.assertEqual(1006, len(dates_data), "dates_data: Wrong length")
        # Augmented data has 1 additional period added to it for the prediction
        # placeholder.  This is why the length is 1007.
        self.assertEqual(1007, len(aug_data), "aug_data: Wrong length")

    def test_scale_data(self):
        # Create an instance of the price prediction object
        pp = PricePredict(model_dir='../models/', chart_dir='../charts/', preds_dir='../predictions/')

        # Load data from Yahoo Finance
        ticker = "AAPL"
        start_date = "2020-01-01"
        end_date = "2023-12-31"
        data, features = pp.fetch_data_yahoo(ticker, start_date, end_date)
        self.assertGreaterEqual(len(data), 1, "data: Wrong length")

        # Augment the data with additional indicators/features
        aug_data, features, targets, dates_data = pp.augment_data(data, features)
        self.assertEqual(19, features, "features: Wrong count")
        self.assertEqual(1006, len(dates_data), "dates_data: Wrong length")
        self.assertEqual(1007, len(aug_data), "aug_data: Wrong length")

        # Scale the data so the model can use it more effectively
        scaled_data, scaler = pp.scale_data(aug_data)
        self.assertEqual(1007, len(scaled_data), "scaled_data: Wrong length")
        self.assertIsNotNone(scaler, "scaler: Wrong length")

    def test_restore_scale_pred(self):
        # Create an instance of the price prediction object
        pp = PricePredict(model_dir='../models/', chart_dir='../charts/', preds_dir='../predictions/')

        # Load data from Yahoo Finance
        ticker = "IBM"
        start_date = "2020-01-01"
        end_date = "2023-12-31"
        data, features = pp.fetch_data_yahoo(ticker, start_date, end_date)
        self.assertGreaterEqual(len(data), 1, "data: Wrong length")

        # Augment the data with additional indicators/features
        aug_data, features, targets, dates_data = pp.augment_data(data, features)
        self.assertEqual(19, features, "features: Wrong count")
        self.assertEqual(1006, len(dates_data), "dates_data: Wrong length")
        self.assertEqual(1007, len(aug_data), "aug_data: Wrong length")

        # Scale the data so the model can use it more effectively
        scaled_data, scaler = pp.scale_data(np.array(aug_data))
        self.assertEqual(1007, len(scaled_data), "scaled_data: Wrong length")
        self.assertIsNotNone(scaler, "scaler: Scaler is None")

        # The restore_scale_pred function takes a single column of predictions of
        # the scaled 'TargetNextClose' feature and restores the original scale.
        scaled_col = scaled_data[:, -pp.targets:]
        restored_pred = pp.restore_scale_pred(scaled_col)
        self.assertEqual(1007, len(restored_pred), "restored_pred: Wrong length")
        # Check tolerance of 99.9% accuracy...
        eql_array = np.isclose(np.array(aug_data['TargetNextClose']), restored_pred[:, 1])
        pcnt_true = np.count_nonzero(eql_array) / len(eql_array)
        self.assertGreater(pcnt_true, .999, "restored_pred/TargetNextClose: Wrong data")
        eql_array = np.isclose(np.array(aug_data['TargetNextHigh']), restored_pred[:, 2])
        pcnt_true = np.count_nonzero(eql_array) / len(eql_array)
        self.assertGreater(pcnt_true, .999, "restored_pred/TargetNextHigh: Wrong data")
        eql_array = np.isclose(np.array(aug_data['TargetNextLow']), restored_pred[:, 3])
        pcnt_true = np.count_nonzero(eql_array) / len(eql_array)
        self.assertGreater(pcnt_true, .999, "restored_pred/TargetNextLow: Wrong data")

    def test_load_model(self):
        # Create an instance of the price prediction object
        pp = PricePredict(model_dir='../models/', chart_dir='../charts/', preds_dir='../predictions/')

        # =========================================
        # First we need to create a model file
        ticker = "CSCO"
        test_ticker = "Test-" + ticker
        # Data download dates
        start_date = "2024-06-01"
        end_date = "2024-07-30"
        # Model file dates
        mdl_start_date = "2015-01-01"
        mdl_end_date = "2024-07-30"

        model_path = Test_PricePredict._create_test_model(self, pp, ticker, test_ticker, mdl_start_date, mdl_end_date)
        # Create a Model that can be loaded...
        # =========================================

        # =========================================
        # Load data from Yahoo Finance
        data, features = pp.fetch_data_yahoo(ticker, start_date, end_date)
        self.assertEqual(39, len(data), "data: Wrong length")
        data, features, targets, dates = pp.augment_data(data, features)
        self.assertEqual(19, features, "Wrong feature count")

        # Load an existing model file
        model_path = pp.model_path
        # Load model *args
        # Load the model via *args
        model = pp.load_model(model_path)
        self.assertIsNotNone(model, "Model not loaded")
        save_op = getattr(model, 'save', None)
        self.assertTrue(callable(save_op), "model: 'save' method not found")

        # Load model via **kwargs
        model = pp.load_model(model_path=model_path)
        self.assertIsNotNone(model, "Model not loaded")
        save_op = getattr(model, 'save', None)
        self.assertTrue(callable(save_op), "model: 'save' method not found")

        # Load model *args, build model_path
        model = pp.load_model(test_ticker, mdl_start_date, mdl_end_date, pp.model_dir)
        save_op = getattr(model, 'save', None)
        self.assertTrue(callable(save_op), "model: 'save' method not found")

        # Load model **kwargs, build model_path
        model = pp.load_model(ticker=test_ticker, dateStart=mdl_start_date, dateEnd=mdl_end_date, modelDir=pp.model_dir)
        self.assertIsNotNone(model, "Model not loaded")
        save_op = getattr(model, 'save', None)
        self.assertTrue(callable(save_op), "model: 'save' method not found")

        # Delete the model file that we crated and loaded
        if os.path.isfile(model_path):
            os.remove(model_path)
        # End of the load_model test
        # =========================================

    def test_prep_model_inputs(self):
        # Create an instance of the price prediction object
        pp = PricePredict(model_dir='../models/', chart_dir='../charts/', preds_dir='../predictions/')

        # =========================================
        # First we need to create a model file
        ticker = "INTC"
        test_ticker = "Test-" + ticker
        # Data download dates
        start_date = "2023-06-01"
        end_date = "2024-07-30"
        # Model file dates
        mdl_start_date = "2015-01-01"
        mdl_end_date = "2024-07-30"

        model_path = Test_PricePredict._create_test_model(self, pp, ticker, test_ticker, mdl_start_date, mdl_end_date)
        # Create a Model that can be loaded...
        # =========================================

        # =========================================
        # Load data from Yahoo Finance...
        data, features = pp.fetch_data_yahoo(ticker, start_date, end_date)
        self.assertEqual(291, len(data), "data: Wrong length")

        # Load am existing model file...
        model_path = pp.model_dir + test_ticker + f"_{pp.period}_" + mdl_start_date + "_" + mdl_end_date + ".keras"
        # Load the model
        model = pp.load_model(model_path)
        self.assertIsNotNone(model, "Model not loaded")
        self.assertGreaterEqual(len(data), 100, "data: Wrong length")

        # Augment the data
        aug_data, features, targets, dates_data = pp.augment_data(data, features)
        self.assertEqual(19, features, "features: Wrong count")
        self.assertEqual(291, len(dates_data), "dates_data: Wrong length")
        self.assertEqual(292, len(aug_data), "aug_data: Wrong length")

        # Scale the data
        scaled_data, scaler = pp.scale_data(aug_data)
        self.assertEqual(292, len(scaled_data), "scaled_data: Wrong length")
        self.assertIsNotNone(scaler, "scaler: Wrong length")

        # Prepare the scaled data for model inputs
        X, y = pp.prep_model_inputs(scaled_data, features)
        # X and Y are shortened by the number backcandles used in the model
        self.assertEqual(292 - pp.back_candles, len(X), "X: Wrong length")
        self.assertEqual(292 - pp.back_candles, len(y), "y: Wrong length")
        # Delete the model file that we crated and loaded

        if os.path.isfile(model_path):
            os.remove(model_path)
        # End of the prep_load_model_inputs test
        # =========================================

    def test_train_model(self):
        # Create an instance of the price prediction object
        pp = PricePredict(model_dir='../models/', chart_dir='../charts/', preds_dir='../predictions/')

        # Set the ticker and date range (what we want to pull from yahoo finance)
        ticker = "TSLA"
        start_date = "2020-01-01"
        end_date = "2023-12-31"

        # Load data from Yahoo Finance
        data, features = pp.fetch_data_yahoo(ticker, start_date, end_date)
        self.assertGreaterEqual(len(data), 1, "data: Wrong length")
        aug_data, features, targets, dates_data = pp.augment_data(data, features)
        self.assertEqual(features, 19, "features: Wrong count")
        self.assertEqual(1006, len(dates_data), "dates_data: Wrong length")
        self.assertEqual(1007, len(aug_data), "aug_data: Wrong length")

        # Scale the data
        scaled_data, scaler = pp.scale_data(aug_data)
        self.assertEqual(1007, len(scaled_data), "scaled_data: Wrong length")
        self.assertIsNotNone(scaler, "scaler: Wrong length")

        # Prepare the scaled data for model inputs
        X, y = pp.prep_model_inputs(scaled_data, features)
        self.assertEqual(1007 - pp.back_candles, len(X), "X: Wrong length")
        self.assertEqual(1007 - pp.back_candles, len(y), "y: Wrong length")

        # Train the model
        # Use a small batch size and epochs to test the model training
        pp.epochs = 10
        pp.batch_size = 5
        model, y_pred, mse = pp.train_model(X, y)
        pcnt_nan = (len(y_pred) - np.count_nonzero(~np.isnan(y_pred))) / len(y_pred)
        self.assertGreater(.8, pcnt_nan, f"y_pred: Most values are NaN [{pcnt_nan * 100}%]")
        self.assertIsNotNone(model, "model: is None")
        save_op = getattr(model, 'save', None)
        self.assertTrue(callable(save_op), "model: 'save' method not found")
        self.assertIsNotNone(y_pred, "y_pred: is None")
        self.assertEqual(199, len(y_pred), "y_pred: Wrong length")
        self.assertEqual(pp.PeriodDaily, pp.period, f"period[{pp.period}]: Wrong period")

        # View the test prediction results
        time = datetime.now()
        time_str = time.strftime('%Y-%m-%d %H:%M:%S')
        title = f"test_train_model: {ticker} -- Period {pp.period} {time_str}"
        close = pp.data_scaled[pp.split_limit + pp.back_candles:, 4]
        pred_close = pp.pred[:, 1]
        pred_high = pp.pred[:, 2]
        pred_low = pp.pred[:, 3]
        pp.plot_pred_results(close, None, None,
                             pred_close, pred_high, pred_low, title=title)

        # ==============================================================
        # Test a dataset with a Weekly period
        # ==============================================================

        # Create an instance of the price prediction object
        pp = PricePredict(period='W', model_dir='../models/', chart_dir='../charts/', preds_dir='../predictions/')

        # Set the ticker and date range (what we want to pull from yahoo finance)
        ticker = "TSLA"
        start_date = "2015-01-01"
        end_date = "2023-12-31"

        # Load data from Yahoo Finance
        data, features = pp.fetch_data_yahoo(ticker, start_date, end_date)
        self.assertGreaterEqual(len(data), 1, "data: Wrong length")
        aug_data, features, targets, dates_data = pp.augment_data(data, features)
        self.assertEqual(features, 19, "features: Wrong count")
        self.assertEqual(470, len(dates_data), "dates_data: Wrong length")
        self.assertEqual(471, len(aug_data), "aug_data: Wrong length")

        # Scale the data
        scaled_data, scaler = pp.scale_data(aug_data)
        self.assertEqual(471, len(scaled_data), "scaled_data: Wrong length")
        self.assertIsNotNone(scaler, "scaler: Wrong length")

        # Prepare the scaled data for model inputs
        X, y = pp.prep_model_inputs(scaled_data, features)
        self.assertEqual(456, len(X), "X: Wrong length")
        self.assertEqual(456, len(y), "y: Wrong length")

        # Train the model
        # Use a small batch size and epochs to test the model training
        pp.epochs = 10
        pp.batch_size = 5
        model, y_pred, mse = pp.train_model(X, y)
        pcnt_nan = (len(y_pred) - np.count_nonzero(~np.isnan(y_pred))) / len(y_pred)
        self.assertGreater(.8, pcnt_nan, f"y_pred: Most values are NaN [{pcnt_nan * 100}%]")
        self.assertIsNotNone(model, "model: is None")
        save_op = getattr(model, 'save', None)
        self.assertTrue(callable(save_op), "model: 'save' method not found")
        self.assertIsNotNone(y_pred, "y_pred: is None")
        self.assertEqual(92, len(y_pred), "y_pred: Wrong length")
        self.assertEqual(pp.PeriodWeekly, pp.period, f"period[{pp.period}]: Wrong period")

        # View the test prediction results
        time = datetime.now()
        time_str = time.strftime('%Y-%m-%d %H:%M:%S')
        title = f"test_train_model: {ticker} --  Period {pp.period}  {time_str}"
        close = pp.data_scaled[pp.split_limit + pp.back_candles:, 4]
        pred_close = pp.pred_rescaled[:, 1]
        pred_high = pp.pred_rescaled[:, 2]
        pred_low = pp.pred_rescaled[:, 3]
        pp.plot_pred_results(pp.target_close, pp.target_high, pp.target_low,
                             pred_close, pred_high, pred_low, title=title)

    def test_bayesian_optimization(self):
        # Create an instance of the price prediction object
        pp = PricePredict(model_dir='../models/',
                          chart_dir='../charts/', preds_dir='../predictions/',
                          logger=self.logger)

        # Load data from Yahoo Finance
        ticker = "IBM"
        start_date = "2020-01-01"
        end_date = "2023-12-31"
        data, features = pp.fetch_data_yahoo(ticker, start_date, end_date)
        self.assertGreaterEqual(len(data), 1, "data: Wrong length")

        # Augment the data with additional indicators/features
        aug_data, features, targets, dates_data = pp.augment_data(data, features)
        self.assertEqual(features, 19, "features: Wrong count")
        self.assertEqual(1006, len(dates_data), "dates_data: Wrong length")
        self.assertEqual(1007, len(aug_data), "aug_data: Wrong length")

        # Scale the data so the model can use it more effectively
        scaled_data, scaler = pp.scale_data(aug_data)
        self.assertEqual(1007, len(scaled_data), "scaled_data: Wrong length")
        self.assertIsNotNone(scaler, "scaler: Wrong length")

        # Prepare the scaled data for model inputs
        X, y = pp.prep_model_inputs(scaled_data, features)
        self.assertEqual(1007 - pp.back_candles, len(X), "X: Wrong length")
        self.assertEqual(1007 - pp.back_candles, len(y), "y: Wrong length")

        # Use a small batch size and epochs to test the model training
        pp.epochs = 3
        pp.batch_size = 1
        # Train the model
        model, y_pred, mse = pp.train_model(X, y)
        self.assertIsNotNone(model, "model: Wrong length")

        # Perform Bayesian optimization
        pp.bayesian_optimization(X, y)

    def test_save_model(self):
        # Create an instance of the price prediction object
        pp = PricePredict(model_dir='../models/', chart_dir='../charts/', preds_dir='../predictions/')

        # Load data from Yahoo Finance
        ticker = "AAPL"
        test_ticker = "Test-" + ticker
        start_date = "2020-01-01"
        end_date = "2023-12-31"
        data, features = pp.fetch_data_yahoo(ticker, start_date, end_date)
        self.assertGreaterEqual(len(data), 1, "data: Wrong length")

        # Augment the data with additional indicators/features
        aug_data, features, targets, dates_data = pp.augment_data(data, features)
        self.assertEqual(features, 19, "features: Wrong count")
        self.assertEqual(1006, len(dates_data), "dates_data: Wrong length")
        self.assertEqual(1007, len(aug_data), "aug_data: Wrong length")

        # Scale the data so the model can use it more effectively
        scaled_data, scaler = pp.scale_data(aug_data)
        self.assertEqual(1007, len(scaled_data), "scaled_data: Wrong length")
        self.assertIsNotNone(scaler, "scaler: Wrong length")

        # Prepare the scaled data for model inputs
        X, y = pp.prep_model_inputs(scaled_data, features)
        self.assertEqual(1007 - pp.back_candles, len(X), "X: Wrong length")
        self.assertEqual(1007 - pp.back_candles, len(y), "y: Wrong length")

        # Use a small batch size and epochs to test the model training
        pp.epochs = 3
        pp.batch_size = 1
        # Train the model
        model, y_pred, mse = pp.train_model(X, y)
        self.assertIsNotNone(model, "model: Wrong length")

        # Save the model
        model, model_path = pp.save_model(ticker=test_ticker)
        # Checks to verify that we have a savable model object
        self.assertIsNotNone(model, "model: Is None")
        save_op = getattr(model, 'save', None)
        self.assertTrue(callable(save_op), "model: 'save' method not found")
        # Verify that the _Test_ model file was created
        self.assertTrue(os.path.isfile(model_path), "model_path: File does not exist")
        if os.path.isfile(model_path):
            os.remove(model_path)

    def test_predict_price(self):
        # Create an instance of the price prediction object
        pp = PricePredict(model_dir='../models/', chart_dir='../charts/', preds_dir='../predictions/')

        # Set the ticker and date range (what we want to pull from yahoo finance)
        ticker = "AAPL"

        start_date = "2020-01-01"
        end_date = "2023-12-31"
        data, features = pp.fetch_data_yahoo(ticker, start_date, end_date)
        self.assertGreaterEqual(len(data), 1, "data: Wrong length")

        # Augment the data with additional indicators/features
        aug_data, features, targets, dates_data = pp.augment_data(data, features)
        self.assertEqual(features, 19, "features: Wrong count")
        self.assertEqual(1006, len(dates_data), "dates_data: Wrong length")
        self.assertEqual(1007, len(aug_data), "aug_data: Wrong length")

        # Scale the augmented data
        scaled_data, scaler = pp.scale_data(aug_data)
        self.assertEqual(1007, len(scaled_data), "scaled_data: Wrong length")
        self.assertIsNotNone(scaler, "scaler: is None")

        # Prepare the scaled data for model inputs
        X, y = pp.prep_model_inputs(scaled_data, features)
        self.assertEqual(1007 - pp.back_candles, len(X), "X: Wrong length")
        self.assertEqual(1007 - pp.back_candles, len(y), "y: Wrong length")

        # Use a small batch size and epochs to test the model training
        pp.epochs = 10
        pp.batch_size = 5
        # Train the model
        model, y_pred, mse = pp.train_model(X, y)
        self.assertIsNotNone(model, "model: is None")
        y_pred = pp.predict_price(pp.X_test)
        # View the test prediction results
        time = datetime.now()
        time_str = time.strftime('%Y-%m-%d %H:%M:%S')
        title = f"test_predict_price: {ticker} -- Period {pp.period} {time_str}"
        test_close = y[pp.split_limit:, 1]  # Close price
        pred_close = y_pred[:, 1]  # Predicted close price
        pp.plot_pred_results(test_close[-len(pred_close):-2], pred_close, None,
                            None, None, None,
                             title=title)

        ticker = "TSLA"
        pp.period = PricePredict.Period1min
        days_to_load = int(3000 / PricePredict.PeriodMultiplier[pp.period])
        # Set the start date to 7 days ag
        dt_start_date = datetime.strptime('2024-10-20', "%Y-%m-%d")
        start_date = (dt_start_date - timedelta(days=days_to_load)).strftime("%Y-%m-%d")
        # Set the end date to today
        end_date = dt_start_date.strftime("%Y-%m-%d")
        data, features = pp.fetch_data_yahoo(ticker, start_date, end_date)
        self.assertGreaterEqual(len(data), 1000, "data: Wrong length")

        # Augment the data with additional indicators/features
        aug_data, features, targets, dates_data = pp.augment_data(data, features)
        self.assertEqual(features, 19, "features: Wrong count")
        self.assertEqual(1950, len(dates_data), "dates_data: Wrong length")
        self.assertEqual(1951, len(aug_data), "aug_data: Wrong length")

        # Scale the augmented data
        scaled_data, scaler = pp.scale_data(aug_data)
        self.assertEqual(1951, len(scaled_data), "scaled_data: Wrong length")
        self.assertIsNotNone(scaler, "scaler: is None")

        # Prepare the scaled data for model inputs
        X, y = pp.prep_model_inputs(scaled_data, features)
        self.assertEqual(1951 - pp.back_candles, len(X), "X: Wrong length")
        self.assertEqual(1951 - pp.back_candles, len(y), "y: Wrong length")

        # Use a small batch size and epochs to test the model training
        pp.epochs = 10
        pp.batch_size = 5
        # Train the model
        model, y_pred, mse = pp.train_model(X, y)
        self.assertIsNotNone(model, "model: is None")
        y_pred = pp.predict_price(pp.X_test)
        # View the test prediction results
        time = datetime.now()
        time_str = time.strftime('%Y-%m-%d %H:%M:%S')
        title = f"test_predict_price: {ticker} -- Period {pp.period} {time_str}"
        test_close = y[pp.split_limit:, 1]  # Close price
        pred_close = y_pred[:, 1]  # Predicted close price
        pp.plot_pred_results(test_close[-len(pred_close):-2], pred_close, None,
                            None, None, None,
                             title=title)

    def test_adjust_prediction(self):
        pp = PricePredict(model_dir='../models/', chart_dir='../charts/', preds_dir='../predictions/')
        pp.verbose = True

        # Set the ticker and date range (what we want to pull from yahoo finance)
        ticker = "AAPL"
        start_date = "2020-01-01"
        end_date = "2023-12-31"
        # Load data from Yahoo Finance
        data, features = pp.fetch_data_yahoo("AAPL", start_date, end_date)
        self.assertGreaterEqual(len(data), 1, "data: Wrong length")

        # Augment the data
        aug_data, features, targets, dates_data = pp.augment_data(data, features)
        self.assertEqual(features, 19, "features: Wrong count")
        self.assertEqual(1006, len(dates_data), "dates_data: Wrong length")
        self.assertEqual(1007, len(aug_data), "aug_data: Wrong length")

        # Scale the data
        scaled_data, scaler = pp.scale_data(aug_data)
        self.assertEqual(1007, len(scaled_data), "scaled_data: Wrong length")
        self.assertIsNotNone(scaler, "scaler: Wrong length")

        # Prepare the scaled data for model inputs
        X, y = pp.prep_model_inputs(scaled_data, features)
        self.assertEqual(1007 - pp.back_candles, len(X), "X: Wrong length")
        self.assertEqual(1007 - pp.back_candles, len(y), "y: Wrong length")

        # Use a small batch size and epochs to test the model training
        pp.epochs = 10
        pp.batch_size = 5

        # Train the model
        model, y_pred, mse = pp.train_model(X, y)
        self.assertIsNotNone(model, "model: Wrong length")

        # Predict some prices
        X_test = pp.X_test
        y_pred = pp.predict_price(X_test)
        y_test = pp.y_test
        y_len = y_test.shape[0]

        # Adjust the prediction (data alignment)
        adj_pred_close, adj_pred_high, adj_pred_low = pp.adjust_prediction()

        # View the prediction results
        time = datetime.now()
        time_str = time.strftime('%Y-%m-%d %H:%M:%S')
        title = f"test_adjust_prediction: {ticker} -- Period {pp.period} {time_str}"
        pp.plot_pred_results(
            pp.target_close, pp.target_high, pp.target_low,
            pp.adj_pred_close, pp.adj_pred_high, pp.adj_pred_low, title=title)

    def test_gen_prediction_chart(self):
        # Create an instance of the price prediction object
        pp = PricePredict(model_dir='../models/', chart_dir='../charts/', preds_dir='../predictions/')

        # =========================================
        # First we need to create a model file
        ticker = "AMZN"
        test_ticker = "Test-" + ticker
        # Data download dates
        start_date = "2023-06-01"
        end_date = "2024-07-30"
        # Model file dates
        mdl_start_date = "2015-01-01"
        mdl_end_date = "2024-07-30"

        model_path = Test_PricePredict._create_test_model(self, pp, ticker, test_ticker, mdl_start_date, mdl_end_date)
        pp.seasonality()
        # Create a Model that can be loaded...
        # =========================================

        # =========================================
        # Set the ticker and date range (what we want to pull from yahoo finance)
        # Load and use the model
        adj_pred_rescaled, pred_dates = pp.fetch_and_predict(model_path=model_path,
                                                             date_start=mdl_start_date, date_end=mdl_end_date)
        success = False
        if adj_pred_rescaled is not None and pred_dates is not None:
            success = True
        self.assertTrue(success, "train_and_test: Success should be True")

        # self.logger.info("Current Dir: ", os.getcwd())

        # Save the prediction data
        file_path = pp.gen_prediction_chart(last_candles=40, save_plot=True)
        self.assertTrue(os.path.isfile(file_path), f"gen_prediction_chart: File does not exist [{file_path}]")
        if os.path.isfile(file_path):
            os.remove(file_path)

        # Delete the model file that we crated and loaded
        if os.path.isfile(model_path):
            os.remove(model_path)
        # End of the load_model test
        # =========================================

    def test_fetch_train_and_predict(self):
        # Create an instance of the price prediction object
        pp = PricePredict(model_dir='../models/', chart_dir='../charts/', preds_dir='../predictions/')

        # Set the ticker and date range (what we want to pull from yahoo finance)
        ticker = "AAPL"
        train_start_date = "2020-01-01"
        train_end_date = "2023-12-31"
        pred_start_date = "2024-01-01"
        pred_end_date = "2024-02-29"


        # Train and test the model
        model = pp.fetch_train_and_predict(ticker,
                                           train_start_date, train_end_date,
                                           pred_start_date, pred_end_date)
        self.assertIsNotNone(model, "train_and_test: Returned None")

        # View the test prediction results
        time = datetime.now()
        time_str = time.strftime('%Y-%m-%d %H:%M:%S')
        title = f"test_train_model: {ticker} -- Period {pp.period} {time_str}"
        pp.plot_pred_results(pp.target_close[-len(pp.adj_pred_close)+2:], None, None,
                             pp.adj_pred_close, pp.adj_pred_high, pp.adj_pred_low,
                             title=title)

    def test_fetch_and_predict(self):
        # Create an instance of the price prediction object
        pp = PricePredict(model_dir='../models/', chart_dir='../charts/', preds_dir='../predictions/')

        # =========================================
        # First we need to create a model file
        ticker = "IBM"
        test_ticker = "Test-" + ticker
        # Data download dates
        start_date = "2023-06-01"
        end_date = "2024-07-30"
        # Model file dates
        mdl_start_date = "2015-01-01"
        mdl_end_date = "2024-07-30"

        model_path = Test_PricePredict._create_test_model(self, pp, ticker, test_ticker, start_date, end_date)
        # Create a Model that can be loaded...
        # =========================================

        # =========================================
        # Load an existing model file
        # Set the ticker and date range (what we want to pull from yahoo finance)

        # Load and use the model
        adj_pred_rescaled, pred_dates = pp.fetch_and_predict(model_path=model_path, date_start=start_date,
                                                             date_end=end_date)
        if adj_pred_rescaled is not None and pred_dates is not None:
            success = True
        self.assertTrue(success, "train_and_test: Success should be True")
        self.assertEqual(PricePredict.PeriodDaily, pp.period, f"period[{pp.period}]: Wrong period")
        adj_pred_close = adj_pred_rescaled[:, 1]
        adj_pred_high = adj_pred_rescaled[:, 2]
        adj_pred_low = adj_pred_rescaled[:, 3]

        # View the test prediction results
        time = datetime.now()
        time_str = time.strftime('%Y-%m-%d %H:%M:%S')
        title = f"test_train_model: {ticker} -- Period {pp.period} {time_str}"
        pp.plot_pred_results(pp.target_close, pp.pred_close, pp.adj_pred_close,
                             adj_pred_close, adj_pred_high, adj_pred_low,
                             title=title)

        # Delete the model file that we crated and loaded
        if os.path.isfile(model_path):
            os.remove(model_path)
        # End of the fetch_and_predict test
        # =========================================

    def test_prediction_analysis(self):
        # Create an instance of the price prediction object
        pp = PricePredict(model_dir='../models/', chart_dir='../charts/', preds_dir='../predictions/')

        # =========================================
        # Test the case where there is no prediction data
        success = pp.prediction_analysis()
        # There is no prediction data, so this should fail
        self.assertFalse(success, "prediction_report: Success should be False")
        # =========================================

        # =========================================
        # First we need to create a model file
        ticker = "WMT"
        test_ticker = "Test-" + ticker
        # Data download dates
        start_date = "2023-06-01"
        end_date = "2024-07-30"
        # Model file dates
        mdl_start_date = "2015-01-01"
        mdl_end_date = "2024-07-30"

        model_path = Test_PricePredict._create_test_model(self, pp, ticker, test_ticker, start_date, end_date)
        # Create a Model that can be loaded...
        # =========================================

        # =========================================
        # Load an existing model file
        # Load and predict so that we have some prediction data
        adj_predictions, pred_dates = pp.fetch_and_predict(model_path, start_date, end_date)
        if adj_predictions is not None and pred_dates is not None:
            success = True
        self.assertTrue(success, "fetch_and_predict: Success should be True")
        self.logger.info("Predictions Successfully performed.")

        pp.seasonality()

        # Try creating a prediction report again
        success = pp.prediction_analysis()
        self.assertTrue(success, "prediction_report: Success should be True")

        # View the test prediction results
        time = datetime.now()
        time_str = time.strftime('%Y-%m-%d %H:%M:%S')
        title = f"test_train_model: {ticker} -- Period {pp.period} {time_str}"
        pp.plot_pred_results(pp.target_close, pp.target_high, pp.target_low,
                    pp.adj_pred_close, pp.adj_pred_high, pp.adj_pred_low,
                    title=title)

        # Delete the model file that we crated and loaded
        if os.path.isfile(model_path):
            os.remove(model_path)
        # End of the prediction_analysis test
        # =========================================

    def test_save_prediction_data(self):
        # Create an instance of the price prediction object
        pp = PricePredict(model_dir='../models/', chart_dir='../charts/', preds_dir='../predictions/')

        # Set the ticker and date range (what we want to pull from yahoo finance)
        ticker = "AAPL"
        test_ticker = "Test-" + ticker
        # Data download dates
        start_date = "2024-06-01"
        end_date = "2024-07-30"
        # Model file dates
        mdl_start_date = "2015-01-01"
        mdl_end_date = "2024-07-30"

        # Train and test the model
        model = pp.fetch_train_and_predict(ticker,
                                           mdl_start_date, mdl_end_date,
                                           start_date, end_date)
        self.assertIsNotNone(model, "train_and_test: Returned None")
        if pp.adj_pred is not None and pp.adj_pred_close is not None:
            success = True
        self.assertTrue(success, "train_and_test: Success should be True")

        # Save the prediction data
        pp.ticker = test_ticker
        file_paths = pp.save_prediction_data()
        for file_path in file_paths:
            self.assertTrue(os.path.isfile(file_path), f"save_prediction_data: File does not exist [{file_path}]")
            # Add code that validates the CSV data in the file_path

            # Delete the model file that we crated and loaded
            if os.path.isfile(file_path):
                os.remove(file_path)

    def test_model_report(self):
        # Create an instance of the price prediction object
        pp = PricePredict(model_dir='../models/', chart_dir='../charts/', preds_dir='../predictions/')

        # =========================================
        # Perform a model report test...
        # Should return False if there is no model loaded...
        success = pp.model_report()
        self.assertFalse(success, "model_report: Success should be False")
        # =========================================


        # =========================================
        # First we need to create a model file
        ticker = "NKE"
        test_ticker = "Test-" + ticker
        # Data download dates
        start_date = "2024-06-01"
        end_date = "2024-07-30"
        # Model file dates
        mdl_start_date = "2015-01-01"
        mdl_end_date = "2024-07-30"

        model_path = Test_PricePredict._create_test_model(self, pp, ticker, test_ticker, start_date, end_date)
        # Create a Model that can be loaded...
        # =========================================

        # =========================================
        # Perform a model report test...
        # We have loaded a model file
        # Load model *args
        # Load the model via *args
        model = pp.load_model(model_path)
        self.assertIsNotNone(model, "Model not loaded")
        save_op = getattr(model, 'save', None)
        self.assertTrue(callable(save_op), "model: 'save' method not found")

        success = success = pp.model_report()
        self.assertTrue(success, "model_report: Success should be True")

        # Delete the model file that we crated and loaded
        if os.path.isfile(model_path):
            os.remove(model_path)
        # End of the load_model test
        # =========================================

    def test_periodic_correlation(self):
        # Create an instance of the price prediction object
        pp1 = PricePredict(period=PricePredict.PeriodDaily, model_dir='../models/', chart_dir='../charts/', preds_dir='../predictions/')
        pp2 = PricePredict(period=PricePredict.PeriodDaily, model_dir='../models/', chart_dir='../charts/', preds_dir='../predictions/')

        # Load data from Yahoo Finance
        ticker1 = "AAPL"
        ticker2 = "TSLA"
        start_date = "2020-01-01"
        end_date = "2023-12-31"
        data1, features1 = pp1.fetch_data_yahoo(ticker1, start_date, end_date)
        self.assertGreaterEqual(1006, len(data1), "data1: Wrong length")
        data2, features2 = pp2.fetch_data_yahoo(ticker2, start_date, end_date)
        self.assertGreaterEqual(1006, len(data2), "data2: Wrong length")

        # Perform the correlation analysis
        ret_dict = pp1.periodic_correlation(pp2)
        self.assertIsNotNone(ret_dict, "periodic_correlation: Returned None")
        self.assertEqual(PricePredict.PeriodDaily, pp1.period, f"period[{pp1.period}]: Wrong period")
        self.assertEqual(PricePredict.PeriodDaily, pp2.period, f"pp2.period[{pp2.period}]: Wrong period")
        exp_dict = {'total_days': 1004,
                    'correlated_days': 675,
                    'uncorrelated_days': 329,
                    'pct_corr': 0.6723107569721115,
                    'pct_uncorr': 0.32768924302788843,
                    'pearson_corr': 0.34496343125631956,
                    'spearman_corr': 0.34496343125631995,
                    'kendall_corr': 0.34496343125632,
                    'avg_corr': 0.3449634312563199
                    }
        self.assertEqual(exp_dict, ret_dict, f"exp_dict[{exp_dict}] does not match ret_dict[{ret_dict}]")

        # Perform the correlation analysis for the last 50 days
        ret_dict = pp1.periodic_correlation(pp2, pc_period_len=50)
        self.assertIsNotNone(ret_dict, "periodic_correlation: Returned None")
        self.assertEqual(PricePredict.PeriodDaily, pp1.period, f"period[{pp1.period}]: Wrong period")
        self.assertEqual(PricePredict.PeriodDaily, pp2.period, f"pp2.period[{pp2.period}]: Wrong period")
        exp_dict = {'total_days': 49,
                    'correlated_days': 33,
                    'uncorrelated_days': 16,
                    'pct_corr': 0.673469387755102,
                    'pct_uncorr': 0.32653061224489793,
                    'pearson_corr': 0.34233931937116724,
                    'spearman_corr': 0.3423393193711671,
                    'kendall_corr': 0.34233931937116713,
                    'avg_corr': 0.3423393193711671
                    }
        self.assertEqual(exp_dict, ret_dict, f"exp_dict[{exp_dict}] does not match ret_dict[{ret_dict}]")

        # Perform the correlation analysis for the last 7 days
        ret_dict = pp1.periodic_correlation(pp2, pc_period_len=7)
        self.assertIsNotNone(ret_dict, "periodic_correlation: Returned None")
        self.assertEqual(PricePredict.PeriodDaily, pp1.period, f"period[{pp1.period}]: Wrong period")
        self.assertEqual(PricePredict.PeriodDaily, pp2.period, f"pp2.period[{pp2.period}]: Wrong period")
        exp_dict = {'total_days': 6,
                    'correlated_days': 4,
                    'uncorrelated_days': 2,
                    'pct_corr': 0.6666666666666666,
                    'pct_uncorr': 0.3333333333333333,
                    'pearson_corr': 0.4472135954999579,
                    'spearman_corr': 0.4472135954999579,
                    'kendall_corr': 0.4472135954999579,
                    'avg_corr': 0.4472135954999579
                    }
        self.assertEqual(exp_dict, ret_dict, f"exp_dict[{exp_dict}] does not match ret_dict[{ret_dict}]")

    def test_seasonality(self):
        # Create an instance of the price prediction object
        pp = PricePredict(period=PricePredict.PeriodDaily, model_dir='../models/', chart_dir='../charts/', preds_dir='../predictions/')

        # =========================================
        # First we need to create a model file
        ticker = "^IXIC"
        test_ticker = "Test-" + ticker
        # Data download dates
        start_date = "2015-01-01"
        end_date = "2024-07-30"

        model_path = Test_PricePredict._create_test_model(self, pp, ticker, test_ticker, start_date, end_date)
        # Create a Model that can be loaded...
        # =========================================

        # =========================================
        # Load data from Yahoo Finance
        data1, features1 = pp.fetch_data_yahoo(ticker, start_date, end_date)
        self.assertGreaterEqual(2408, len(data1), "data1: Wrong length")

        # Perform the seasonality analysis
        seasonal_dec = pp.seasonality(sd_period_len=30)
        self.assertIsNotNone(seasonal_dec, "seasonality: Returned None")
        self.assertEqual(2409, seasonal_dec.observed.shape[0], f"period[{pp.period}]: Wrong period")

        seasonal_dec = pp.seasonality(sd_period_len=30, save_chart=True,
                                       show_chart=True)
        self.assertIsNotNone(seasonal_dec, "seasonality: Returned None")
        self.assertEqual(2409, seasonal_dec.observed.shape[0], f"period[{pp.period}]: Wrong period")
        self.assertTrue(os.path.isfile(pp.seasonal_chart_path), f"pp.graph_path: File does not exist [{pp.period}]")
        if os.path.isfile(pp.seasonal_chart_path):
            os.remove(pp.seasonal_chart_path)

        # Delete the model file that we crated and loaded
        if os.path.isfile(model_path):
            os.remove(model_path)
        # End of the seasonality test
        # =========================================

    def test_groq_sentiment(self):
        # Create an instance of the price prediction object
        pp = PricePredict(period=PricePredict.PeriodDaily, model_dir='../models/', chart_dir='../charts/', preds_dir='../predictions/')

        # =========================================
        # First we need to create a model file
        ticker = "AAPL"
        test_ticker = "Test-" + ticker
        # Data download dates
        start_date = "2024-06-01"
        end_date = "2024-07-30"
        # Model file dates
        mdl_start_date = "2015-01-01"
        mdl_end_date = "2024-07-30"

        pp.ticker = ticker
        pp.groq_sentiment()
        self.assertIsNotNone(pp.sentiment_json, "groq_sentiment: Returned None")
        self.assertIsNotNone(pp.sentiment_json['balance_sheet_analysis']['cash_and_cash_equivalents'],
                             "groq_sentiment ['balance_sheet_analysis']['cash_and_cash_equivalents']: Returned None")
