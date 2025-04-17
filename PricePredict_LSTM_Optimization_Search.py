"""
file: PricePredict_LSTM_Optimization_Search.py

Use Bayesian Optimization to search for the optimal hyperparameters for the LSTM model
- Find the optical range for the hyperparameters search

# PricePredict LSTM Optimization

Using bayesian optimization to optimize the hyperparameters of the LSTM model.
- Do hidden layers improve the model?

- Latest test that results in the best trend prediction...

## Test Parameters
```
pp.bayesian_optimization(X, y,
                         opt_max_init=100,
                         opt_max_iter=20,
                         pb_lstm_units=(32, 256),
                         pb_lstm_dropout=(0.1, 0.5),
                         pb_adam_learning_rate=(0.001, 0.1),
                         pb_epochs=(100, 300),
                         pb_batch_size=(1, 1024),
                         pb_hidden_layers=(1, 4),
                         pb_hidden_layer_units_1=(16, 256),
                         pb_hidden_layer_units_2=(32, 256),
                         pb_hidden_layer_units_3=(64, 256),
                         pb_hidden_layer_units_4=(128, 256))
 ```

## Best Hyperparameters
```
{'adam_learning_rate': 0.007133047958571564,
 'batch_size': 778.4103564949222,
 'epochs': 207.8240175819626,
 'hidden_layer_units_1': 59.5543770288849,
 'hidden_layer_units_2': 33.32458485892822,
 'hidden_layer_units_3': 74.53889954301027,
 'hidden_layer_units_4': 186.98073315100643,
 'hidden_layers': 1.2787028637916622,
 'lstm_dropout': 0.1277210415797052,
 'lstm_units': 240.4579822805544
}
```

## Good Trend Prediction 62% tren prediction
```
{'actual_vs_pred': {'mean_trading_range': 0.0,
                    'mean_delta': 0.0,
                    'corr_day_cnt': 639,
                    'corr_days': 639,
                    'uncorr_days': 387,
                    'pct_corr': 62.2807,
                    'pct_uncorr': 37.7193},
 'actual_vs_adj_pred': {'mean_trading_range': 2.07,
                        'mean_delta': 1.42,
                        'corr_days': 639,
                        'uncorr_days': 387,
                        'pct_corr': 62.2807,
                        'pct_uncorr': 37.7193}
}
```
"""

import logging
import os
import berkeleydb as bdb
import mplfinance as mpf
import tensorflow as tf
import keras

from pprint import pprint
from datetime import datetime, timedelta
from pricepredict import PricePredict
from tensorflow.keras.callbacks import TensorBoard

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

# Use an Object Cache to reduce the prep time for creating and loading the PricePredict objects.
if 'ObjCache' not in globals():
    global ObjCache
    ObjCache = bdb.btopen('ppo_cache.db', 'c')

DirPPO = './ppo/'


def get_ppo(symbol: str, period: str,
            bypass_cache=False, file_offset=0):

    global ObjCache

    # print(f'Type of ObjCache: {type(ObjCache)}')

    ppo_name = symbol + '_' + period

    if bypass_cache is False:
        if bytes(ppo_name, 'latin1') in ObjCache.keys():
            print(f"Using Cached PPO: {ppo_name}")
            ppo = PricePredict.unserialize(ObjCache[bytes(ppo_name, 'latin1')])
            return 'None', ppo

    file_name_starts_with = symbol + '_' + period
    # Find all PPO files for the symbol in the PPO directory
    ppo_files = [f for f in os.listdir(DirPPO) if f.startswith(file_name_starts_with) and f.endswith('.dilz')]
    ppo = None
    if len(ppo_files) > 0:
        # Sort the files by date
        ppo_files.sort()
        print(f"Files Found: {len(ppo_files)}")
        # Get the latest PPO file
        ppo_file = ppo_files[-(1 + file_offset)]
        # Unpickle the PPO file using dilz
        print(f"Reading PPO File: {ppo_file}")
        with open(DirPPO + ppo_file, 'rb') as f:
            f_obj = f.read()
            ppo = PricePredict.unserialize(f_obj)

    if ppo is None:
        ppo_file = ppo_name
        print(f"Creating PPO: {ppo_file}")
        ppo = PricePredict(symbol,
                           model_dir='../models/',
                           chart_dir='../charts/',
                           preds_dir='../predictions/',
                           period=period)
        # Train the models on 5 yeas of data...
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=5 * 400)
        end_date = end_dt.strftime('%Y-%m-%d')
        start_date = start_dt.strftime('%Y-%m-%d')
        ppo.fetch_train_and_predict(ppo.ticker,
                                    start_date, end_date,
                                    start_date, end_date,
                                    period=PricePredict.PeriodWeekly,
                                    force_training=False,
                                    use_curr_model=True,
                                    save_model=False)

    # Cache the ppo
    ObjCache[bytes(ppo_name, 'latin1')] = ppo.serialize_me()

    return ppo_file, ppo


def bayes_search(ticker: str, period: str,
                 back_candles=1000,
                 train_model=False, find_best_model=False,
                 # --- Hyperparameters For Training ---
                 hp_adam_learning_rate=None,
                 hp_batch_size=None,
                 hp_epochs=None,
                 hp_hidden_layer_units=None,
                 hp_hidden_layers=None,
                 hp_lstm_dropout=None,
                 hp_lstm_units=None,
                 # --- Hyperparameter Ranges for Model Optimization ---
                 hpo_adam_learning_rate: (float, float) = None,
                 hpo_batch_size: (int, int) = None,
                 hpo_epochs: (float, float) = None,
                 hpo_hidden_layer_units: (int, int) = None,
                 hpo_hidden_layers: [(int, int)] = None,
                 hpo_lstm_dropout: (float, float) = None,
                 hpo_lstm_units: (int, int) = None):

    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)

    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)

    # Create an instance of the price prediction object
    pp = PricePredict(ticker, period, back_candles=back_candles,
                      model_dir='./models/', chart_dir='./charts/', preds_dir='./predictions/', ppo_dir='./ppo/',
                      verbose=False, logger=logger, log_level=logging.ERROR,
                      keras_verbosity=1, tf_logs_dir=log_dir,
                      keras_callbacks=[tensorboard_callback], tf_profiler=True)

    # Load data from Yahoo Finance
    weeks = back_candles % 5
    days = back_candles + (weeks + 2)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

    if train_model:
        # Load and prep data from Yahoo Finance
        pp.ticker = ticker
        X, y = pp.fetch_n_prep(pp.ticker, start_date, end_date, train_split=0.8)
        # Train the model
        model, y_pred, mse = pp.train_model(X, y,
                                            epochs=hp_epochs,
                                            adam_learning_rate=hp_adam_learning_rate,
                                            lstm_dropout=hp_lstm_dropout,
                                            lstm_units=hp_lstm_units,
                                            batch_size=hp_batch_size,
                                            hidden_layer_units=hp_hidden_layer_units,
                                            hidden_layers=hp_hidden_layers,
                                            )
        # Now perform the prediction
        pp.predict_price(pp.X)

    if find_best_model:
        # Perform Bayesian optimization
        # - Test with all Parameters and 1 2 and 3 hidden layers
        pp.model = None
        pp.bayes_opt_hypers = None
        pp.ticker = ticker
        X, y = pp.fetch_n_prep(pp.ticker, start_date, end_date, train_split=0.8)
        pp.bayesian_optimization(X, y,
                                 # opt_max_init=100,
                                 # opt_max_iter=100,
                                 opt_max_init=10,
                                 opt_max_iter=100,
                                 pb_lstm_units=hpo_lstm_units,  # (220, 260),
                                 pb_lstm_dropout=hpo_lstm_dropout,  # (0.1, 0.2),
                                 pb_adam_learning_rate=hpo_adam_learning_rate,  # (0.005, 0.008),
                                 pb_epochs=hpo_epochs,  # (200, 350),
                                 pb_batch_size=hpo_batch_size,  # (700, 900),
                                 pb_hidden_layers=hpo_hidden_layers,  # (4, 4),
                                 pb_hidden_layer_units_1=hpo_hidden_layer_units[0],  # (16,16),
                                 pb_hidden_layer_units_2=hpo_hidden_layer_units[1],  # (32, 32),
                                 pb_hidden_layer_units_3=hpo_hidden_layer_units[2],  # (64, 64),
                                 pb_hidden_layer_units_4=hpo_hidden_layer_units[3])  # (128, 128))

    # Save the pp object
    file_path = pp.store_me()
    # Output the pp .zill file
    print(f"Saved the pp object to: {file_path}")
    # Output the best hyperparameters
    print(f"Best Prediction Hyperparameters:")
    pprint(pp.bayes_best_pred_hp)
    # Output the prediction analysis
    print(f"Best Prediction Analysis:")
    pprint(pp.analysis)

    return pp

if __name__ == "__main__":

    # True: Run the Bayesian Hyperparameter Search
    # False: Load the latest saved PPO and generate the prediction chart
    train_model = False
    find_best_model = False
    ticker = '^GSPC'

    if train_model or find_best_model:
        ppo = bayes_search(ticker, 'D', back_candles=2000
                           # --- Hyperparameters For Training ---
                           , train_model=train_model
                           , hp_batch_size=801
                           , hp_epochs=221
                           , hp_lstm_dropout=0.1315779901171808
                           , hp_lstm_units=234
                           , hp_adam_learning_rate=0.007930918606069523
                           , hp_hidden_layer_units=[67, 186]
                           , hp_hidden_layers=2
                           # --- Hyperparameter Ranges for Model Optimization ---
                           , find_best_model=find_best_model
                           , hpo_batch_size=(700, 900)
                           , hpo_epochs=(200, 350)
                           , hpo_adam_learning_rate=(0.005, 0.008)
                           , hpo_lstm_dropout=(0.1, 0.2)
                           , hpo_lstm_units=(256, 256)
                           # (220, 260)
                           , hpo_hidden_layers=(2, 2)
                           # (4,4, None, None) {Must have 4 tuples or None}
                           , hpo_hidden_layer_units=[(1024, 1024), (128, 128), None, None]
                           # [(16, 16), (32, 32), (64, 64), (128, 128)]
                           )
        file, fig = ppo.gen_prediction_chart(save_plot=False,show_plot=True, last_candles=2000)
        mpf.show()
    else:
        # ppo_file, ppo = get_ppo('IBM', 'D', bypass_cache=True, file_offset=0)
        # file, fig = ppo.gen_prediction_chart(save_plot=False,show_plot=True)
        # mpf.show()

        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365 * 10)).strftime('%Y-%m-%d')

        ppo_file, ppo = get_ppo('IBM', 'D', bypass_cache=True, file_offset=0)
        ppo.ticker = '^GSPC'
        ppo.invalidate_cache()
        ppo.fetch_data_yahoo(ppo.ticker, start_date, end_date)
        ppo.fetch_and_predict(ppo.ticker, start_date, end_date)
        file, fig = ppo.gen_prediction_chart(save_plot=False,show_plot=True, last_candles=2000)
        mpf.show()

