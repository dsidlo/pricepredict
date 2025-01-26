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


def get_ppo(symbol: str, period: str, bypass_cache=False, file_offset=0):
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


def bayes_search():

    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)

    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)

    # Create an instance of the price prediction object
    pp = PricePredict(model_dir='./models/', chart_dir='./charts/', preds_dir='./predictions/', ppo_dir='./ppo/',
                      verbose=False, logger=logger, log_level=logging.ERROR,
                      keras_verbosity=1, tf_logs_dir=log_dir,
                      keras_callbacks=[tensorboard_callback], tf_profiler=True)

    # Load data from Yahoo Finance
    ticker = "IBM"
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365 * 5)).strftime('%Y-%m-%d')
    # Load and prep data from Yahoo Finance
    pp.ticker = ticker
    X, y = pp._fetch_n_prep(pp.ticker, start_date, end_date, train_split=0.8)
    # Train the model
    model, y_pred, mse = pp.train_model(X, y)

    # Perform Bayesian optimization
    # - Test with all Parameters and 1 2 and 3 hidden layers
    pp.model = None
    pp.bayes_opt_hypers = None
    pp.bayesian_optimization(X, y,
                             # opt_max_init=100,
                             # opt_max_iter=100,
                             opt_max_init=50,
                             opt_max_iter=50,
                             pb_lstm_units=(220, 260),
                             pb_lstm_dropout=(0.1, 0.2),
                             pb_adam_learning_rate=(0.005, 0.008),
                             pb_epochs=(200, 350),
                             pb_batch_size=(700, 900),
                             pb_hidden_layers=(3, 3),
                             pb_hidden_layer_units_1=(50, 70),
                             pb_hidden_layer_units_2=(32, 256),
                             pb_hidden_layer_units_3=(64, 256),
                             pb_hidden_layer_units_4=(128, 256))

    # Save the pp object
    file_path = pp.store_me()
    # Output the pp .zill file
    print(f"Saved the pp object to: {file_path}")
    # Output the best hyperparameters
    print(f"Best Prediction Hyperparameters:")
    pprint(pp.bayes_best_pred_hp)
    # Output the prediction analysis
    print(f"Best Prediction Analysis:")
    pprint(pp.bayes_best_pred_hp)
    # Plot the prediction chart
    pp.gen_prediction_chart()


if __name__ == "__main__":

    run_bayes_search = True

    if run_bayes_search:
        bayes_search()
    else:
        ppo_file, ppo = get_ppo('IBM', 'D', bypass_cache=True, file_offset=0)
        file, fig = ppo.gen_prediction_chart(show_plot=True)
        mpf.show()

