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
from pprint import pprint
from datetime import datetime, timedelta
from pricepredict import PricePredict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

def bayes_search():

    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)

    # Create an instance of the price prediction object
    pp = PricePredict(model_dir='./models/', chart_dir='./charts/', preds_dir='./predictions/', ppo_dir='./ppo/',
                      verbose=False, logger=logger, log_level=logging.ERROR,
                      keras_verbosity=1)

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
                             opt_max_init=50,
                             opt_max_iter=20,
                             pb_lstm_units=(220, 260),
                             pb_lstm_dropout=(0.1, 0.2),
                             pb_adam_learning_rate=(0.005, 0.008),
                             pb_epochs=(200, 350),
                             pb_batch_size=(700, 900),
                             pb_hidden_layers=(1, 2),
                             pb_hidden_layer_units_1=(50, 70),
                             pb_hidden_layer_units_2=(32, 256),
                             pb_hidden_layer_units_3=(64, 256),
                             pb_hidden_layer_units_4=(128, 256))

    # Save the pp object
    file_path = pp.store_me()
    # Output the pp .zill file
    print(f"Saved the pp object to: {file_path}")
    # Output the best hyperparameters
    print(f"Best Hyperparameters:")
    pprint(pp.bayes_opt_hypers)
    # Output the prediction analysis
    print(f"Prediction Analysis:")
    pprint(pp.analysis)

    time = datetime.now()
    time_str = time.strftime('%Y-%m-%d %H:%M:%S')
    title = f"test_bayes_opt all params: {ticker} --  Period {pp.period}  {time_str}"
    pp.plot_pred_results(pp.target_close[2:], pp.target_high[2:], pp.target_low[2:],
                         pp.adj_pred[:, 1], pp.adj_pred[:, 2], pp.adj_pred[:, 3], title=title)



if __name__ == "__main__":
    bayes_search()
