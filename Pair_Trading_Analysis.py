import sys
from types import ModuleType, FunctionType
from gc import get_referents

# Helper function to get the size of an object (Curiosity)
# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST = type, ModuleType, FunctionType


def getsize(obj):
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size


import pandas as pd
import statsmodels.api as sm
from decimal import Decimal
from pandas_decimal import DecimaldDtype


def get_trading_pair_spread(ppos: tuple, beta: Decimal = None,
                            prev_days: int = None,
                            start_period: int = None, end_period: int = None,
                            start_date: str = None, end_date: str = None):
    # Create a DataFrame of the closing prices from the PPO[0 and 1].orig_data dataframes
    closes1 = ppos[0].orig_data['Close'].astype(DecimaldDtype(5))
    closes2 = ppos[1].orig_data['Close'].astype(DecimaldDtype(5))
    # Make closes1 and closes2 the same length
    min_len = min(len(closes1), len(closes2))
    if prev_days is None:
        prev_days = min_len
    elif prev_days > min_len:
        prev_days = min_len
    if start_period is not None and end_period is not None:
        # Gather closes based numeric index
        closes1 = closes1[start_period:end_period]
        closes2 = closes2[start_period:end_period]
    elif start_date is not None and end_date is not None:
        # Gather closes based on the date index column
        closes1 = closes1.loc[start_date:end_date]
        closes2 = closes2.loc[start_date:end_date]
    else:
        # Default to the last prev_days
        closes1 = closes1.tail(prev_days)
        closes2 = closes2.tail(prev_days)
    df_closes = pd.DataFrame({'Stock_A': closes1, 'Stock_B': closes2})
    # df_closes.replace([np.inf, -np.inf], None, inplace=True)
    df_closes = df_closes.bfill().ffill()
    try:
        if beta is None:
            # Perform OLS to find beta
            X = df_closes['Stock_B']
            X = sm.add_constant(X)  # Adds a constant term to the predictor
            model = sm.OLS(df_closes['Stock_A'], X).fit()
            beta = model.params['Stock_B']
    except Exception as e:
        print(f"Error: {e}")
        beta = np.float32(1.0)

    # Detrend the closes
    # closes1m = (closes1 - closes1.rolling(window=3)).mean()
    closes1m = closes1.rolling(window=3).apply(lambda x: (x - x.mean()).mean())
    # closes2m = (closes2 - closes2.rolling(window=3)).mean()
    closes2m = closes2.rolling(window=3).apply(lambda x: (x - x.mean()).mean())
    df_detrend = pd.DataFrame({'Stock_A': closes1m, 'Stock_B': closes2m})
    df_detrend = df_detrend.bfill().ffill()
    # Calculate the spread and its mean using the Hedge-Ratio beta
    df_detrend['Spread'] = df_closes['Stock_A'] - beta * df_closes['Stock_B']
    spread_mean = df_detrend['Spread'].mean()
    # Create a line that is 1 standard deviation above from the spread-mean
    df_detrend['Mean_1std_a'] = spread_mean + df_detrend['Spread'].std()
    # Create a line that is 2 standard deviation above from the spread-mean
    df_detrend['Mean_2std_a'] = spread_mean + 2 * df_detrend['Spread'].std()
    # Create a line that is 1 standard deviation below from the spread-mean
    df_detrend['Mean_1std_b'] = spread_mean - df_detrend['Spread'].std()
    # Create a line that is 2 standard deviation below from the spread-mean
    df_detrend['Mean_2std_b'] = spread_mean - 2 * df_detrend['Spread'].std()

    return ppos, df_closes, df_detrend, spread_mean, beta


import matplotlib.pyplot as plt


def show_annotation(sel):
    x, y = sel.target
    ind = sel.index
    sel.annotation.set_text(f'{x:.0f}, {y:.0f}: {labels[ind]}')


def plot_spread(ppos: tuple, beta: Decimal = None,
                prev_days: int = None,
                title: str = None,
                spread_name: str = 'Spread',
                spread_color: str = 'black',
                start_period: int = None, end_period: int = None,
                start_date: str = None, end_date: str = None):
    ppos, df_closes, df_detrend, spread_mean, beta = get_trading_pair_spread(ppos, beta,
                                                                             prev_days,
                                                                             start_period, end_period,
                                                                             start_date, end_date)
    # Save the plot data to the PPO objects
    pair = (ppos[0].ticker, ppos[1].ticker)
    sp = spread_mean.copy()
    cl = df_closes.copy(deep=True)
    cl.reset_index(inplace=True)
    cl = cl.to_json()
    dc = df_detrend.copy(deep=True)
    dc.reset_index(inplace=True)
    dc = dc.to_json()
    spread_analysis = {'pair': (ppos[0].ticker, ppos[1].ticker),
                       'spread_mean': sp,
                       'beta': beta,
                       'closes': cl,
                       'detrended_closes': dc
                       }
    ppos[0].spread_analysis[pair] = spread_analysis
    ppos[1].spread_analysis[pair] = spread_analysis

    # Plot the spread with mean line
    plt.plot(df_detrend['Spread'], marker='o', label=spread_name, color=spread_color)
    plt.plot(df_detrend['Mean_2std_a'], label='2std_a', color='green')
    plt.plot(df_detrend['Mean_1std_a'], label='1std_a', color='blue')
    plt.plot(df_detrend['Mean_1std_b'], label='1std_b', color='blue')
    plt.plot(df_detrend['Mean_2std_b'], label='2std_b', color='green')
    plt.axhline(spread_mean, color='red', linestyle='--', label='Mean Spread')
    plt.legend()
    if title is None:
        title = 'Spread Between Stock A and Stock B'
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel(spread_name)
    # Enable x, y grid lines
    plt.grid(True)
    plt.show()

    return plt, beta

# Import Libraries
import os.path
import numpy as np
import pandas as pd
import logging
import sys
import json
import dill
import pandas as pd
import matplotlib.pyplot as plt
import copy
from pricepredict import PricePredict
from datetime import datetime, timedelta

# Use an Object Cache to reduce the prep time for creating and loading the PricePredict objects.
if 'ObjCache' not in globals():
    global ObjCache
    ObjCache = {}

DirPPO = './ppo/'


def get_ppo(symbol: str, period: str):
    file_name_starts_with = symbol + '_' + period
    # Find all PPO files for the symbol in the PPO directory
    ppo_files = [f for f in os.listdir(DirPPO) if f.startswith(file_name_starts_with)]
    # Sort the files by date
    ppo_files.sort()
    # Get the latest PPO file
    ppo_file = ppo_files[-1]
    # Unpickle the PPO file using dill
    with open(DirPPO + ppo_file, 'rb') as f:
        ppo = PricePredict.unserialize(f)
    return ppo_file, ppo


def get_tradingpair_ppos(trading_pair: tuple):
    tp1_weekly_ppo_file, tp1_weekly_ppo = get_ppo(trading_pair[0], PricePredict.PeriodWeekly)
    tp1_daily_ppo_file, tp1_daily_ppo = get_ppo(trading_pair[0], PricePredict.PeriodDaily)
    tp2_weekly_ppo_file, tp2_weekly_ppo = get_ppo(trading_pair[1], PricePredict.PeriodWeekly)
    tp2_daily_ppo_file, tp2_daily_ppo = get_ppo(trading_pair[1], PricePredict.PeriodDaily)
    print(
        f'{trading_pair[0]} Weekly PPO: {tp1_weekly_ppo_file} {tp1_weekly_ppo}:[{round(getsize(tp1_weekly_ppo) / 1024 / 1024, 2)}]M')
    print(
        f'{trading_pair[0]} Daily PPO: {tp1_daily_ppo_file} {tp1_daily_ppo}:[{round(getsize(tp1_daily_ppo) / 1024 / 1024, 2)}]M')
    print(
        f'{trading_pair[1]} Weekly PPO: {tp2_weekly_ppo_file} {tp2_weekly_ppo}:[{round(getsize(tp2_weekly_ppo) / 1024 / 1024, 2)}]M')
    print(
        f'{trading_pair[1]} Daily PPO: {tp2_daily_ppo_file} {tp2_daily_ppo}:[{round(getsize(tp2_daily_ppo) / 1024 / 1024, 2)}]M')
    return tp1_weekly_ppo, tp1_daily_ppo, tp2_weekly_ppo, tp2_daily_ppo


def check_ppo_orig_data(ppo: PricePredict, msg: str = None):
    is_index_datetime = isinstance(ppo.orig_data.index, pd.DatetimeIndex)
    is_date_in_index = 'Date' in ppo.orig_data.index.names
    if msg is not None and (is_date_in_index is True or is_index_datetime is True):
        print(msg)
    if is_index_datetime is False:
        print(f"orig_data index is not a DatetimeIndex: {ppo.ticker} {ppo.period}")
    if is_date_in_index is False:
        print(f"orig_data index does not have a 'Date' column: {ppo.ticker} {ppo.period}")


def get_prop_ppos(trading_pair: tuple):
    global ObjCache

    model_dir = './models/'
    chart_dir = './charts/'
    preds_dir = './predictions/'

    tp1_weekly_ppo = PricePredict(ticker=trading_pair[0], period=PricePredict.PeriodWeekly,
                                  model_dir=model_dir, chart_dir=chart_dir, preds_dir=preds_dir)
    tp1_daily_ppo = PricePredict(ticker=trading_pair[0], period=PricePredict.PeriodDaily,
                                 model_dir=model_dir, chart_dir=chart_dir, preds_dir=preds_dir)
    tp2_weekly_ppo = PricePredict(ticker=trading_pair[1], period=PricePredict.PeriodWeekly,
                                  model_dir=model_dir, chart_dir=chart_dir, preds_dir=preds_dir)
    tp2_daily_ppo = PricePredict(ticker=trading_pair[1], period=PricePredict.PeriodDaily,
                                 model_dir=model_dir, chart_dir=chart_dir, preds_dir=preds_dir)

    # Train the models on 5 yeas of data...
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=5 * 400)
    end_date = end_dt.strftime('%Y-%m-%d')
    start_date = start_dt.strftime('%Y-%m-%d')

    print(f"ObjCache: {ObjCache.keys()}")

    # Load 2 years of data for the trading pair
    ppo_name = trading_pair[0] + '_weekly_ppo'
    if ppo_name not in ObjCache.keys():
        tp1_weekly_ppo.fetch_train_and_predict(tp1_weekly_ppo.ticker,
                                               start_date, end_date,
                                               start_date, end_date,
                                               period=PricePredict.PeriodWeekly,
                                               force_training=False,
                                               use_curr_model=True,
                                               save_model=False)
        check_ppo_orig_data(tp1_weekly_ppo, f"After Yahoo Fetch {trading_pair[0]} Weekly PPO")
        ObjCache[ppo_name] = tp1_weekly_ppo.serialize_me()
    else:
        tp1_weekly_ppo = PricePredict.unserialize(ObjCache[ppo_name])
        check_ppo_orig_data(tp1_weekly_ppo, f"After loading from ObjCache {trading_pair[0]} Weekly PPO")

    ppo_name = trading_pair[0] + '_daily_ppo'
    if ppo_name not in ObjCache.keys():
        tp1_daily_ppo.fetch_train_and_predict(tp1_daily_ppo.ticker,
                                              start_date, end_date,
                                              start_date, end_date,
                                              period=PricePredict.PeriodDaily,
                                              force_training=False,
                                              use_curr_model=True,
                                              save_model=False)
        check_ppo_orig_data(tp1_daily_ppo, f"After Yahoo Fetch {trading_pair[0]} Daily PPO")
        ObjCache[ppo_name] = tp1_daily_ppo.serialize_me()
    else:
        tp1_daily_ppo = PricePredict.unserialize(ObjCache[ppo_name])
        check_ppo_orig_data(tp1_daily_ppo, f"After loading from ObjCache {trading_pair[0]} Daily PPO")

    ppo_name = trading_pair[1] + '_weekly_ppo'
    if ppo_name not in ObjCache.keys():
        tp2_weekly_ppo.fetch_train_and_predict(tp2_weekly_ppo.ticker,
                                               start_date, end_date,
                                               start_date, end_date,
                                               period=PricePredict.PeriodWeekly,
                                               force_training=False,
                                               use_curr_model=True,
                                               save_model=False)
        check_ppo_orig_data(tp2_weekly_ppo, f"After Yahoo Fetch {trading_pair[1]} Weekly PPO")
        ObjCache[ppo_name] = tp2_weekly_ppo.serialize_me()
    else:
        tp2_weekly_ppo = PricePredict.unserialize(ObjCache[ppo_name])
        check_ppo_orig_data(tp2_weekly_ppo, f"After loading from ObjCache {trading_pair[1]} Weekly PPO")

    ppo_name = trading_pair[1] + '_daily_ppo'
    if ppo_name not in ObjCache.keys():
        tp2_daily_ppo.fetch_train_and_predict(tp2_daily_ppo.ticker,
                                              start_date, end_date,
                                              start_date, end_date,
                                              force_training=False,
                                              use_curr_model=True,
                                              save_model=False)
        check_ppo_orig_data(tp2_daily_ppo, f"After Yahoo Fetch {trading_pair[1]} Daily PPO")
        ObjCache[ppo_name] = tp2_daily_ppo.serialize_me()
    else:
        tp2_daily_ppo = PricePredict.unserialize(ObjCache[ppo_name])
        check_ppo_orig_data(tp2_daily_ppo, f"After loading from ObjCache {trading_pair[1]} Daily PPO")

    return tp1_weekly_ppo, tp1_daily_ppo, tp2_weekly_ppo, tp2_daily_ppo


def analyze_trading_pair(trading_pair: tuple):
    # Gather the Weekly and Daily PPOs for the trading pair from the ./ppo/ dir.
    # tp1_weekly_ppo, tp1_daily_ppo, tp2_weekly_ppo, tp2_daily_ppo = get_tradingpair_ppos(trading_pair)

    # Creates ppo objects and caches them to ObjCache.
    tp1_weekly_ppo, tp1_daily_ppo, tp2_weekly_ppo, tp2_daily_ppo = get_prop_ppos(trading_pair)

    # Plot the median & spread of the trading pair given the daily PPOs)
    # Plot the Weekly Spread using the Weekly calculated Beta
    plt, beta = plot_spread((tp1_weekly_ppo, tp2_weekly_ppo),
                            title=f"Weekly Spread [{trading_pair[0]} vs {trading_pair[1]}]",
                            spread_name='Weekly')
    print(f"Weekly Hedge Ratio: {beta}")
    # # Plot the Daily Spread, Using the Weekly Beta
    # plt, beta = plot_spread((tp1_daily_ppo, tp2_daily_ppo), beta, 60,
    #             title=f"Daily Spread [{trading_pair[0]} vs {trading_pair[1]}]",
    #             spread_name='Daily (Wkly Beta)', spread_color='grey')
    # print(f"Daily using Weekly Hedge Ratio: {beta}")
    # # Plot the Daily Spread, Using the Daily calculated Beta
    # plt, beta = plot_spread((tp1_daily_ppo, tp2_daily_ppo), None, 60,
    #                         title=f"Daily Spread [{trading_pair[0]} vs {trading_pair[1]}]",
    #                         spread_name='Daily', spread_color='orange')
    # print(f"Daily Hedge Ratio: {beta}")
    # plt, beta = plot_spread((tp1_daily_ppo, tp2_daily_ppo),
    #                         title=f"Daily[1:37] Spread [{trading_pair[0]} vs {trading_pair[1]}]",
    #                         spread_name='Daily [1:37]', spread_color='orange',
    #                         start_period=1, end_period=37)
    # print(f"Daily[1:37] Hedge Ratio {beta}")
    # plt, beta = plot_spread((tp1_daily_ppo, tp2_daily_ppo),
    #                         title=f"Daily[4/1/21 to 8/1/21] Spread [{trading_pair[0]} vs {trading_pair[1]}]",
    #                         spread_name='Daily [4/1/21 to 8/1/21]', spread_color='orange',
    #                         start_date='4/1/2021', end_date='7/30/2021')
    # print(f"Daily[4/1/21 to 8/1/21] Hedge Ratio {beta}")

    start_date = '2019-08-11 '
    end_date = '2019-10-10'
    plt, beta = plot_spread((tp1_daily_ppo, tp2_daily_ppo),
                            title=f"Daily[4/1/21 to 8/1/21] Spread [{trading_pair[0]} vs {trading_pair[1]}]",
                            spread_name='Daily [4/1/21 to 8/1/21]', spread_color='orange',
                            start_date=start_date, end_date=end_date)
    print(f"Daily[4/1/21 to 8/1/21] Hedge Ratio {beta}")

    return plt


if 'plt' in locals():
    plt.close()

plt = analyze_trading_pair(('QCOM', 'EURUSD=X'))

