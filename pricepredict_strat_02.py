"""
File: pricepredict_strat_02.py

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
  - Sell the close at the end of the day.
  - Stopping out...
    - The default stop is .20 cents.
    - If the delta between the predicted close and the open is less than the stop,
      then the stop is the delta, or the pred_close.

  - Sell if the actual price on a long trade hits the predicted high price
  - Buy if the actual price on a short trade hits the predicted low price

Notes:
  - I was concerned with the adjusted predictions potentially slightly shifting the original prediction values
    once the actual price was determined. But, it does not look like that happens.
  - It is not necessary to perform incremental predictions to perform an
    trading analysis. Simply run a prediction for the entire test range.
    Then perform trading analysis on the entire test range.
"""
import sys

import pandas as pd

from pricepredict import PricePredict
from datetime import datetime, timedelta

# Explsing mplfinance globally so that interactive plotting occurs...
# import mplfinance

# Tickers to analyze
all_tickers = [
"AAPL","ABT","ACN","ADBE","ADM","ADP","AIG","ALKS","ALL","AMGN","AMZN","AON","APD","APTV","AVGO","AXON","AXP","BA","BAC","BAX","BKNG","BLK","BRK-B","CAT","CB","CCI","CI","CL","CLDX","CLS","CLSK","CMCSA","CNC","CRM","CRSP","CSCO","CSX","CVX","DHR","DIS","DUK","ECL","EGIO","EL","EMR","EOG","ERIC","EXAS","EXEL","FOLD","FXI","FXP","GD","GE","GILD","GLPG","GOOG","GOOGL","HAE","HD","HON","IART","IBM","IMAX","INTC","INTU","ISRG","ITW","JAZZ","JD","JNJ","JPM","KO","LIN","LIVN","LLY","LMT","LOW","LRCX","LSCC","MA","MASI","MCD","MDLZ","MDT","META","MMM","MO","MRK","MRVL","MSFT","NEE","NFLX","NKE","NNDM","NOW","NVCR","NVDA","ORCL","PACB","PEP","PFE","PG","PLD","PNC","PSA","PTCT","PYPL","QCOM","RTX","SBUX","SCHW","SEDG","SO","SPGI","SYK","SYY","T","TMO","TRV","TSLA","TXN","UCTT","UNH","UPS","USB","V","VRTX","VZ","WM","WMT","WOLF","XNCR","XOM","ZM","GC=F","^DJI","ANTM.JK","^GSPC","^IXIC","^N225","^XAX","XAB=F","XAE=F","XAF=F","XAI=F","XAK=F","XAU=F","EURUSD=X","GBPUSD=X","JPYUSD=X","000001.SS"
]

all_tickers = [
"AAPL","ABT","ACN","ADBE","ADM","ADP","AIG","ALKS","ALL","AMGN","AMZN"
]

# Load the strategy_results_feedback.csv file into a dataframe
df_results_feedback = pd.read_csv('strategy_results_feedback.csv')

def main():
    results_table = []
    for ticker in all_tickers:
        print(f"===> Backtesting strategy for {ticker}...")
        results = backtest_strategy(ticker)
        results_table.append(results)
        pass

    df_results = pd.DataFrame(results_table, columns=['Ticker', 'StartDate', 'Days', 'Gains', 'Loss', 'Profit', 'Winning Trades', 'Losing Trades',
                                                      # Lost longs/short gains because stopped out
                                                      'Cnt Lost Long Gains', 'Avg Lost Long Gains',
                                                      'Cnt Lost Short Gains', 'Avg Lost Short Gains',
                                                      # Price near high/low stats (For Near Predicted Highs/Lows optimization
                                                      'Cnt Near High', 'Min Near High', 'Max Near High', 'Avg Near High',
                                                      'Cnt Near Low', 'Min Near Low', 'Max Near Low', 'Avg Near Low',
                                                      # Long-Low/Short-High stats (For trailing stop optimization)
                                                      'Cnt Long Low', 'Min Long Low', 'Max Long Low', 'Avg Long Low',
                                                      'Cnt Short High', 'Min Short High', 'Max Short High', 'Avg Short High'])
    # Write the results to a CSV file
    df_results.to_csv('strategy_results.csv', index=False)


def backtest_strategy(ticker):
    """
     {past}------------------------------------{present}
     |...train_days...|...test_days...|...pred_days...|
    """
    test_days = 90
    pred_days = 365 * 3 + 15
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
    pp = PricePredict(ticker, period=PricePredict.PeriodDaily)
    pp.cache_training_data(ticker, train_start_date_str, train_end_date_str, pp.period)
    pp.cache_prediction_data(ticker, pred_start_date_str, pred_end_date_str, pp.period)
    pp.cached_train_predict_report()

    # print(f"===> Plotting predictions for {pp.ticker}...")
    # plot_predictions(pp, pred_end_date, show_plot=True)

    return strategy_analysis(pp)


def plot_predictions(pp, end_date, show_plot=False):
    # View the test prediction results
    title = f"test_train_model: {pp.ticker} -- Period {pp.period} {end_date}"
    pred_close = pp.adj_pred_close
    pred_high = pp.adj_pred_high
    pred_low = pp.adj_pred_low
    # close = list(pp.orig_data.iloc[15:len(pred_close)+13, 4])
    close = list(pp.orig_data.iloc[:len(pred_close)-1, 4])
    pp.gen_prediction_chart(save_plot=True, show_plot=show_plot, last_candles=800)

def strategy_analysis(pp):
    pred_close = pp.adj_pred_close
    pred_high = pp.adj_pred_high
    pred_low = pp.adj_pred_low
    # close = list(pp.orig_data.iloc[15:len(pred_close)+13, 4])
    dates = list(pp.date_data[:len(pred_close)-1])
    open = list(pp.orig_data.iloc[:len(pred_close)-1, 0])
    high = list(pp.orig_data.iloc[:len(pred_close)-1, 1])
    low = list(pp.orig_data.iloc[:len(pred_close)-1, 2])
    close = list(pp.orig_data.iloc[:len(pred_close)-1, 3])

    min_price_near_high = sys.maxsize
    max_price_near_low = 0
    tot_price_near_high = 0
    cnt_price_near_high = 0
    avg_price_near_high = 0

    min_price_near_low = sys.maxsize
    max_price_near_high = 0
    tot_price_near_low = 0
    cnt_price_near_low = 0
    avg_price_near_low = 0

    min_long_low = sys.maxsize
    max_long_low = 0
    tot_long_low = 0
    cnt_long_low = 0
    avg_long_low = 0

    min_short_high = sys.maxsize
    max_short_high = 0
    tot_short_high = 0
    cnt_short_high = 0
    avg_short_high = 0

    lost_long_gains = 0
    cnt_lost_long_gains = 0
    avg_lost_long_gains = 0

    lost_short_gains = 0
    cnt_lost_short_gains = 0
    avg_lost_short_gains = 0

    df_dates = pd.DataFrame(dates)
    df_open = pd.DataFrame(open)
    df_high = pd.DataFrame(high)
    df_low = pd.DataFrame(low)
    df_close = pd.DataFrame(close)
    df_pred_close = pd.DataFrame(pred_close)
    df_pred_high = pd.DataFrame(pred_high)
    df_pred_low = pd.DataFrame(pred_low)
    df_tbl = pd.concat([df_dates, df_open, df_high, df_low, df_close, df_pred_close, df_pred_high, df_pred_low], axis=1)
    df_tbl.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Pred Close', 'Pred High', 'Pred Low']
    # Find row where Open > pred_close and Open < Close
    df_rows = df_tbl[(df_tbl['Open'] > df_tbl['Pred Close']) & (df_tbl['Open'] < df_tbl['Close'])]
    pass

    # Strategy:
    gains = 0
    loss = 0
    stop_loss = .25
    cnt_winning_trades = 0
    cnt_losing_trades = 0

    min_near_high = df_results_feedback.loc[df_results_feedback['Ticker'] == pp.ticker, 'Min Near High'].values[0]
    avg__near_high = df_results_feedback.loc[df_results_feedback['Ticker'] == pp.ticker, 'Avg Near High'].values[0]
    rng_near_high = (avg__near_high - min_near_high) / 9

    min_long_low = df_results_feedback.loc[df_results_feedback['Ticker'] == pp.ticker, 'Min Long Low'].values[0]
    avg_long_low = df_results_feedback.loc[df_results_feedback['Ticker'] == pp.ticker, 'Avg Long Low'].values[0]
    rng_long_low = (avg_long_low - min_long_low) / 9

    for i in range(len(close)):

        near_high_go_short = False
        near_low_go_long = False

        if abs(open[i] > pred_high[i]) <= rng_near_high:
            near_high_go_short = True
            print(f"Near High Go Long: {abs(open[i] - pred_high[i])} <= {rng_near_high}")
        if abs(open[i] < pred_low[i]) <= rng_long_low:
            near_low_go_long = True
            print(f"Near Low Go Short: {abs(open[i] - pred_low[i])} <= {rng_long_low}")

        if open[i] > pred_close[i]:
            print(f"Buy at {open[i]} on {dates[i]}")
            if (open[i] < close[i]) or near_low_go_long:
                # The bar when long...
                # Tweek the stop loss...
                # stop_loss = df_results_feedback.loc[df_results_feedback['Ticker'] == pp.ticker, 'Avg Long Low'].values[0]
                stop_loss += stop_loss * .25
                dist2pred_close = open[i] - pred_close[i]
                if dist2pred_close < stop_loss:
                    stop_loss = dist2pred_close
                if open[i] - low[i] > stop_loss:
                    # Our stop loss was hit...
                    loss -= stop_loss
                    cnt_losing_trades += 1
                    print(f"Loss: {loss}")

                    # Did the stopp trade ultimately become a winning trade?
                    if open[i] < close[i]:
                        gain = close[i] - open[i]
                        lost_long_gains += gain
                        cnt_lost_long_gains += 1
                        print(f"Lost long: {gain}, stopped out at {open[i] - stop_loss}")
                else:
                    # Our Stop was not hit...
                    gain = close[i] - open[i]
                    gains += gain
                    cnt_winning_trades += 1
                    print(f"Profit: {gain}")

                # Get stats on the lows of the long bars...
                low_delta = open[i] - low[i]
                min_long_low = min(min_long_low, low_delta)
                max_long_low = max(max_long_low, low_delta)
                tot_long_low += low_delta
                cnt_long_low += 1
            else:
                loss -= stop_loss
                print(f"Loss: {loss}")


        elif open[i] < pred_close[i]:
            print(f"Sell at {open[i]} on {dates[i]}")
            if open[i] > close[i] or near_high_go_short:
                # The bar when short...
                # Tweek the stop loss...
                # stop_loss = df_results_feedback.loc[df_results_feedback['Ticker'] == pp.ticker, 'Avg Short High'].values[0]
                stop_loss += stop_loss * .25
                dist2pred_close = pred_close[i] - open[i]
                if dist2pred_close < stop_loss:
                    stop_loss = dist2pred_close
                if high[i] - open[i] > stop_loss:
                    # Our stop loss was hit...
                    loss -= stop_loss
                    cnt_losing_trades += 1
                    print(f"Loss: {loss}")

                    # Did the stopp trade ultimately become a winning trade?
                    if open[i] > close[i]:
                        gain = open[i] - close[i]
                        lost_short_gains += gain
                        cnt_lost_short_gains += 1
                        print(f"Lost short: {gain}, stopped out at {open[i] + stop_loss}")
                else:
                    # Our Stop was not hit...
                    gain = open[i] - close[i]
                    gains += gain
                    cnt_winning_trades += 1
                    print(f"Profit: {gain}")

                # Get stats on the highs of the short bars...
                high_delta = high[i] - open[i]
                min_short_high = min(min_short_high, high_delta)
                max_short_high = max(max_short_high, high_delta)
                tot_short_high += high_delta
                cnt_short_high += 1
            else:
                loss -= stop_loss
                print(f"Loss: {loss}")

        # Gather price near high stats (for up-bars, above pred_close)...
        if pred_close[i] < open[i] and open[i] < close[i]:
            price_near_high = abs(high[i] - pred_high[i])
            min_price_near_high = min(min_price_near_high, price_near_high)
            max_price_near_high = max(max_price_near_high, price_near_high)
            tot_price_near_high += price_near_high
            cnt_price_near_high += 1

        # Gather price near low stats (for down-bars, below pred_close)...
        if pred_close[i] > open[i] and open[i] > close[i]:
            price_near_low = abs(pred_low[i] - low[i])
            min_price_near_low = min(min_price_near_low, price_near_low)
            max_price_near_low = max(max_price_near_low, price_near_low)
            tot_price_near_low += price_near_low
            cnt_price_near_low += 1

    if cnt_price_near_high > 0:
        avg_price_near_high = tot_price_near_high / cnt_price_near_high
    if cnt_price_near_low > 0:
        avg_price_near_low = tot_price_near_low / cnt_price_near_low
    if cnt_long_low > 0:
        avg_long_low = tot_long_low / cnt_long_low
    if cnt_short_high > 0:
        avg_short_high = tot_short_high / cnt_short_high
    if cnt_lost_long_gains > 0:
        avg_lost_long_gains = lost_long_gains / cnt_lost_long_gains
    if cnt_lost_short_gains > 0:
        avg_lost_short_gains = lost_short_gains / cnt_lost_short_gains

    print(f"\nGains: {gains:.2f}  Loss: {loss:.2f} Profit: {gains + loss:.2f}")
    print(f"\nPrice Near High Stats [{cnt_price_near_high}]:")
    print(f"   Min: {min_price_near_high}")
    print(f"   Max: {max_price_near_high}")
    print(f"   Avg: {avg_price_near_high}")
    print(f"\nPrice Near Low Stats [{cnt_price_near_low}]:")
    print(f"   Min: {min_price_near_low}")
    print(f"   Max: {max_price_near_low}")
    print(f"   Avg: {avg_price_near_low}")

    return (pp.ticker, dates[0], len(close), gains, loss, gains + loss, cnt_winning_trades, cnt_losing_trades,
            # Lost longs/short gains because stopped out
            cnt_lost_long_gains, avg_lost_long_gains, cnt_lost_short_gains, avg_lost_short_gains,
            # Price near high/low stats (For Near Predicted Highs/Lows optimization
            cnt_price_near_high, min_price_near_high, max_price_near_high, avg_price_near_high,
            cnt_price_near_low, min_price_near_low, max_price_near_low, avg_price_near_low,
            # Long-Low/Short-High stats (For trailing stop optimization)
            cnt_long_low, min_long_low, max_long_low, avg_long_low,
            cnt_short_high, min_short_high, max_short_high, avg_short_high)




main()
