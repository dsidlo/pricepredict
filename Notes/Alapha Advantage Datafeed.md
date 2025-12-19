# Alpha Advantage Datafeed

Add the [Alpha Advantage](https://www.alphavantage.co/) datafeed.

### Financial Instruments Available via Alpha Vantage Python API

The Alpha Vantage Python library (available via PyPI as `alpha_vantage`) is a wrapper for the Alpha Vantage REST API, providing access to a wide range of financial data through simple Python functions. It supports querying historical, real-time, and fundamental data for various instruments. Below is a categorized list of the primary financial instruments supported, based on the API's endpoints. These include equities, derivatives, currencies, and commodities. Note that while most data is free (with rate limits), premium features like real-time quotes or extended history may require a subscription.

#### 1. **Equities (Stocks, ETFs, and Mutual Funds)**
   - **Description**: Global stock market data, including time series (OHLCV: open, high, low, close, volume), quotes, and searches for over 100,000 symbols across major exchanges.
   - **Examples**:
     - US stocks: IBM, AAPL, MSFT.
     - International stocks: TSCO.LON (UK), RELIANCE.BSE (India), 600104.SHH (China).
     - ETFs: QQQ.
     - Mutual funds: Searchable via keyword (e.g., "Vanguard").
   - **Access via Python**: Use functions like `TimeSeries().get_daily(symbol='IBM')` or `get_quote_endpoint(symbol='AAPL')`.

#### 2. **Options**
   - **Description**: US equity options chains with real-time/historical data, including implied volatility (IV) and Greeks (delta, gamma, theta, vega, rho). Covers 15+ years of history.
   - **Examples**:
     - Options on stocks like IBM (e.g., contract: IBM270115C00390000 for specific strike/expiration).
   - **Access via Python**: Functions like `get_options_chain(symbol='IBM')` (premium for real-time).

#### 3. **Forex (Foreign Exchange - Physical Currencies)**
   - **Description**: Real-time and historical exchange rates for fiat currency pairs, with intraday, daily, weekly, and monthly time series.
   - **Examples**:
     - Pairs: EUR/USD, USD/JPY, GBP/EUR (specified as `from_currency` and `to_currency`).
   - **Access via Python**: `Forex().get_currency_exchange_rate(from_currency='EUR', to_currency='USD')` or `get_intraday(from_symbol='EUR', to_symbol='USD')`.

#### 4. **Cryptocurrencies**
   - **Description**: Digital currency prices paired with fiat currencies, including intraday (1-60 min intervals), daily, weekly, and monthly data.
   - **Examples**:
     - BTC/USD, ETH/EUR, LTC/BTC (specified as `symbol` and `market`, e.g., BTC paired with USD).
   - **Access via Python**: `CryptoCurrencies().get_digital_currency_daily(symbol='BTC', market='USD')` (premium for intraday).

#### 5. **Commodities**
   - **Description**: Historical prices for major physical commodities from sources like the U.S. Energy Information Administration. Fixed set of commodities (no custom symbols); data in daily/weekly/monthly intervals.
   - **Examples**:
     - Energy: WTI (crude oil), Brent (crude oil), NATURAL_GAS.
     - Metals: COPPER, ALUMINUM.
     - Agriculture: WHEAT, CORN, COTTON, SUGAR, COFFEE.
     - Index: ALL_COMMODITIES (global index).
   - **Access via Python**: Dedicated functions like `get_wti(function='WTI')` or `get_brent()`.

#### Additional Notes
- **Derived Data**: Technical indicators (e.g., SMA, EMA) and advanced analytics (e.g., correlations) can be applied to any of the above instruments (equities, forex, crypto).
- **Fundamental and Macro Data**: Not direct instruments but related overviews, earnings, economic indicators (e.g., CPI, GDP, Treasury yields), and sector performances (e.g., technology sector) for equities.
- **Limitations**: Instruments must use standard symbols; global coverage but US-focused for options and some indicators. Always obtain a free API key from alphavantage.co for authentication in Python code.

For full details and code examples, refer to the official documentation or the Python library's GitHub repository.

### Python Code Examples for Equities Using Alpha Vantage API

The `alpha_vantage` Python library provides straightforward functions for accessing equities data (stocks, ETFs, etc.). Below are practical code examples. 

**Prerequisites**:
- Install the library: `pip install alpha_vantage`
- Get a free API key from [alphavantage.co](https://www.alphavantage.co/support/#api-key).
- Replace `'YOUR_API_KEY'` with your actual key in the code.
- All examples use the `TimeSeries` module for historical data, `get_quote_endpoint` for real-time quotes, and `SectorPerformances` for sector overviews. Data is returned as Pandas DataFrames for easy manipulation.

#### 1. **Daily Time Series (Adjusted for Splits/Dividends)**
   Fetches daily OHLCV (Open, High, Low, Close, Volume) data for a stock.

   ```python
   from alpha_vantage.timeseries import TimeSeries
   import pandas as pd

   # Initialize the TimeSeries module with your API key
   ts = TimeSeries(key='YOUR_API_KEY', output_format='pandas')

   # Get daily adjusted time series for IBM
   data, meta_data = ts.get_daily_adjusted(symbol='IBM', outputsize='compact')  # 'compact' for last 100 days; 'full' for 20+ years

   # Display the most recent data
   print(data.head())
   # Output: DataFrame with columns like '1. open', '2. high', '3. low', '4. close', '5. adjusted close', '6. volume', etc.
   ```

#### 2. **Intraday Time Series (1-Minute Intervals)**
   Useful for short-term analysis; limited to recent data (premium for full history).

   ```python
   from alpha_vantage.timeseries import TimeSeries
   import pandas as pd

   ts = TimeSeries(key='YOUR_API_KEY', output_format='pandas')

   # Get intraday data for AAPL (1-min intervals, last 5 days)
   data, meta_data = ts.get_intraday(symbol='AAPL', interval='1min', outputsize='5')

   # Sort by timestamp and print recent rows
   data = data.sort_index()
   print(data.tail())
   # Output: DataFrame with timestamp index and OHLCV columns.
   ```

#### 3. **Real-Time Quote**
   Retrieves the latest price, volume, and other snapshot data for a stock.

   ```python
   from alpha_vantage.quote import Quote
   import pandas as pd

   q = Quote(key='YOUR_API_KEY', output_format='pandas')

   # Get quote for MSFT
   data, meta_data = q.get_quote_endpoint(symbol='MSFT')

   # Display the quote DataFrame (single row)
   print(data)
   # Output: Columns like '01. symbol', '02. open', '03. high', '04. low', '05. price', '09. change percent', etc.
   ```

#### 4. **Global Quote (for International Stocks)**
   Similar to real-time quote but supports non-US symbols (e.g., UK or Indian stocks).

   ```python
   from alpha_vantage.foreignexchange import ForeignExchange  # Note: Uses forex module for global, but equities via quote
   from alpha_vantage.quote import Quote
   import pandas as pd

   q = Quote(key='YOUR_API_KEY', output_format='pandas')

   # Get global quote for Reliance Industries (India: RELIANCE.BSE)
   data, meta_data = q.get_quote_endpoint(symbol='RELIANCE.BSE')

   print(data)
   # Output: Similar to above, with latest metrics.
   ```

#### 5. **Sector Performance Overview**
   Compares real-time performance across equity sectors (e.g., Technology, Energy).

   ```python
   from alpha_vantage.sectorperformance import SectorPerformances
   import pandas as pd

   sp = SectorPerformances(key='YOUR_API_KEY', output_format='pandas')

   # Get sector performances
   data, meta_data = sp.get_sector()

   # Display the DataFrame (sorted by change percent)
   data = data.sort_values('Rank A: Real-Time Performance', ascending=False)
   print(data)
   # Output: DataFrame with sectors like 'Technology', columns for 1D change %, year-high/low, etc.
   ```

#### 6. **Batch Stock Quotes (Multiple Symbols)**
   Efficiently fetch quotes for several stocks at once (up to 256 symbols).

   ```python
   from alpha_vantage.batchstockquotes import BatchStockQuotes
   import pandas as pd

   bsq = BatchStockQuotes(key='YOUR_API_KEY', output_format='pandas')

   # Get batch quotes for AAPL, GOOGL, TSLA
   symbols = ['AAPL', 'GOOGL', 'TSLA']
   data, meta_data = bsq.get_batch_stock_quotes(symbols)

   print(data)
   # Output: Multi-row DataFrame with quotes for each symbol.
   ```

#### Tips for Usage
- **Output Format**: Set `output_format='pandas'` for DataFrames (requires `pandas` installed) or `'json'` for raw dicts.
- **Rate Limits**: Free tier allows 5 calls/minute and 500/day. Use `outputsize='compact'` to stay efficient.
- **Error Handling**: Wrap calls in try-except for API errors (e.g., invalid symbol).
  ```python
  try:
      data, meta = ts.get_daily(symbol='INVALID')
  except ValueError as e:
      print(f"Error: {e}")
  ```
- **Visualization**: Pipe DataFrames to Matplotlib for plots, e.g., `data['4. close'].plot()`.

These examples cover core equities functionality. For more (e.g., fundamentals like earnings), check the library's [GitHub README](https://github.com/RomelTorres/alpha_vantage). If you need examples for other instruments or custom tweaks, let me know!

### Differences Between Alpha Vantage and yfinance Stock Data Structures

Alpha Vantage (via its Python wrapper `alpha_vantage`) and yfinance (Yahoo Finance's unofficial Python library) both provide historical stock data (e.g., OHLCV: Open, High, Low, Close, Volume) as pandas DataFrames, making them compatible with common analysis workflows. However, their structures differ in column naming, metadata handling, adjustment details, and multi-symbol support. These differences stem from the underlying APIs: Alpha Vantage's structured JSON responses versus Yahoo's scraped, normalized format. Below, I break down the key differences, focusing on daily historical data for equities (e.g., via `TimeSeries.get_daily_adjusted()` in `alpha_vantage` and `Ticker.history()` or `yf.download()` in yfinance).

#### 1. **Overall Return Format**
   - **Alpha Vantage**: Returns a tuple `(data, meta_data)`, where `data` is a pandas DataFrame and `meta_data` is a dictionary with query details (e.g., symbol, last refresh timestamp, time zone).
   - **yfinance**: Returns a single pandas DataFrame (no separate metadata object). Ticker-specific info (e.g., currency, sector) is accessible via `Ticker.info` as a dict, but not bundled with historical data.

#### 2. **Index (Timestamps)**
   - **Both**: DatetimeIndex (e.g., '2023-10-01' for daily data), sorted chronologically (oldest first in DataFrames). Alpha Vantage's library auto-converts string dates to datetime; yfinance uses timezone-aware datetimes (e.g., '2020-01-02 00:00:00-05:00').
   - **Difference**: Minor—yfinance includes timezone info by default; Alpha Vantage uses UTC or exchange-specific (e.g., US/Eastern) without timezone in the index unless specified.

#### 3. **Columns (OHLCV and Related Fields)**
   Alpha Vantage uses verbose, numbered prefixes matching its JSON API, while yfinance uses clean, intuitive names. For adjusted data:

   | Aspect                  | Alpha Vantage (e.g., `get_daily_adjusted`) | yfinance (e.g., `Ticker.history()`) |
   |-------------------------|--------------------------------------------|-------------------------------------|
   | **Core OHLCV Columns** | '1. open', '2. high', '3. low', '4. close', '5. adjusted close', '6. volume' | 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume' |
   | **Additional Fields**  | '7. dividend amount', '8. split coefficient' (per row; '0.00' or '1.0000' on non-event days) | 'Dividends', 'Stock Splits' (separate series via `ticker.dividends`/`ticker.splits`; not in main DF unless `actions=True`) |
   | **Data Types**         | Mostly strings (e.g., volume as str); requires conversion to float/int for analysis | Floats for prices/volumes; bool/int for splits/dividends |
   | **Naming Style**       | Numbered and dotted (e.g., '5. adjusted close') for JSON fidelity | Clean, capitalized (no prefixes) for direct pandas use |

   - **Key Difference**: Alpha Vantage's prefixes can feel clunky for plotting/calculations (e.g., `df['4. close'].plot()`), often needing renaming. yfinance is more pandas-native.

#### 4. **Adjustments for Splits/Dividends**
   - **Alpha Vantage**: Separate endpoint (`TIME_SERIES_DAILY_ADJUSTED`) includes explicit per-day dividend/split fields in the DataFrame. Raw daily endpoint lacks adjustments.
   - **yfinance**: Built-in 'Adj Close' column applies automatic adjustments to all prices (Open/High/Low/Close retroactively scaled). Full actions (dividends/splits) are optional add-ons.
   - **Difference**: yfinance simplifies adjusted analysis in one column; Alpha Vantage offers granular event data but requires the adjusted endpoint.

#### 5. **Multi-Symbol Support**
   - **Alpha Vantage**: No native multi-ticker historical DataFrame; use `BatchStockQuotes` for snapshots (returns DF with rows per symbol) or loop calls and concat DataFrames manually.
   - **yfinance**: Handles multiple tickers in one call (e.g., `yf.download(['AAPL', 'MSFT'])`), returning a DataFrame with MultiIndex columns (levels: Ticker, then OHLCV) or flattened with suffixes.
   - **Difference**: yfinance excels for batch historical data (e.g., side-by-side comparison); Alpha Vantage is better for single-symbol depth but less efficient for multiples.

#### 6. **Metadata and Query Info**
   | Field/Example          | Alpha Vantage (meta_data dict) | yfinance (via Ticker.info or inferred) |
   |------------------------|--------------------------------|----------------------------------------|
   | Symbol                | '2. Symbol': 'AAPL'            | Inferred from ticker; 'symbol' in info |
   | Last Update           | '3. Last Refreshed': '2023-10-01 16:00:00' | Inferred from index max; no explicit field |
   | Time Zone/Interval    | '5. Time Zone': 'US/Eastern', '4. Interval': 'daily' | Timezone in index; interval via param |
   | Coverage              | '4. Output Size': 'Compact' (100 days) or 'Full' (20+ years) | Via params (e.g., period='2y'); no explicit field |

   - **Difference**: Alpha Vantage's explicit metadata aids auditing; yfinance keeps it lightweight, relying on params/docs.

#### Additional Notes
- **Coverage and Granularity**: Both support daily/weekly/monthly and intraday (1min+), but Alpha Vantage caps free calls (5/min, 500/day) and offers more global symbols. yfinance has no limits but risks scraping changes.
- **Customization**: Both allow JSON/CSV raw output, but DataFrames are default. For analysis, yfinance often requires less preprocessing due to cleaner structure.
- **Potential Gotchas**: Alpha Vantage data may lag slightly (end-of-day); yfinance is near-real-time but unofficial (risk of breakage).

These structures make yfinance more user-friendly for quick prototyping, while Alpha Vantage suits API-focused or enterprise workflows. For code examples, see prior responses or official repos.


