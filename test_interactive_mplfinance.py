import mplfinance as mpf
import yfinance as yf

# Download stock data from Yahoo Finance
symbol = 'AAPL'
df = yf.download(symbol, period='6mo')

# Add moving averages (MA) and volume to the plot
mpf.plot(df, type='candle', mav=(5, 20), volume=True)

