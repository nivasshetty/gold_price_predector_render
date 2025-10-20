import yfinance as yf # type: ignore
import pandas as pd # type: ignore

# Gold Futures ticker symbol (USD)
ticker = "GC=F"  # Yahoo Finance symbol for Gold Futures

# Download data from 2010 to today
gold_data = yf.download(ticker, start="2010-01-01", end=None)  # end=None means up to today

# Display first few rows
print(gold_data.head())

# Save to CSV file
gold_data.to_csv("GOLD_prices_2010_to_today.csv")

print("\n✅ GOLD price data downloaded successfully!")
print("📁 Saved as 'GOLD_prices_2010_to_today.csv'")
