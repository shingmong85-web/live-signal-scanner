import requests
import pandas as pd

PAIR = "BTCUSDT"
INTERVAL = "1m"
LIMIT = 100

url = "https://api.binance.com/api/v3/klines"
params = {
    "symbol": PAIR,
    "interval": INTERVAL,
    "limit": LIMIT
}

data = requests.get(url, params=params, timeout=10).json()

df = pd.DataFrame(data, columns=[
    "time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "buy_volume", "buy_quote_volume", "ignore"
])

df["close"] = df["close"].astype(float)

df["EMA20"] = df["close"].ewm(span=20).mean()
df["EMA50"] = df["close"].ewm(span=50).mean()

last = df.iloc[-1]

if last["EMA20"] > last["EMA50"]:
    signal = "CALL / UP BIAS"
elif last["EMA20"] < last["EMA50"]:
    signal = "PUT / DOWN BIAS"
else:
    signal = "NO SIGNAL"

print("Pair:", PAIR)
print("Last Price:", last["close"])
print("EMA20:", round(last["EMA20"], 4))
print("EMA50:", round(last["EMA50"], 4))
print("Signal:", signal)
