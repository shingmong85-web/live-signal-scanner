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

response = requests.get(url, params=params, timeout=10)
response.raise_for_status()

data = response.json()

df = pd.DataFrame(data, columns=[
    "time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "buy_volume",
    "buy_quote_volume",
    "ignore"
])

df["close"] = pd.to_numeric(df["close"])

df["EMA20"] = df["close"].ewm(span=20, adjust=False).mean()
df["EMA50"] = df["close"].ewm(span=50, adjust=False).mean()

last = df.iloc[-1]

if last["EMA20"] > last["EMA50"]:
    signal = "CALL / UP BIAS"
elif last["EMA20"] < last["EMA50"]:
    signal = "PUT / DOWN BIAS"
else:
    signal = "NO SIGNAL"

print("================================")
print("LIVE SIGNAL SCANNER")
print("================================")
print("Pair:", PAIR)
print("Last Price:", last["close"])
print("EMA20:", round(last["EMA20"], 4))
print("EMA50:", round(last["EMA50"], 4))
print("Signal:", signal)
print("================================")
