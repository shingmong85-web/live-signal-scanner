import pandas as pd
import random

# Demo 1-minute candles
price = 100.0
rows = []

for i in range(200):
    change = random.uniform(-0.30, 0.30)

    open_price = price
    close_price = price + change
    high_price = max(open_price, close_price) + random.uniform(0, 0.10)
    low_price = min(open_price, close_price) - random.uniform(0, 0.10)

    rows.append({
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "close": close_price
    })

    price = close_price

df = pd.DataFrame(rows)

# Indicators
df["EMA20"] = df["close"].ewm(span=20, adjust=False).mean()
df["EMA50"] = df["close"].ewm(span=50, adjust=False).mean()

delta = df["close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()

rs = gain / loss.replace(0, 1e-10)
df["RSI"] = 100 - (100 / (1 + rs))

last = df.iloc[-1]

# Educational bias only
if last["EMA20"] > last["EMA50"] and last["RSI"] >= 50:
    bias = "UP BIAS"
elif last["EMA20"] < last["EMA50"] and last["RSI"] <= 50:
    bias = "DOWN BIAS"
else:
    bias = "NO SIGNAL"

print("================================")
print("DEMO MARKET ANALYZER")
print("================================")
print("Last Price:", round(last["close"], 5))
print("EMA20:", round(last["EMA20"], 5))
print("EMA50:", round(last["EMA50"], 5))
print("RSI:", round(last["RSI"], 2))
print("Result:", bias)
print("================================")
print("Paper/demo analysis only.")
