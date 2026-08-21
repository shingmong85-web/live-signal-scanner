import pandas as pd
import random

# -----------------------------
# DEMO 1-MINUTE MARKET DATA
# -----------------------------
price = 100.0
rows = []

for i in range(300):
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

# -----------------------------
# CREATE 5-MINUTE CANDLES
# -----------------------------
df["group"] = df.index // 5

df_5m = df.groupby("group").agg({
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last"
}).reset_index(drop=True)

# -----------------------------
# EMA
# -----------------------------
df_5m["EMA20"] = df_5m["close"].ewm(
    span=20, adjust=False
).mean()

df_5m["EMA50"] = df_5m["close"].ewm(
    span=50, adjust=False
).mean()

df_5m["EMA200"] = df_5m["close"].ewm(
    span=200, adjust=False
).mean()

# -----------------------------
# RSI
# -----------------------------
delta = df_5m["close"].diff()

gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()

rs = gain / loss.replace(0, 1e-10)

df_5m["RSI"] = 100 - (100 / (1 + rs))

# -----------------------------
# ANALYSIS
# -----------------------------
last = df_5m.iloc[-1]

if (
    last["EMA20"] > last["EMA50"]
    and last["EMA50"] > last["EMA200"]
    and last["RSI"] > 50
):
    result = "UP BIAS"

elif (
    last["EMA20"] < last["EMA50"]
    and last["EMA50"] < last["EMA200"]
    and last["RSI"] < 50
):
    result = "DOWN BIAS"

else:
    result = "NO SIGNAL"

# -----------------------------
# OUTPUT
# ----------------------------
